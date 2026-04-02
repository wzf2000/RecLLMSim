"""
Self-distillation 轨迹采集：以已有 SFT 模型作为 teacher，在 context 中注入 gold 答案，
要求 teacher 仅生成分析文本（analysis），raw_content 由代码用 gold 值组装，
从根本上消除 gold mismatch 问题。

Teacher 看到：   完整 context  +  gold_score / gold_reason（已告知）  →  只输出分析文本
Student 学到：   完整 context（无 gold 提示）  →  预测正确答案 + 推理轨迹

旧方案的 mismatch 根因：teacher 仍执行"预测"任务，先验倾向覆盖 gold 提示；
新方案彻底分离两件事：teacher 负责"解释"，代码负责"组装正确 JSON"。

输出 JSONL 格式与 collect_api_model_traces.py 完全兼容，可直接传给 sft_from_traces.py。

用法：
    python collect_self_distill_traces.py \\
      --sft_checkpoint ./ckpts/sft_qwen3_from_gpt5_correct \\
      --base_model_name Qwen/Qwen3-8B \\
      --output_jsonl ./outputs/self_distill_traces.jsonl \\
      --data_split train \\
      --num_samples_per_prompt 1
"""

import json
import os
import re
from argparse import ArgumentParser
from typing import Any

import torch
from loguru import logger
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer

from .collect_api import get_rows_from_split
from lib.satisfaction_constants import get_reason_to_id
from .sft import (
    QWEN3_THINK_BEGIN,
    QWEN3_THINK_END,
    split_history_turns,
)


# ---------------------------------------------------------------------------
# Teacher prompt：告知 gold 答案，只要求生成解释性分析文本
# ---------------------------------------------------------------------------

def build_teacher_prompt(
    persona: str,
    task_context: str,
    history_text: str,
    assistant_reply: str,
    gold_score: int,
    gold_reason: str,
) -> str:
    """
    与旧版的关键区别：不再要求 teacher 执行预测任务，改为”解释已知答案”。
    teacher 的先验预测倾向不再干扰输出，从根本上消除 gold mismatch。
    """
    return (
        "你是一名会进行细粒度对话质量分析的评估员。\n\n"
        f"用户画像：{persona}\n\n"
        f"任务背景：{task_context}\n\n"
        f"最近对话历史：{history_text}\n\n"
        f"当前助手回复：{assistant_reply}\n\n"
        f"已知该场景下用户满意度评分为 {gold_score} 分（1-5），原因标签为「{gold_reason}」。\n\n"
        "请结合以上信息，深入推理并解释为什么该场景对应此满意度分数和原因标签。\n"
        "请直接输出你的详细分析文本，无需输出 JSON 或评分。"
    )


# ---------------------------------------------------------------------------
# Tokenize / 生成工具
# ---------------------------------------------------------------------------

def build_teacher_source_text(
    tokenizer: PreTrainedTokenizer,
    prompt: str,
    enable_thinking: bool = True,
) -> str:
    messages = [
        {"role": "system", "content": "You are a skilled conversational analyst."},
        {"role": "user", "content": prompt},
    ]
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(  # type: ignore
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
        except TypeError:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
    return f"System: You are a skilled conversational analyst.\nUser: {prompt}\nAssistant:"


def extract_reasoning_content(text: str) -> str:
    """从 <think>...</think> 中提取推理内容。"""
    start_idx = text.find(QWEN3_THINK_BEGIN)
    end_idx = text.rfind(QWEN3_THINK_END)
    if start_idx >= 0 and end_idx > start_idx:
        return text[start_idx + len(QWEN3_THINK_BEGIN):end_idx].strip()
    return ""


def extract_visible_content(text: str) -> str:
    """提取 </think> 之后的可见段（即 JSON 输出）。"""
    end_idx = text.rfind(QWEN3_THINK_END)
    if end_idx >= 0:
        return text[end_idx + len(QWEN3_THINK_END):].strip()
    return text.strip()


# ---------------------------------------------------------------------------
# 已完成样本 ID（断点续写）
# ---------------------------------------------------------------------------

def load_finished_ids(output_jsonl: str) -> set[str]:
    finished: set[str] = set()
    if not os.path.exists(output_jsonl):
        return finished
    with open(output_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                finished.add(str(obj["sample_id"]))
            except Exception:
                continue
    return finished


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

_SPECIAL_TOKEN_PATTERNS = re.compile(r"<\|[^|]+\|>")


def _clean_special_tokens(text: str) -> str:
    """去除 <|im_end|>、<|endoftext|> 等特殊 token 以及首尾空白。"""
    return _SPECIAL_TOKEN_PATTERNS.sub("", text).strip()


@torch.inference_mode()
def collect_self_distill_traces(
    sft_checkpoint: str,
    base_model_name: str,
    output_jsonl: str,
    data_split: str = "train",
    split_seed: int = 42,
    max_history_turns: int = 5,
    max_length: int = 2048,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    num_samples_per_prompt: int = 1,
    enable_thinking: bool = True,
    min_reasoning_tokens: int = 20,
    min_analysis_chars: int = 30,
    limit: int | None = None,
) -> None:
    """
    Parameters
    ----------
    num_samples_per_prompt : int
        每条样本采样的推理轨迹数量。>1 时以 temperature 采样，所有通过过滤的轨迹均写入。
    min_reasoning_tokens : int
        think 块的最小 token 数，低于此阈值的轨迹丢弃（仅 enable_thinking=True 时生效）。
    min_analysis_chars : int
        分析文本的最小字符数，低于此阈值的轨迹丢弃（防止退化为空输出）。
    enable_thinking : bool
        是否开启 Qwen3 think 模式（建议保持 True 以获得推理链）。
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info(f"Loading base model: {base_model_name}")
    base = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        trust_remote_code=True,
        dtype=torch.bfloat16 if device == "cuda" else torch.float32,
    )
    logger.info(f"Loading SFT adapter: {sft_checkpoint}")
    model = PeftModel.from_pretrained(base, sft_checkpoint)
    model.to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(sft_checkpoint, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    raw_rows = get_rows_from_split(split=data_split, split_seed=split_seed)
    if limit is not None:
        raw_rows = raw_rows[:limit]

    os.makedirs(os.path.dirname(output_jsonl) or ".", exist_ok=True)
    finished_ids = load_finished_ids(output_jsonl)
    logger.info(f"Total rows: {len(raw_rows)}, already finished: {len(finished_ids)}")

    n_written = 0
    n_skipped_short_reasoning = 0
    n_skipped_short_analysis = 0

    for idx, row in enumerate(pbar := tqdm(raw_rows, desc="self-distill")):
        pbar.set_postfix(
            written=n_written,
            s_reason=n_skipped_short_reasoning,
            s_analysis=n_skipped_short_analysis,
        )
        persona = str(row.get("persona", "") or "")
        task_context = str(row.get("task_context", "") or "")
        assistant_reply = str(row.get("assistant_reply", "") or "")
        gold_score = int(row["gold_score"])
        gold_reason = str(row["gold_reason"])

        turns = split_history_turns(str(row.get("history", "") or ""))
        kept_turns = turns[-max_history_turns:] if len(turns) > max_history_turns else turns
        history_text = "\n".join(kept_turns)

        teacher_prompt = build_teacher_prompt(
            persona, task_context, history_text, assistant_reply, gold_score, gold_reason
        )
        source_text = build_teacher_source_text(tokenizer, teacher_prompt, enable_thinking=enable_thinking)
        enc = tokenizer(
            source_text,
            return_tensors="pt",
            add_special_tokens=False,
            truncation=True,
            max_length=max_length - max_new_tokens,
        )
        enc = {k: v.to(device) for k, v in enc.items()}

        do_sample = temperature > 0
        gen_kwargs: dict[str, Any] = dict(
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        if do_sample:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["num_return_sequences"] = num_samples_per_prompt

        out_ids = model.generate(**enc, **gen_kwargs)
        input_len = enc["input_ids"].shape[1]

        for seq_idx in range(out_ids.shape[0]):
            unique_id = f"{idx}_{seq_idx}"
            if unique_id in finished_ids:
                continue

            gen_tokens = out_ids[seq_idx][input_len:]
            full_text = tokenizer.decode(gen_tokens, skip_special_tokens=False)

            reasoning_content = extract_reasoning_content(full_text)

            # teacher 的可见输出是纯分析文本（非 JSON），直接清洗后作为 analysis
            analysis = _clean_special_tokens(extract_visible_content(full_text))
            # 若没有 think 块（enable_thinking=False 或模型未生成），全文即 analysis
            if not analysis and not reasoning_content:
                analysis = _clean_special_tokens(full_text)

            # 过滤：think 块过短（enable_thinking 时）
            if enable_thinking:
                reasoning_token_count = len(
                    tokenizer(reasoning_content, add_special_tokens=False)["input_ids"]
                )
                if reasoning_token_count < min_reasoning_tokens:
                    n_skipped_short_reasoning += 1
                    continue

            # 过滤：分析文本过短
            if len(analysis) < min_analysis_chars:
                n_skipped_short_analysis += 1
                continue

            # raw_content 由代码组装，gold 值写死，不依赖模型输出
            raw_content = json.dumps(
                {"classification": gold_score, "reason": gold_reason, "analysis": analysis},
                ensure_ascii=False,
            )

            record: dict[str, Any] = {
                "sample_id": unique_id,
                "model": f"{base_model_name}+{os.path.basename(sft_checkpoint.rstrip('/'))}",
                "prompt": teacher_prompt,
                "prediction": gold_score,
                "reason_prediction": gold_reason,
                "gold_score": gold_score,
                "gold_reason": gold_reason,
                "analysis": analysis,
                "reasoning_content": reasoning_content,
                "raw_content": raw_content,
                "meta": {
                    "user": row.get("user", "unknown"),
                    "persona": persona,
                    "task_context": task_context,
                    "history": row.get("history", ""),
                    "assistant_reply": assistant_reply,
                },
            }
            with open(output_jsonl, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            n_written += 1

    logger.info(
        f"Done. written={n_written}, "
        f"skipped(reasoning_too_short)={n_skipped_short_reasoning}, "
        f"skipped(analysis_too_short)={n_skipped_short_analysis}"
    )

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = ArgumentParser(description="Self-distillation trace collection")
    p.add_argument("--sft_checkpoint", type=str, required=True, help="SFT 输出目录（含 LoRA adapter）")
    p.add_argument("--base_model_name", type=str, default="Qwen/Qwen3-8B")
    p.add_argument(
        "--output_jsonl",
        type=str,
        default="./outputs/self_distill_traces.jsonl",
    )
    p.add_argument(
        "--data_split",
        type=str,
        default="train",
        choices=["train", "valid", "val", "test", "all"],
    )
    p.add_argument("--split_seed", type=int, default=42)
    p.add_argument("--max_history_turns", type=int, default=5)
    p.add_argument("--max_length", type=int, default=2048)
    p.add_argument("--max_new_tokens", type=int, default=512)
    p.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="采样温度，0 表示贪心解码",
    )
    p.add_argument(
        "--num_samples_per_prompt",
        type=int,
        default=1,
        help="每条样本采样的轨迹数量，>1 时产生多条轨迹（数据增强）",
    )
    p.add_argument(
        "--no_thinking",
        action="store_true",
        help="禁用 Qwen3 think 模式（不推荐，会导致 reasoning_content 为空）",
    )
    p.add_argument(
        "--min_reasoning_tokens",
        type=int,
        default=20,
        help="think 块的最小 token 数，低于此的轨迹丢弃（enable_thinking=True 时生效）",
    )
    p.add_argument(
        "--min_analysis_chars",
        type=int,
        default=30,
        help="分析文本的最小字符数，低于此的轨迹丢弃",
    )
    p.add_argument("--limit", type=int, default=None, help="仅处理前 N 条，调试用")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    collect_self_distill_traces(
        sft_checkpoint=args.sft_checkpoint,
        base_model_name=args.base_model_name,
        output_jsonl=args.output_jsonl,
        data_split=args.data_split,
        split_seed=args.split_seed,
        max_history_turns=args.max_history_turns,
        max_length=args.max_length,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        num_samples_per_prompt=args.num_samples_per_prompt,
        enable_thinking=not args.no_thinking,
        min_reasoning_tokens=args.min_reasoning_tokens,
        min_analysis_chars=args.min_analysis_chars,
        limit=args.limit,
    )
