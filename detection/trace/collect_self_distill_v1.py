"""
Self-distillation 轨迹采集：以已有 SFT 模型作为 teacher，在 context 中注入 gold 答案，
要求 teacher 生成推理轨迹；student 再用这些轨迹做 SFT，无需依赖 API 闭源模型。

Teacher 看到：   完整 context  +  gold_score / gold_reason 提示  →  生成 think 推理 + JSON
Student 学到：   完整 context（无 gold 提示）  →  预测正确答案 + 推理轨迹

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

from trace.collect_api import get_rows_from_split
from lib.satisfaction_constants import get_reason_to_id
from trace.sft import (
    QWEN3_THINK_BEGIN,
    QWEN3_THINK_END,
    parse_model_json,
    split_history_turns,
)


# ---------------------------------------------------------------------------
# Teacher prompt：在标准 prompt 末尾注入 gold 提示
# ---------------------------------------------------------------------------

def build_teacher_prompt(
    persona: str,
    task_context: str,
    history_text: str,
    assistant_reply: str,
    gold_score: int,
    gold_reason: str,
) -> str:
    reason_labels = list(get_reason_to_id().keys())
    reason_labels_text = "、".join(reason_labels)
    return (
        "你是一名会进行细粒度对话质量分析的评估员。\n"
        "请基于给定信息先进行推理，再同时预测：\n"
        "1) 当前用户对助手回复的满意度分数（1-5）\n"
        "2) 潜在原因（必须从给定标签中选择，包含“满意”）\n\n"
        f"用户画像：{persona}\n\n"
        f"任务背景：{task_context}\n\n"
        f"最近对话历史：{history_text}\n\n"
        f"当前助手回复：{assistant_reply}\n\n"
        f"可选原因标签：{reason_labels_text}\n\n"
        f"（教学提示：本题正确答案为 classification={gold_score}，reason={gold_reason}。"
        "请在推理过程中充分理解并解释该判断的依据，最终输出的 JSON 须与提示完全一致。）\n\n"
        "请严格输出一个 JSON 对象，不要输出其他内容：\n"
        "{\n"
        '  "classification": 1-5中的整数,\n'
        '  "reason": "从可选原因标签中选择一个",\n'
        '  "analysis": "你的详细推理过程"\n'
        "}\n"
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
    limit: int | None = None,
) -> None:
    """
    Parameters
    ----------
    num_samples_per_prompt : int
        每条样本采样的推理轨迹数量。>1 时以 temperature 采样，所有通过过滤的轨迹均写入。
    min_reasoning_tokens : int
        推理内容（think 块）的最小 token 数，低于此阈值的轨迹丢弃。
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

    valid_reasons = set(get_reason_to_id().keys())
    n_written = 0
    n_skipped_parse = 0
    n_skipped_mismatch = 0
    n_skipped_short = 0
    n_analysis_empty = 0

    for idx, row in enumerate(pbar := tqdm(raw_rows, desc="self-distill")):
        pbar.set_postfix(written=n_written, s_parse=n_skipped_parse, s_mismatch=n_skipped_mismatch, s_short=n_skipped_short, empty=n_analysis_empty)
        sample_base_id = str(idx)
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

        do_sample = temperature > 0 and num_samples_per_prompt >= 1
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
        # out_ids shape: (num_samples, seq_len)
        input_len = enc["input_ids"].shape[1]

        sample_count = 0
        for seq_idx in range(out_ids.shape[0]):
            unique_id = f"{idx}_{seq_idx}"
            if unique_id in finished_ids:
                continue

            gen_tokens = out_ids[seq_idx][input_len:]
            full_text = tokenizer.decode(gen_tokens, skip_special_tokens=False)

            reasoning_content = extract_reasoning_content(full_text)
            visible_text = extract_visible_content(full_text)

            pred_score, pred_reason = parse_model_json(visible_text)
            if pred_score is None:
                n_skipped_parse += 1
                continue
            if pred_score != gold_score or (pred_reason or "").strip() != gold_reason.strip():
                n_skipped_mismatch += 1
                continue

            # 推理长度过滤
            reasoning_token_count = len(tokenizer(reasoning_content, add_special_tokens=False)["input_ids"])
            if reasoning_token_count < min_reasoning_tokens:
                n_skipped_short += 1
                continue

            # 提取 analysis 字段（从 JSON 中）
            try:
                visible_clean = visible_text.strip()
                if visible_clean.startswith("```"):
                    lines = visible_clean.split("\n")
                    visible_clean = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
                # 去除<|im_end|>后的内容
                visible_clean = visible_clean.split("<|im_end|>")[0]
                # 去除<|endoftext|>
                visible_clean = visible_clean.split("<|endoftext|>")[0]
                obj = json.loads(visible_clean)
                analysis = str(obj.get("analysis", "")).strip()
                # 规范化 raw_content（使 gold 值写死，防止细微不一致）
                raw_content = json.dumps(
                    {"classification": gold_score, "reason": gold_reason, "analysis": analysis},
                    ensure_ascii=False,
                )
            except Exception:
                analysis = ""
                raw_content = visible_text.strip()
                n_analysis_empty += 1

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
            sample_count += 1   

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
        help="推理内容的最小 token 数，低于此的轨迹丢弃",
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
        limit=args.limit,
    )
