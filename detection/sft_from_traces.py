import json
import os
import re
import wandb
from argparse import ArgumentParser
from typing import Any

import torch
from datasets import Dataset
from loguru import logger
from peft import LoraConfig, get_peft_model
from sklearn.model_selection import GroupShuffleSplit
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizer,
    Trainer,
    TrainingArguments,
    default_data_collator,
)

from satisfaction_constants import get_reason_to_id


def split_history_turns(history: str) -> list[str]:
    """
    按采集端格式划分轮次：每轮以行首的 `user：` / `assistant：` 开头（全角冒号 `：`），
    与 `f'{utt["role"]}：{utt["content"]}\\n'` 一致；单轮 content 内可含任意换行。
    """
    if not (history or "").strip():
        return []
    text = history.strip()
    # 仅在新行行首匹配角色前缀，避免误切单轮内的换行
    pattern = re.compile(r"(?im)^(?P<role>user|assistant)：")
    matches = list(pattern.finditer(text))
    if not matches:
        return [text]
    turns: list[str] = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        chunk = text[start:end].rstrip()
        if chunk:
            turns.append(chunk)
    return turns


def build_prompt_like_collect(
    persona: str,
    task_context: str,
    history_text: str,
    assistant_reply: str,
) -> str:
    """与 detection/collect_api_model_traces.py 中 build_prompt 对齐。"""
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
        "请严格输出一个 JSON 对象，不要输出其他内容：\n"
        "{\n"
        '  "classification": 1-5中的整数,\n'
        '  "reason": "从可选原因标签中选择一个",\n'
        '  "analysis": "你的详细推理过程"\n'
        "}\n"
    )


def resolve_prompt_for_row(
    row: dict[str, Any],
    tokenizer: PreTrainedTokenizer,
    max_length: int,
    include_reasoning_content: bool,
    max_history_turns_cap: int = 5,
    source_budget_override: int | None = None,
) -> str:
    """
    优先用 meta 按采集脚本格式重建 prompt；若 source 过长，则减少 history 轮次（保留最近若干轮），
    与采集端「滑动窗口」一致，最多保留 max_history_turns_cap 轮。

    source_budget_override: 推理评测时可传入「prompt 允许的最大 token 数」（例如 max_length - max_new_tokens），
    若设置则不再用教师 target 长度来估算 budget。
    """
    meta = row.get("meta")
    if not isinstance(meta, dict):
        return str(row.get("prompt", ""))

    persona = str(meta.get("persona", "") or "")
    task_context = str(meta.get("task_context", "") or "")
    assistant_reply = str(meta.get("assistant_reply", "") or "")
    turns = split_history_turns(str(meta.get("history", "") or ""))

    if source_budget_override is not None:
        source_budget = max(1, int(source_budget_override))
    else:
        target = build_assistant_target(row, include_reasoning_content=include_reasoning_content)
        target_ids = tokenizer(target, add_special_tokens=False)["input_ids"]
        target_len = len(target_ids)
        if target_len >= max_length:
            # tokenize_example 会只保留 target；prompt 仍给最短合理形式
            k = 0
            history_text = ""
            prompt = build_prompt_like_collect(persona, task_context, history_text, assistant_reply)
            return prompt

        source_budget = max_length - target_len
    # 与采集一致：最多考虑最近 max_history_turns_cap 轮
    max_k = min(max_history_turns_cap, len(turns))
    for k in range(max_k, -1, -1):
        kept_turns = turns[-k:] if k > 0 else []
        history_text = "\n".join(kept_turns)
        prompt = build_prompt_like_collect(persona, task_context, history_text, assistant_reply)
        source_text = build_source_text(tokenizer, prompt)
        src_len = len(tokenizer(source_text, add_special_tokens=False)["input_ids"])
        if src_len <= source_budget:
            return prompt

    # 理论上 k=0 应满足；若仍超出（极长 persona 等），返回最短 prompt，后续仍可在 tokenize 里截断 source
    return build_prompt_like_collect(persona, task_context, "", assistant_reply)


def load_jsonl(path: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def filter_correct_records(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for row in rows:
        pred_score = row.get("prediction")
        gold_score = row.get("gold_score")
        pred_reason = row.get("reason_prediction")
        gold_reason = row.get("gold_reason")
        if pred_score == gold_score and pred_reason == gold_reason:
            selected.append(row)
    return selected


def build_assistant_target(row: dict[str, Any], include_reasoning_content: bool = False) -> str:
    reasoning_content = row.get("reasoning_content")
    raw_content = row.get("raw_content")
    raw_text = raw_content.strip() if isinstance(raw_content, str) and raw_content.strip() else ""
    reasoning_text = reasoning_content.strip() if isinstance(reasoning_content, str) and reasoning_content.strip() else ""
    if raw_text:
        if include_reasoning_content and reasoning_text:
            return f"{reasoning_text}\n\n{raw_text}"
        return raw_text

    fallback = {
        "classification": int(row["prediction"]),
        "reason": row["reason_prediction"],
        "analysis": row.get("analysis", ""),
    }
    return json.dumps(fallback, ensure_ascii=False)


def build_source_text(tokenizer: PreTrainedTokenizer, prompt: str) -> str:
    messages = [
        {"role": "system", "content": "You are a skilled conversational analyst."},
        {"role": "user", "content": prompt},
    ]
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(  # type: ignore
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            pass
    return f"System: You are a skilled conversational analyst.\nUser: {prompt}\nAssistant:"


def tokenize_example(
    row: dict[str, Any],
    tokenizer: PreTrainedTokenizer,
    max_length: int,
    include_reasoning_content: bool = False,
    max_history_turns: int = 5,
) -> dict[str, list[int]]:
    prompt = resolve_prompt_for_row(
        row,
        tokenizer,
        max_length=max_length,
        include_reasoning_content=include_reasoning_content,
        max_history_turns_cap=max_history_turns,
    )
    target = build_assistant_target(row, include_reasoning_content=include_reasoning_content)

    source_text = build_source_text(tokenizer, prompt)
    # 分别对 source / target 分词，再拼接，避免「整段从右截断」时把 assistant 目标全部截掉，
    # 导致 labels 全为 -100、eval_loss 出现 NaN。
    source_ids = tokenizer(source_text, add_special_tokens=False)["input_ids"]
    target_ids = tokenizer(target, add_special_tokens=False)["input_ids"]

    # 优先保留 target；source 过长时从左侧截断（保留末尾上下文，更接近「最近对话」）
    if len(target_ids) >= max_length:
        target_ids = target_ids[-max_length:]
        source_ids = []
    else:
        budget = max_length - len(target_ids)
        if len(source_ids) > budget:
            source_ids = source_ids[-budget:]

    input_ids = source_ids + target_ids
    attention_mask = [1] * len(input_ids)
    labels = [-100] * len(source_ids) + target_ids

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def get_train_valid_dataset(
    records: list[dict[str, Any]],
    tokenizer: PreTrainedTokenizer,
    max_length: int,
    split_seed: int = 42,
    include_reasoning_content: bool = False,
    max_history_turns: int = 5,
) -> tuple[Dataset, Dataset]:
    users = [str(r.get("meta", {}).get("user", "unknown")) for r in records]
    n = len(records)
    if n < 2:
        raise ValueError("Need at least 2 records to build train/valid dataset.")

    # 这里仅需要 train/valid 两路切分，避免 test_ratio=0 触发 train_size=1.0 的边界错误
    gss = GroupShuffleSplit(n_splits=1, train_size=0.9, test_size=0.1, random_state=split_seed)
    all_idx = list(range(n))
    train_rel, valid_rel = next(gss.split(all_idx, groups=users))
    train_idx = [all_idx[i] for i in train_rel]
    valid_idx = [all_idx[i] for i in valid_rel]

    # 极小数据兜底：保证 valid 至少 1 条、train 至少 1 条
    if not valid_idx:
        valid_idx = [train_idx[-1]]
        train_idx = train_idx[:-1]
    if not train_idx:
        train_idx = [valid_idx[0]]
        valid_idx = valid_idx[1:] if len(valid_idx) > 1 else [valid_idx[0]]

    train_rows = [records[i] for i in train_idx]
    valid_rows = [records[i] for i in valid_idx]

    train_ds = Dataset.from_list(train_rows)
    valid_ds = Dataset.from_list(valid_rows if valid_rows else train_rows[: max(1, min(64, len(train_rows)))])

    train_ds = train_ds.map(
        lambda x: tokenize_example(
            x,
            tokenizer,
            max_length,
            include_reasoning_content=include_reasoning_content,
            max_history_turns=max_history_turns,
        ),
        remove_columns=train_ds.column_names,
    )
    valid_ds = valid_ds.map(
        lambda x: tokenize_example(
            x,
            tokenizer,
            max_length,
            include_reasoning_content=include_reasoning_content,
            max_history_turns=max_history_turns,
        ),
        remove_columns=valid_ds.column_names,
    )
    return train_ds, valid_ds


def train_sft(
    input_jsonl: str,
    model_name: str,
    output_dir: str,
    max_length: int = 2048,
    batch_size: int = 2,
    grad_accum: int = 8,
    num_epochs: int = 3,
    learning_rate: float = 2e-4,
    split_seed: int = 42,
    include_reasoning_content: bool = False,
    max_history_turns: int = 5,
) -> None:
    wandb.init(project="satisfaction_prediction_sft", name=f"sft_from_traces_{'with_reasoning' if include_reasoning_content else 'without_reasoning'}")
    rows = load_jsonl(input_jsonl)
    rows = filter_correct_records(rows)
    if not rows:
        raise ValueError("No correct records found in input traces.")
    logger.info(f"Correct records for SFT: {len(rows)}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_ds, valid_ds = get_train_valid_dataset(
        rows,
        tokenizer,
        max_length=max_length,
        split_seed=split_seed,
        include_reasoning_content=include_reasoning_content,
        max_history_turns=max_history_turns,
    )
    logger.info(f"Train size: {len(train_ds)}, Valid size: {len(valid_ds)}")
    logger.info(f"Include reasoning_content in target: {include_reasoning_content}")
    logger.info(f"max_history_turns (cap, align with collect): {max_history_turns}")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        dtype=torch.bfloat16,
    )
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        num_train_epochs=num_epochs,
        learning_rate=learning_rate,
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch",
        bf16=torch.cuda.is_available(),
        fp16=False,
        report_to="wandb",
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        processing_class=tokenizer,
        data_collator=default_data_collator,
    )
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info(f"SFT finished. Model saved to: {output_dir}")


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--input_jsonl", type=str, required=True, help="轨迹 jsonl 路径")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-8B")
    parser.add_argument("--output_dir", type=str, default="./ckpts/sft_from_api_traces")
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument(
        "--include_reasoning_content",
        action="store_true",
        help="是否将 reasoning_content 与 raw_content 拼接后作为监督目标",
    )
    parser.add_argument(
        "--max_history_turns",
        type=int,
        default=5,
        help="history 最多保留的轮次数（与采集端滑动窗口一致）；过长时会从更少轮次尝试以适配 max_length",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_sft(**vars(args))
