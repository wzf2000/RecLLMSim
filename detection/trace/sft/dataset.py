from __future__ import annotations

from typing import Any

from datasets import Dataset
from sklearn.model_selection import GroupShuffleSplit
from transformers import PreTrainedTokenizer

from .prompts import build_source_text, resolve_prompt_for_row
from .targets import build_assistant_target


def tokenize_example(
    row: dict[str, Any],
    tokenizer: PreTrainedTokenizer,
    max_length: int,
    include_reasoning_content: bool = False,
    max_history_turns: int = 5,
    think_wrap: str = "qwen3",
) -> dict[str, list[int]]:
    prompt = resolve_prompt_for_row(
        row,
        tokenizer,
        max_length=max_length,
        include_reasoning_content=include_reasoning_content,
        max_history_turns_cap=max_history_turns,
        think_wrap=think_wrap,
    )
    target = build_assistant_target(
        row,
        include_reasoning_content=include_reasoning_content,
        think_wrap=think_wrap,
    )

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

    # Pad to max_length so default_data_collator can stack tensors
    pad_len = max_length - len(input_ids)
    if pad_len > 0:
        pad_id = tokenizer.pad_token_id or 0
        input_ids = input_ids + [pad_id] * pad_len
        attention_mask = attention_mask + [0] * pad_len
        labels = labels + [-100] * pad_len

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
    think_wrap: str = "qwen3",
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
            think_wrap=think_wrap,
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
            think_wrap=think_wrap,
        ),
        remove_columns=valid_ds.column_names,
    )
    return train_ds, valid_ds
