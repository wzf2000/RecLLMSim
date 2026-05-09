from __future__ import annotations

from loguru import logger

from lib.data_split import split_by_user_group_shuffle_split
from lib.metric_statistics import get_satisfaction_data
from predictor.bert import format_profile


def preprocess_to_rows(data_list: list[dict]) -> list[dict]:
    rows: list[dict] = []
    for sample in data_list:
        persona = format_profile(sample["profile"])
        task_context = sample["task_context"]
        history_window: list[str] = []
        assistant_turn_idx = 0

        for utt in sample["history"]:
            if utt["role"] == "assistant":
                rows.append(
                    {
                        "persona": persona,
                        "task_context": task_context,
                        "history": "\n".join(history_window),
                        "assistant_reply": utt["content"],
                        "gold_score": int(sample["satisfaction_scores"][assistant_turn_idx]),
                        "gold_reason": sample["dissatisfaction_reasons"][assistant_turn_idx],
                        "user": sample.get("user", "unknown"),
                    }
                )
                assistant_turn_idx += 1

            history_window.append(f'{utt["role"]}：{utt["content"]}\n')
            while len(history_window) > 5:
                history_window.pop(0)
    return rows




def get_rows_from_split(
    split: str = "train",
    split_seed: int = 42,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
) -> list[dict]:
    data_list = get_satisfaction_data()
    rows = preprocess_to_rows(data_list)
    users = [row["user"] for row in rows]
    train_idx, valid_idx, test_idx = split_by_user_group_shuffle_split(
        users,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=split_seed,
    )
    idx_map = {
        "train": train_idx,
        "valid": valid_idx,
        "val": valid_idx,
        "test": test_idx,
        "all": list(range(len(rows))),
    }
    if split not in idx_map:
        raise ValueError(f"Invalid split: {split}, expected one of train/valid/val/test/all")
    selected = [rows[i] for i in idx_map[split]]
    logger.info(
        f"Split sizes -> train: {len(train_idx)}, valid: {len(valid_idx)}, test: {len(test_idx)}, selected({split}): {len(selected)}"
    )
    return selected


