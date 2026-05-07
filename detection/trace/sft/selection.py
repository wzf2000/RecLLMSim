from __future__ import annotations

import json
from typing import Any


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


def _row_first_attempt_wrong(row: dict[str, Any]) -> bool:
    pred_score = row.get("prediction")
    gold_score = row.get("gold_score")
    pred_reason = row.get("reason_prediction")
    gold_reason = row.get("gold_reason")
    return not (pred_score == gold_score and pred_reason == gold_reason)


def filter_reflected_wrong_records(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    与 collect_api_model_traces.generate_reflections_from_file 输出对齐：
    首轮预测错误且已写入 reflection（含 revised_reasoning）的样本。
    """
    selected: list[dict[str, Any]] = []
    for row in rows:
        if not _row_first_attempt_wrong(row):
            continue
        ref = row.get("reflection")
        if not isinstance(ref, dict):
            continue
        if not str(ref.get("revised_reasoning") or "").strip():
            continue
        selected.append(row)
    return selected


def select_records_for_sft(
    rows: list[dict[str, Any]],
    trace_source: str,
) -> list[dict[str, Any]]:
    if trace_source == "correct_only":
        return filter_correct_records(rows)
    if trace_source == "correct_plus_reflected_wrong":
        correct = filter_correct_records(rows)
        reflected = filter_reflected_wrong_records(rows)
        return correct + reflected
    raise ValueError(f"Unknown trace_source: {trace_source}")
