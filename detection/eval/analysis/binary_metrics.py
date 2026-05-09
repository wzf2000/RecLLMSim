from __future__ import annotations

import json

import numpy as np


def _binary_satisfaction_metrics(results: list[dict], threshold: float = 3.5) -> dict:
    """
    将 label 按 <=3 / >3 二分类（不满意/满意）。
    将 pred_score 按 threshold 二分类（pred>=threshold 认为满意）。
    """
    y_true = np.array([1 if float(r["label_score"]) > 3 else 0 for r in results], dtype=int)
    y_pred = np.array([1 if float(r["pred_score"]) >= threshold else 0 for r in results], dtype=int)
    if y_true.size == 0:
        return {"count": 0}
    return {
        "count": int(y_true.size),
        "threshold": float(threshold),
        "accuracy": float(np.mean(y_true == y_pred)),
        "pos_rate_label": float(y_true.mean()),
        "pos_rate_pred": float(y_pred.mean()),
        "pos_label_accuracy": float(np.mean(y_true[y_pred == 1] == 1)),
        "neg_label_accuracy": float(np.mean(y_true[y_pred == 0] == 0)),
    }


def analyze_binary_satisfaction(results: list[dict], threshold: float = 3.5) -> dict:
    """全局 + 前一轮不满意子集的二分类准确率（label: <=3/>3，pred: >=threshold）。"""
    overall = _binary_satisfaction_metrics(results, threshold=threshold)

    subset = []
    for r in results:
        turn = r["turn"]
        if turn <= 1:
            continue
        file_path = r["file_path"]
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        turn_cnt = 0
        prev_score = None
        for utt in data["history"]:
            if utt["role"] == "assistant":
                turn_cnt += 1
                if turn_cnt == turn - 1:
                    prev_score = int(utt["satisfaction"])
                    break
        assert prev_score is not None
        if prev_score <= 3:
            subset.append(r)
    after_prev_dissatisfied = _binary_satisfaction_metrics(subset, threshold=threshold)
    return {"overall": overall, "after_prev_dissatisfied": after_prev_dissatisfied}


