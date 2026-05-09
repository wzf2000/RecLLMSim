from __future__ import annotations

from collections import defaultdict

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, root_mean_squared_error

from .io import round_score


def reason_confusion_matrix(results: list[dict]) -> dict:
    """Reason 预测的混淆统计：label_reason -> 预测成各 reason 的数量。"""
    label_list = [r["label_reason"] for r in results]
    pred_list = [r["pred_reason"] for r in results]
    all_reasons = sorted(set(label_list) | set(pred_list))
    cm = defaultdict(lambda: defaultdict(int))
    for label, pred in zip(label_list, pred_list):
        cm[label][pred] += 1
    return {"labels": all_reasons, "matrix": {k: dict(v) for k, v in cm.items()}}


def overall_metrics(results: list[dict]) -> dict:
    """整体指标。"""
    label_scores = [float(r["label_score"]) for r in results]
    pred_scores = [r["pred_score"] for r in results]
    label_reasons = [r["label_reason"] for r in results]
    pred_reasons = [r["pred_reason"] for r in results]
    score_round = [round_score(p) for p in pred_scores]
    score_correct = [1 if score_round[i] == int(label_scores[i]) else 0 for i in range(len(results))]

    return {
        "n_samples": len(results),
        "mae": mean_absolute_error(label_scores, pred_scores),
        "rmse": root_mean_squared_error(label_scores, pred_scores),
        "pearson": pearsonr(label_scores, pred_scores)[0],
        "spearman": spearmanr(label_scores, pred_scores)[0],
        "score_accuracy": np.mean(score_correct),
        "reason_accuracy": accuracy_score(label_reasons, pred_reasons),
        "reason_f1_weighted": f1_score(label_reasons, pred_reasons, average="weighted", zero_division=0),
    }


