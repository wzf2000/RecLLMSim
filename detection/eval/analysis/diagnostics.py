from __future__ import annotations

from collections import defaultdict

import numpy as np


def analyze_large_error_cases(results: list[dict], abs_diff_threshold: float = 2.0) -> dict:
    """
    预测与 GT 差距过大（|pred-label|>=threshold）的样本：
    - 样本量与占比
    - label 分数分布
    - （附加）按方向：预测偏高/偏低的比例
    """
    diffs = np.array([float(r["pred_score"]) - float(r["label_score"]) for r in results], dtype=float)
    mask = np.abs(diffs) >= abs_diff_threshold
    idx = np.where(mask)[0].tolist()
    if len(results) == 0:
        return {"count": 0, "ratio": 0.0, "threshold": abs_diff_threshold}

    label_dist = defaultdict(int)
    for i in idx:
        label_dist[int(float(results[i]["label_score"]))] += 1

    return {
        "threshold": float(abs_diff_threshold),
        "count": int(len(idx)),
        "ratio": float(len(idx) / len(results)),
        "label_distribution": dict(sorted(label_dist.items(), key=lambda x: x[0])),
        "higher_ratio": float(np.mean(diffs[mask] > 0)) if len(idx) else 0.0,
        "lower_ratio": float(np.mean(diffs[mask] < 0)) if len(idx) else 0.0,
    }


def analyze_pred_reason_score_alignment(results: list[dict], threshold: float = 3.5, satisfied_reason: str = "满意") -> dict:
    """
    reason 分类 与 满意度回归 的对齐程度（基于预测）：
    - pred_score>=threshold 时，pred_reason 是否为 satisfied_reason
    - pred_reason==satisfied_reason 时，pred_score 是否 >=threshold
    """
    if not results:
        return {"count": 0, "threshold": float(threshold), "satisfied_reason": satisfied_reason}

    pred_satisfied = np.array([float(r["pred_score"]) >= threshold for r in results], dtype=bool)
    pred_reason_satisfied = np.array([r["pred_reason"] == satisfied_reason for r in results], dtype=bool)

    n = len(results)
    n_pred_satisfied = int(pred_satisfied.sum())
    n_reason_satisfied = int(pred_reason_satisfied.sum())
    n_both = int(np.logical_and(pred_satisfied, pred_reason_satisfied).sum())

    # P(pred_reason==满意 | pred_score>=threshold)
    p_reason_given_score = float(n_both / n_pred_satisfied) if n_pred_satisfied else None
    # P(pred_score>=threshold | pred_reason==满意)
    p_score_given_reason = float(n_both / n_reason_satisfied) if n_reason_satisfied else None

    mismatch = np.logical_xor(pred_satisfied, pred_reason_satisfied)
    mismatch_rate = float(mismatch.mean())
    mismatch_breakdown = {
        "score_satisfied_but_reason_not": int(np.logical_and(pred_satisfied, ~pred_reason_satisfied).sum()),
        "reason_satisfied_but_score_not": int(np.logical_and(~pred_satisfied, pred_reason_satisfied).sum()),
    }

    return {
        "count": n,
        "threshold": float(threshold),
        "satisfied_reason": satisfied_reason,
        "n_pred_satisfied": n_pred_satisfied,
        "n_pred_reason_satisfied": n_reason_satisfied,
        "n_both_satisfied": n_both,
        "p_reason_satisfied_given_score_satisfied": p_reason_given_score,
        "p_score_satisfied_given_reason_satisfied": p_score_given_reason,
        "mismatch_rate": mismatch_rate,
        "mismatch_breakdown": mismatch_breakdown,
    }


