from __future__ import annotations

import json
from collections import defaultdict

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

from .io import round_score


def analyze_by_label_score(results: list[dict]) -> dict:
    """不同 label 分数（1-5）对应的预测准确程度。"""
    by_score = defaultdict(list)
    for r in results:
        label = int(r["label_score"])
        pred = r["pred_score"]
        pred_reason = r["pred_reason"]
        label_reason = r["label_reason"]
        by_score[label].append({
            "pred_score": pred,
            "pred_round": round_score(pred),
            "label_reason": label_reason,
            "pred_reason": pred_reason,
        })

    out = {}
    for score in sorted(by_score.keys()):
        items = by_score[score]
        pred_scores = [x["pred_score"] for x in items]
        label_scores = [score] * len(items)
        # pred_round = [x["pred_round"] for x in items]
        reason_correct = [1 if x["label_reason"] == x["pred_reason"] else 0 for x in items]
        score_correct = [1 if round_score(p) == score else 0 for p in pred_scores]

        out[score] = {
            "count": len(items),
            "mae": mean_absolute_error(label_scores, pred_scores),
            "rmse": root_mean_squared_error(label_scores, pred_scores),
            "score_accuracy": np.mean(score_correct),
            "reason_accuracy": np.mean(reason_correct),
        }
    return out


def analyze_after_dissatisfied(results: list[dict]) -> dict:
    """前一轮 label<=3（不满意）时，当前轮的预测准确度。"""
    subset = []
    cnt = 0
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
            if r["label_score"] > 3:
                cnt += 1
            subset.append(r)
    print(f"不满意后变为满意的样本数: {cnt}")
    if not subset:
        return {"count": 0, "message": "无「前一轮不满意」的样本"}

    label_scores = [float(r["label_score"]) for r in subset]
    pred_scores = [r["pred_score"] for r in subset]
    reason_correct = [1 if r["label_reason"] == r["pred_reason"] else 0 for r in subset]
    score_round = [round_score(r["pred_score"]) for r in subset]
    score_correct = [1 if score_round[i] == int(label_scores[i]) else 0 for i in range(len(subset))]

    return {
        "count": len(subset),
        "mae": mean_absolute_error(label_scores, pred_scores),
        "rmse": root_mean_squared_error(label_scores, pred_scores),
        "pearson": pearsonr(label_scores, pred_scores)[0],
        "spearman": spearmanr(label_scores, pred_scores)[0],
        "score_accuracy": np.mean(score_correct),
        "reason_accuracy": np.mean(reason_correct),
    }


def analyze_label_calibration(results: list[dict]) -> dict:
    """
    对每个 label 分数：
    - 预测分数均值
    - 偏高/偏低/相等比例（基于 pred_score 与 label_score 的连续比较）
    """
    by_score = defaultdict(list)
    for r in results:
        by_score[int(r["label_score"])].append(float(r["pred_score"]))

    out = {}
    for score in sorted(by_score.keys()):
        preds = np.array(by_score[score], dtype=float)
        label = float(score)
        out[score] = {
            "count": int(preds.size),
            "pred_mean": float(preds.mean()) if preds.size else None,
            "pred_std": float(preds.std(ddof=0)) if preds.size else None,
            "higher_ratio": float(np.mean(preds > label)) if preds.size else None,
            "lower_ratio": float(np.mean(preds < label)) if preds.size else None,
            "equal_ratio": float(np.mean(preds == label)) if preds.size else None,
        }
    return out


