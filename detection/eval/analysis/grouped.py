from __future__ import annotations

import json
from collections import defaultdict

import numpy as np
from sklearn.metrics import mean_absolute_error

from .io import get_task_from_path, round_score


def analyze_by_dissatisfaction_reason(results: list[dict]) -> dict:
    """仅对 label 不满意的样本：各 reason 的预测准确率和预测分布。"""
    dissatisfied = [r for r in results if float(r["label_score"]) <= 3]
    by_reason = defaultdict(list)
    for r in dissatisfied:
        by_reason[r["label_reason"]].append(r)

    out = {}
    all_reasons = set()
    for r in results:
        all_reasons.add(r["label_reason"])
        all_reasons.add(r["pred_reason"])
    all_reasons = sorted(all_reasons)

    for label_reason in sorted(by_reason.keys()):
        items = by_reason[label_reason]
        pred_reasons = [r["pred_reason"] for r in items]
        reason_correct = [1 if r["label_reason"] == r["pred_reason"] else 0 for r in items]
        pred_scores = [r["pred_score"] for r in items]
        label_scores = [float(r["label_score"]) for r in items]

        # 预测分布：各 pred_reason 的占比
        pred_dist = defaultdict(int)
        for pr in pred_reasons:
            pred_dist[pr] += 1
        pred_dist = {k: v / len(items) for k, v in pred_dist.items()}

        out[label_reason] = {
            "count": len(items),
            "reason_accuracy": np.mean(reason_correct),
            "mae": mean_absolute_error(label_scores, pred_scores),
            "prediction_distribution": pred_dist,
        }
    return out


def analyze_by_task(results: list[dict]) -> dict:
    """按任务类型（从 file_path 解析）的总体表现。"""
    by_task = defaultdict(list)
    for r in results:
        task = get_task_from_path(r["file_path"])
        by_task[task].append(r)

    out = {}
    for task in sorted(by_task.keys()):
        items = by_task[task]
        label_scores = [float(r["label_score"]) for r in items]
        pred_scores = [r["pred_score"] for r in items]
        reason_correct = [1 if r["label_reason"] == r["pred_reason"] else 0 for r in items]
        score_round = [round_score(r["pred_score"]) for r in items]
        score_correct = [1 if score_round[i] == int(label_scores[i]) else 0 for i in range(len(items))]

        out[task] = {
            "count": len(items),
            "mae": mean_absolute_error(label_scores, pred_scores),
            "score_accuracy": np.mean(score_correct),
            "reason_accuracy": np.mean(reason_correct),
        }
    return out


def analyze_by_turn(results: list[dict]) -> dict:
    """按对话轮次的预测表现（首轮 vs 多轮）。"""
    by_turn = defaultdict(list)
    for r in results:
        by_turn[r["turn"]].append(r)

    out = {}
    for turn in sorted(by_turn.keys()):
        items = by_turn[turn]
        label_scores = [float(r["label_score"]) for r in items]
        pred_scores = [r["pred_score"] for r in items]
        reason_correct = [1 if r["label_reason"] == r["pred_reason"] else 0 for r in items]
        out[turn] = {
            "count": len(items),
            "mae": mean_absolute_error(label_scores, pred_scores),
            "reason_accuracy": np.mean(reason_correct),
        }
    return out


def _get_chat_model_by_file_path(file_path: str, cache: dict[str, str]) -> str:
    """从 file_path 对应 JSON 中读取 chat_model，使用 cache 避免重复读文件。"""
    if file_path in cache:
        return cache[file_path]
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        cache[file_path] = data.get("chat_model", "unknown")
    except (OSError, json.JSONDecodeError):
        cache[file_path] = "unknown"
    return cache[file_path]


def analyze_by_chat_model(results: list[dict]) -> dict:
    """
    按对话使用的 LLM（来自 file_path 对应 JSON 的 chat_model）分组：
    平均 label 分数、平均预测分数、label 分布（1-5）、预测分布（四舍五入到 1-5）。
    """
    cache: dict[str, str] = {}
    by_model = defaultdict(list)
    for r in results:
        model = _get_chat_model_by_file_path(r["file_path"], cache)
        by_model[model].append(r)

    out = {}
    for model in sorted(by_model.keys()):
        items = by_model[model]
        label_scores = [float(r["label_score"]) for r in items]
        pred_scores = [float(r["pred_score"]) for r in items]
        n = len(items)

        label_dist = defaultdict(int)
        for s in label_scores:
            label_dist[int(s)] += 1
        label_dist = dict(sorted(label_dist.items(), key=lambda x: x[0]))

        pred_rounded = [round_score(p) for p in pred_scores]
        pred_dist = defaultdict(int)
        for s in pred_rounded:
            pred_dist[s] += 1
        pred_dist = dict(sorted(pred_dist.items(), key=lambda x: x[0]))

        out[model] = {
            "count": n,
            "mean_label_score": float(np.mean(label_scores)),
            "mean_pred_score": float(np.mean(pred_scores)),
            "label_distribution": label_dist,
            "pred_distribution": pred_dist,
        }
    return out


