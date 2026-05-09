from __future__ import annotations

from loguru import logger

from lib.user_aware_metrics import (
    compute_user_aware_binary_metrics,
    compute_user_aware_metrics,
    print_user_aware_binary_metrics,
    print_user_aware_metrics,
)
from .boundary_metrics import compute_boundary_metrics, get_sat_confidence, print_boundary_metrics, to_binary_sat
from .global_metrics import compute_global_metrics, print_global_metrics
from .io import load_records
from .stratified import (
    boundary_stratified_analysis,
    print_boundary_stratified_analysis,
    print_stratified_analysis,
    stratified_analysis,
)


def evaluate_single(
    path: str,
    name: str,
    min_samples: int = 3,
) -> dict:
    """评估单个结果文件，返回完整指标字典。"""
    records = load_records(path)
    if not records:
        logger.warning(f"无有效记录: {path}")
        return {}

    gold = [float(r["gold_score"]) for r in records]
    pred = [float(r["pred_score"]) for r in records]
    users = [str(r.get("user", "unknown")) for r in records]

    logger.info(f"\n{'='*60}")
    logger.info(f"  文件: {path}  ({name})")
    logger.info(f"  总样本: {len(gold)},  用户数: {len(set(users))}")

    gm = compute_global_metrics(gold, pred)
    print_global_metrics(gm, label="全局指标")

    bm = compute_boundary_metrics(records)
    print_boundary_metrics(bm, label="3/4 边界（二分类：SAT/DSAT）")

    ua = compute_user_aware_metrics(gold, pred, users, min_samples=min_samples)
    print_user_aware_metrics(ua, header=f"用户感知指标 — {name}")

    gold_bin = [to_binary_sat(r["gold_score"]) for r in records]
    pred_bin = [to_binary_sat(r["pred_score"]) for r in records]
    conf = [get_sat_confidence(r) for r in records]
    ua_bin = compute_user_aware_binary_metrics(
        gold_bin, pred_bin, users, conf, min_samples=min_samples,
    )
    print_user_aware_binary_metrics(ua_bin, header=f"用户感知（二分类）— {name}")

    strat = stratified_analysis(records)
    print_stratified_analysis(strat)

    bstrat = boundary_stratified_analysis(records)
    print_boundary_stratified_analysis(bstrat)

    return {
        "global": gm,
        "boundary": bm,
        "user_aware": ua,
        "user_aware_boundary": ua_bin,
        "stratified": strat,
        "boundary_stratified": bstrat,
        "n": len(gold),
    }


