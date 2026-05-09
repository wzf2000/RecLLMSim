from __future__ import annotations

from collections import defaultdict

from loguru import logger

from .boundary_metrics import compute_boundary_metrics
from .global_metrics import _fmt, compute_global_metrics


def stratified_analysis(records: list[dict]) -> dict[str, dict]:
    """
    分层计算 MAE，维度包括：
      - 目标任务类型（target_task）
      - 记忆更新模式（memory_update_mode）
      - 是否使用记忆（with_memory）
    """
    results: dict[str, dict] = {}

    def _group_and_compute(key_fn, label_prefix: str) -> None:
        groups: dict[str, tuple[list, list]] = defaultdict(lambda: ([], []))
        for r in records:
            k = key_fn(r)
            groups[k][0].append(float(r["gold_score"]))
            groups[k][1].append(float(r["pred_score"]))
        for group_key, (g, p) in sorted(groups.items()):
            full_key = f"{label_prefix}/{group_key}"
            results[full_key] = compute_global_metrics(g, p)

    _group_and_compute(lambda r: r.get("target_task", "unknown"), "by_task")
    _group_and_compute(
        lambda r: str(r.get("memory_update_mode", "unknown")), "by_update_mode"
    )
    _group_and_compute(
        lambda r: "with_memory" if r.get("with_memory") else "no_memory", "by_memory"
    )

    return results


def print_stratified_analysis(results: dict[str, dict]) -> None:
    logger.info("=" * 60)
    logger.info("  分层分析（MAE）")
    logger.info("=" * 60)
    for key, m in sorted(results.items()):
        logger.info(f"  {key:<42}  MAE={_fmt(m['mae'])}  n={m['n_samples']}")
    logger.info("=" * 60)


# ──────────────────────────────────────────────────────────────────────────────
# 分层边界分析
# ──────────────────────────────────────────────────────────────────────────────

def boundary_stratified_analysis(records: list[dict]) -> dict[str, dict]:
    results: dict[str, dict] = {}

    def _group_and_compute(key_fn, label_prefix: str) -> None:
        groups: dict[str, list[dict]] = defaultdict(list)
        for r in records:
            groups[key_fn(r)].append(r)
        for group_key, group_records in sorted(groups.items()):
            full_key = f"{label_prefix}/{group_key}"
            results[full_key] = compute_boundary_metrics(group_records)

    _group_and_compute(lambda r: r.get("target_task", "unknown"), "by_task")
    _group_and_compute(
        lambda r: str(r.get("memory_update_mode", "unknown")), "by_update_mode"
    )
    _group_and_compute(
        lambda r: "with_memory" if r.get("with_memory") else "no_memory", "by_memory"
    )

    return results


def print_boundary_stratified_analysis(results: dict[str, dict]) -> None:
    logger.info("=" * 72)
    logger.info("  分层分析（3/4 边界 Accuracy / F1-DSAT）")
    logger.info("=" * 72)
    for key, m in sorted(results.items()):
        logger.info(
            f"  {key:<42}  Acc={_fmt(m['accuracy'])}  "
            f"F1-DSAT={_fmt(m['f1_dsat'])}  n={m['n_samples']}"
        )
    logger.info("=" * 72)


# ──────────────────────────────────────────────────────────────────────────────
# 个性化增益（Personalization Gain）
# ──────────────────────────────────────────────────────────────────────────────

