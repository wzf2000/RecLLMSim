from __future__ import annotations

from collections import defaultdict

import numpy as np
from loguru import logger
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

from .global_metrics import _fmt


def compute_personalization_gain(
    mem_records: list[dict],
    baseline_records: list[dict],
) -> dict[str, float]:
    """
    计算有记忆 vs. 无记忆 baseline 的个性化增益。

    要求两批记录的 sample_id 可对齐（按 sample_id 匹配）；
    若 sample_id 不存在则按顺序对齐（要求等长）。
    """
    # 尝试用 sample_id 对齐
    baseline_by_id = {r.get("sample_id", i): r for i, r in enumerate(baseline_records)}
    matched_mem, matched_base = [], []
    unmatched = 0
    for r in mem_records:
        sid = r.get("sample_id")
        if sid and sid in baseline_by_id:
            matched_mem.append(r)
            matched_base.append(baseline_by_id[sid])
        else:
            unmatched += 1

    if unmatched > 0:
        logger.warning(f"个性化增益计算：{unmatched} 条记录无法与 baseline 对齐，已跳过。")

    if not matched_mem:
        return {"pg_mae": float("nan"), "pg_rmse": float("nan"), "n_matched": 0}

    gold = [float(r["gold_score"]) for r in matched_mem]
    pred_mem = [float(r["pred_score"]) for r in matched_mem]
    pred_base = [float(r["pred_score"]) for r in matched_base]

    mae_mem = float(mean_absolute_error(gold, pred_mem))
    mae_base = float(mean_absolute_error(gold, pred_base))
    rmse_mem = float(root_mean_squared_error(gold, pred_mem))
    rmse_base = float(root_mean_squared_error(gold, pred_base))

    return {
        "pg_mae":  mae_base - mae_mem,       # > 0 = memory helps
        "pg_rmse": rmse_base - rmse_mem,
        "mae_with_memory": mae_mem,
        "mae_baseline":    mae_base,
        "rmse_with_memory": rmse_mem,
        "rmse_baseline":    rmse_base,
        "n_matched": len(matched_mem),
    }


def per_user_personalization_gain(
    mem_records: list[dict],
    baseline_records: list[dict],
) -> dict[str, float]:
    """
    按用户分别计算 PG，返回加权平均 PG。
    用于分析哪类用户从个性化中受益更多。
    """
    baseline_by_id = {r.get("sample_id", i): r for i, r in enumerate(baseline_records)}

    user_pairs: dict[str, tuple[list, list, list]] = defaultdict(lambda: ([], [], []))
    for r in mem_records:
        sid = r.get("sample_id")
        if sid and sid in baseline_by_id:
            base_r = baseline_by_id[sid]
            user = r.get("user", "unknown")
            user_pairs[user][0].append(float(r["gold_score"]))
            user_pairs[user][1].append(float(r["pred_score"]))
            user_pairs[user][2].append(float(base_r["pred_score"]))

    pg_per_user: list[float] = []
    ns: list[int] = []
    positive_users = 0
    for user, (gold, pred_m, pred_b) in user_pairs.items():
        pg = mean_absolute_error(gold, pred_b) - mean_absolute_error(gold, pred_m)
        pg_per_user.append(pg)
        ns.append(len(gold))
        if pg > 0:
            positive_users += 1

    if not pg_per_user:
        return {}

    return {
        "pu_pg_mae":          float(np.average(pg_per_user, weights=ns)),
        "pu_pg_mae_unweighted": float(np.mean(pg_per_user)),
        "n_users":            len(pg_per_user),
        "n_users_positive_pg": positive_users,
        "pct_users_positive_pg": positive_users / len(pg_per_user),
    }


def print_personalization_gain(pg: dict, pu_pg: dict | None = None) -> None:
    sep = "=" * 60
    logger.info(sep)
    logger.info("  个性化增益（Personalization Gain）")
    logger.info(sep)
    n = pg.get("n_matched", 0)
    logger.info(f"  对齐样本数: {n}")
    pg_mae = pg.get("pg_mae", float("nan"))
    pg_rmse = pg.get("pg_rmse", float("nan"))
    indicator = "↑ memory helps" if pg_mae > 0 else ("↓ memory hurts" if pg_mae < 0 else "= neutral")
    logger.info(
        f"  PG (MAE):  {_fmt(pg_mae)}  {indicator}"
        f"    PG (RMSE): {_fmt(pg_rmse)}"
    )
    logger.info(
        f"  MAE  with_memory={_fmt(pg.get('mae_with_memory', float('nan')))}"
        f"  baseline={_fmt(pg.get('mae_baseline', float('nan')))}"
    )
    logger.info(
        f"  RMSE with_memory={_fmt(pg.get('rmse_with_memory', float('nan')))}"
        f"  baseline={_fmt(pg.get('rmse_baseline', float('nan')))}"
    )
    if pu_pg:
        logger.info("  ── Per-user PG ─────────────────────────────────")
        logger.info(
            f"  加权平均 PG (MAE): {_fmt(pu_pg.get('pu_pg_mae', float('nan')))}"
            f"    未加权: {_fmt(pu_pg.get('pu_pg_mae_unweighted', float('nan')))}"
        )
        logger.info(
            f"  用户总数: {pu_pg.get('n_users', 0)}"
            f"    PG>0 用户数: {pu_pg.get('n_users_positive_pg', 0)}"
            f"  ({pu_pg.get('pct_users_positive_pg', 0):.1%})"
        )
    logger.info(sep)


# ──────────────────────────────────────────────────────────────────────────────
# 多文件对比表
# ──────────────────────────────────────────────────────────────────────────────

