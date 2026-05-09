from __future__ import annotations

from loguru import logger

from .global_metrics import _fmt


def print_comparison_table(all_results: dict[str, dict]) -> None:
    sep = "=" * 80
    metrics = [
        ("MAE",      "mae"),
        ("RMSE",     "rmse"),
        ("Pearson",  "pearson"),
        ("Spearman", "spearman"),
        ("Kappa",    "kappa"),
    ]
    col_w = 40

    logger.info(sep)
    logger.info("  指标对比摘要（全局）")
    logger.info(sep)

    header = f"  {'模型/配置':<{col_w}}" + "".join(f"  {m[0]:>9}" for m in metrics)
    logger.info(header)
    logger.info("  " + "-" * 76)

    for name, r in all_results.items():
        gm = r.get("global", {})
        row = f"  {name:<{col_w}}" + "".join(
            f"  {_fmt(gm.get(mk, float('nan'))):>9}" for _, mk in metrics
        )
        logger.info(row)
    logger.info(sep)


def print_boundary_comparison_table(all_results: dict[str, dict]) -> None:
    sep = "=" * 98
    metrics = [
        ("Acc", "accuracy"),
        ("F1-mac", "f1_macro"),
        ("F1-SAT", "f1_sat"),
        ("F1-DSAT", "f1_dsat"),
        ("Kappa", "kappa"),
        ("AUC", "auc"),
        ("FalseSAT", "false_sat_rate"),
        ("FalseDSAT", "false_dsat_rate"),
    ]
    col_w = 28

    logger.info(sep)
    logger.info("  指标对比摘要（3/4 边界：SAT/DSAT）")
    logger.info(sep)
    header = f"  {'模型/配置':<{col_w}}" + "".join(f"  {m[0]:>8}" for m in metrics)
    logger.info(header)
    logger.info("  " + "-" * 94)

    for name, r in all_results.items():
        bm = r.get("boundary", {})
        row = f"  {name:<{col_w}}" + "".join(
            f"  {_fmt(bm.get(mk, float('nan'))):>8}" for _, mk in metrics
        )
        logger.info(row)
    logger.info(sep)


# ──────────────────────────────────────────────────────────────────────────────
# 主流程
# ──────────────────────────────────────────────────────────────────────────────

