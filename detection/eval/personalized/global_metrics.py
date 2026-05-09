from __future__ import annotations

import warnings

from loguru import logger
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import cohen_kappa_score, mean_absolute_error, root_mean_squared_error


def compute_global_metrics(
    gold: list[float],
    pred: list[float],
) -> dict[str, float]:
    import warnings

    g = [float(v) for v in gold]
    p = [float(v) for v in pred]
    g_int = [round(v) for v in g]
    p_int = [max(1, min(5, round(v))) for v in p]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pear = float(pearsonr(g, p)[0])
        spear = float(spearmanr(g, p)[0])
    try:
        kappa = float(
            cohen_kappa_score(g_int, p_int, weights="quadratic", labels=list(range(1, 6)))
        )
    except Exception:
        kappa = float("nan")

    return {
        "mae":      float(mean_absolute_error(g, p)),
        "rmse":     float(root_mean_squared_error(g, p)),
        "pearson":  pear,
        "spearman": spear,
        "kappa":    kappa,
        "n_samples": len(g),
    }


def _fmt(v: float, fmt: str = ".4f") -> str:
    return f"{v:{fmt}}" if not (v != v) else "  N/A  "  # isnan via v!=v


def print_global_metrics(m: dict, label: str = "") -> None:
    sep = "=" * 60
    if label:
        logger.info(sep)
        logger.info(f"  {label}")
    logger.info(sep)
    logger.info(f"  样本数:  {m['n_samples']}")
    logger.info(f"  MAE:     {_fmt(m['mae'])}    RMSE:     {_fmt(m['rmse'])}")
    logger.info(f"  Pearson: {_fmt(m['pearson'])}    Spearman: {_fmt(m['spearman'])}")
    logger.info(f"  Kappa:   {_fmt(m['kappa'])}")
    logger.info(sep)


# ──────────────────────────────────────────────────────────────────────────────
# 3/4 边界（二分类 SAT/DSAT）指标
# ──────────────────────────────────────────────────────────────────────────────

