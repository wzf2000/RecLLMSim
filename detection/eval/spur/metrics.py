from __future__ import annotations

from loguru import logger
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .constants import SAT_LABEL


def compute_metrics(results: list[dict]) -> dict[str, float]:
    gold = [1 if r["gold_label"] == SAT_LABEL else 0 for r in results]
    pred = [1 if r["pred_label"] == SAT_LABEL else 0 for r in results]
    conf = [r.get("confidence", 0.5) for r in results]
    parse_rate = sum(r.get("parse_ok", False) for r in results) / len(results)

    metrics = {
        "accuracy": accuracy_score(gold, pred),
        "f1_macro": f1_score(gold, pred, average="macro", zero_division=0),
        "f1_sat": f1_score(gold, pred, pos_label=1, average="binary", zero_division=0),
        "f1_dsat": f1_score(gold, pred, pos_label=0, average="binary", zero_division=0),
        "precision_sat": precision_score(gold, pred, pos_label=1, average="binary", zero_division=0),
        "recall_sat": recall_score(gold, pred, pos_label=1, average="binary", zero_division=0),
        "kappa": cohen_kappa_score(gold, pred),
        "auc": roc_auc_score(gold, conf) if len(set(gold)) > 1 else float("nan"),
        "parse_rate": parse_rate,
        "n_samples": len(results),
        "n_sat_gold": int(sum(gold)),
        "n_dsat_gold": int(len(gold) - sum(gold)),
    }
    return metrics


def print_metrics(metrics: dict[str, float], header: str = ""):
    sep = "=" * 55
    if header:
        logger.info(sep)
        logger.info(header)
    logger.info(sep)
    logger.info(f"  样本数:          {metrics['n_samples']}  (SAT={metrics['n_sat_gold']}, DSAT={metrics['n_dsat_gold']})")
    logger.info(f"  解析成功率:      {metrics['parse_rate']:.1%}")
    logger.info(f"  Accuracy:        {metrics['accuracy']:.4f}")
    logger.info(f"  F1-macro:        {metrics['f1_macro']:.4f}")
    logger.info(f"  F1-SAT:          {metrics['f1_sat']:.4f}  (P={metrics['precision_sat']:.4f}, R={metrics['recall_sat']:.4f})")
    logger.info(f"  F1-DSAT:         {metrics['f1_dsat']:.4f}")
    logger.info(f"  Kappa:           {metrics['kappa']:.4f}")
    logger.info(f"  AUC:             {metrics['auc']:.4f}")
    logger.info(sep)
