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

from .global_metrics import _fmt


SAT_LABEL = 1   # score >= 4
DSAT_LABEL = 0  # score <= 3


def to_binary_sat(score: int | float | None) -> int:
    if score is None:
        return DSAT_LABEL
    return SAT_LABEL if int(score) >= 4 else DSAT_LABEL


def get_sat_confidence(record: dict) -> float:
    """
    由 1-5 的 pred_score 构造一个 SAT 置信度（用于 AUC / user-aware binary）。
    采用与 eval/binary_sat.py 相同的简单单调映射：
      1->0.1, 2->0.3, 3->0.5, 4->0.7, 5->0.9
    """
    pred = int(record.get("pred_score", 3))
    return (pred - 1) / 4 * 0.8 + 0.1


def compute_boundary_metrics(records: list[dict]) -> dict[str, float]:
    gold = [to_binary_sat(r["gold_score"]) for r in records]
    pred = [to_binary_sat(r["pred_score"]) for r in records]
    conf = [get_sat_confidence(r) for r in records]

    try:
        auc = float(roc_auc_score(gold, conf))
    except Exception:
        auc = float("nan")

    n_sat_gold = int(sum(gold))
    n_dsat_gold = int(len(gold) - n_sat_gold)
    n_sat_pred = int(sum(pred))
    n_dsat_pred = int(len(pred) - n_sat_pred)

    false_sat = 0
    false_dsat = 0
    for g, p in zip(gold, pred):
        if g == DSAT_LABEL and p == SAT_LABEL:
            false_sat += 1
        elif g == SAT_LABEL and p == DSAT_LABEL:
            false_dsat += 1

    return {
        "n_samples": len(records),
        "n_sat_gold": n_sat_gold,
        "n_dsat_gold": n_dsat_gold,
        "n_sat_pred": n_sat_pred,
        "n_dsat_pred": n_dsat_pred,
        "accuracy": float(accuracy_score(gold, pred)),
        "f1_macro": float(f1_score(gold, pred, average="macro", zero_division=0)),
        "f1_sat": float(f1_score(gold, pred, pos_label=1, average="binary", zero_division=0)),
        "f1_dsat": float(f1_score(gold, pred, pos_label=0, average="binary", zero_division=0)),
        "precision_sat": float(precision_score(gold, pred, pos_label=1, average="binary", zero_division=0)),
        "recall_sat": float(recall_score(gold, pred, pos_label=1, average="binary", zero_division=0)),
        "precision_dsat": float(precision_score(gold, pred, pos_label=0, average="binary", zero_division=0)),
        "recall_dsat": float(recall_score(gold, pred, pos_label=0, average="binary", zero_division=0)),
        "kappa": float(cohen_kappa_score(gold, pred)),
        "auc": auc,
        "false_sat_rate": float(false_sat / n_dsat_gold) if n_dsat_gold else float("nan"),
        "false_dsat_rate": float(false_dsat / n_sat_gold) if n_sat_gold else float("nan"),
    }


def print_boundary_metrics(m: dict, label: str = "") -> None:
    sep = "=" * 60
    if label:
        logger.info(sep)
        logger.info(f"  {label}")
    logger.info(sep)
    logger.info(
        f"  样本数: {m['n_samples']}  "
        f"(gold SAT={m['n_sat_gold']}, DSAT={m['n_dsat_gold']}; "
        f"pred SAT={m['n_sat_pred']}, DSAT={m['n_dsat_pred']})"
    )
    logger.info(f"  Accuracy:  {_fmt(m['accuracy'])}    F1-macro: {_fmt(m['f1_macro'])}")
    logger.info(
        f"  F1-SAT:    {_fmt(m['f1_sat'])}    "
        f"Prec-SAT: {_fmt(m['precision_sat'])}    Rec-SAT: {_fmt(m['recall_sat'])}"
    )
    logger.info(
        f"  F1-DSAT:   {_fmt(m['f1_dsat'])}    "
        f"Prec-DSAT: {_fmt(m['precision_dsat'])}    Rec-DSAT: {_fmt(m['recall_dsat'])}"
    )
    logger.info(
        f"  Kappa:     {_fmt(m['kappa'])}    AUC: {_fmt(m['auc'])}"
    )
    logger.info(
        f"  False SAT（把不满意判成满意）: {_fmt(m['false_sat_rate'])}"
        f"    False DSAT（把满意判成不满意）: {_fmt(m['false_dsat_rate'])}"
    )
    logger.info(sep)


# ──────────────────────────────────────────────────────────────────────────────
# 分层分析
# ──────────────────────────────────────────────────────────────────────────────

