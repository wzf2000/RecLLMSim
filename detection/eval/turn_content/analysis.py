from __future__ import annotations

import json
import math
import os
from collections import defaultdict

from loguru import logger
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    f1_score,
    mean_absolute_error,
    precision_score,
    recall_score,
)

from .io import load_jsonl


def _sat(score: int | float) -> int:
    return int(int(score) >= 4)


def _metrics(records: list[dict]) -> dict:
    if not records:
        return {"n": 0}
    gold = [int(r["gold_score"]) for r in records]
    pred = [int(r["pred_score"]) for r in records]
    gold_bin = [_sat(x) for x in gold]
    pred_bin = [_sat(x) for x in pred]
    n_dsat = sum(1 for x in gold_bin if x == 0)
    n_sat = sum(1 for x in gold_bin if x == 1)
    out = {
        "n": len(records),
        "gold_mean": sum(gold) / len(gold),
        "pred_mean": sum(pred) / len(pred),
        "gold_dsat_rate": n_dsat / len(records),
        "pred_dsat_rate": sum(1 for x in pred_bin if x == 0) / len(records),
        "mae": float(mean_absolute_error(gold, pred)),
        "rmse": math.sqrt(sum((g - p) ** 2 for g, p in zip(gold, pred)) / len(records)),
        "bin_acc": float(accuracy_score(gold_bin, pred_bin)),
        "f1_dsat": float(f1_score(gold_bin, pred_bin, pos_label=0, zero_division=0)),
        "precision_dsat": float(precision_score(gold_bin, pred_bin, pos_label=0, zero_division=0)),
        "recall_dsat": float(recall_score(gold_bin, pred_bin, pos_label=0, zero_division=0)),
        "false_sat": (
            sum(1 for g, p in zip(gold_bin, pred_bin) if g == 0 and p == 1) / n_dsat
            if n_dsat else None
        ),
        "false_dsat": (
            sum(1 for g, p in zip(gold_bin, pred_bin) if g == 1 and p == 0) / n_sat
            if n_sat else None
        ),
    }
    out["qwk"] = (
        float(cohen_kappa_score(gold, pred, weights="quadratic", labels=[1, 2, 3, 4, 5]))
        if len(set(pred)) > 1 else None
    )
    out["pearson"] = float(pearsonr(gold, pred)[0]) if len(set(pred)) > 1 else None
    out["spearman"] = float(spearmanr(gold, pred)[0]) if len(set(pred)) > 1 else None
    return out


def _parse_named_files(items: list[str]) -> list[tuple[str, str]]:
    named: list[tuple[str, str]] = []
    for item in items:
        if "=" in item:
            name, path = item.split("=", 1)
        else:
            name = os.path.splitext(os.path.basename(item))[0]
            path = item
        named.append((name, path))
    return named


def command_analyze(args) -> None:
    annotations = load_jsonl(args.annotations_jsonl)
    ann_by_id = {str(r["sample_id"]): r for r in annotations}
    logger.info(f"Loaded annotations: {len(ann_by_id)}")

    all_results: dict[str, dict] = {}
    for name, path in _parse_named_files(args.result_files):
        records = load_jsonl(path)
        joined: list[dict] = []
        missing = 0
        for record in records:
            ann = ann_by_id.get(str(record.get("sample_id")))
            if ann is None:
                missing += 1
                continue
            joined.append({**record, **{f"content_{k}": v for k, v in ann.items()}})

        groups: dict[str, list[dict]] = {
            "all_annotated": joined,
            "content_like": [r for r in joined if bool(r.get("content_has_substantive_content"))],
            "non_content": [r for r in joined if not bool(r.get("content_has_substantive_content"))],
        }
        by_type: dict[str, list[dict]] = defaultdict(list)
        by_turn_bucket: dict[str, list[dict]] = defaultdict(list)
        for record in joined:
            by_type[str(record.get("content_content_type", "unknown"))].append(record)
            turn_idx = int(record.get("turn_idx", 0))
            bucket = str(turn_idx) if turn_idx < 5 else "5+"
            by_turn_bucket[bucket].append(record)

        all_results[name] = {
            "path": path,
            "n_joined": len(joined),
            "n_missing_annotations": missing,
            "groups": {group: _metrics(items) for group, items in groups.items()},
            "by_content_type": {group: _metrics(items) for group, items in sorted(by_type.items())},
            "by_turn": {
                group: _metrics(by_turn_bucket[group])
                for group in ["0", "1", "2", "3", "4", "5+"]
                if group in by_turn_bucket
            },
        }

    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved analysis: {args.output_json}")

    for name, result in all_results.items():
        logger.info(f"\n{name}: joined={result['n_joined']} missing={result['n_missing_annotations']}")
        for group in ("content_like", "non_content"):
            m = result["groups"][group]
            if m["n"] == 0:
                continue
            logger.info(
                f"  {group:<13} n={m['n']} dsat={m['gold_dsat_rate']:.3f} "
                f"MAE={m['mae']:.4f} QWK={m['qwk']} F1-DSAT={m['f1_dsat']:.4f}"
            )
