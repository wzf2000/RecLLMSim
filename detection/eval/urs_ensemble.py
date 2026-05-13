"""Evaluate offline ensembles of URS satisfaction predictor outputs."""

from __future__ import annotations

import json
import math
import os
import statistics
from argparse import ArgumentParser
from collections import Counter, defaultdict
from dataclasses import dataclass

from loguru import logger

from eval.personalized.boundary_metrics import (
    compute_boundary_metrics,
    get_sat_confidence,
    to_binary_sat,
)
from eval.personalized.global_metrics import compute_global_metrics
from lib.user_aware_metrics import compute_user_aware_binary_metrics, compute_user_aware_metrics


@dataclass(frozen=True)
class RunSpec:
    name: str
    path: str
    weight: float = 1.0


def load_jsonl(path: str) -> dict[str, dict]:
    records: dict[str, dict] = {}
    with open(path, "r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            records.setdefault(obj["sample_id"], obj)
    return records


def parse_run_specs(items: list[str]) -> list[RunSpec]:
    specs: list[RunSpec] = []
    for item in items:
        if "=" not in item:
            raise ValueError(f"Expected name=path or name=weight=path, got: {item}")
        name, rest = item.split("=", 1)
        weight = 1.0
        path = rest
        if "=" in rest:
            weight_text, path = rest.split("=", 1)
            weight = float(weight_text)
        specs.append(RunSpec(name=name, path=path, weight=weight))
    return specs


def _round_score(value: float) -> int:
    return max(1, min(5, int(round(value))))


def _weighted_mean(scores: list[float], weights: list[float]) -> float:
    denom = sum(weights)
    if denom <= 0:
        return statistics.mean(scores)
    return sum(s * w for s, w in zip(scores, weights)) / denom


def _majority_sat_score(scores: list[int], weights: list[float]) -> int:
    sat_weight = sum(w for s, w in zip(scores, weights) if s >= 4)
    dsat_weight = sum(w for s, w in zip(scores, weights) if s <= 3)
    if sat_weight > dsat_weight:
        sat_scores = [s for s in scores if s >= 4]
        return _round_score(statistics.median(sat_scores)) if sat_scores else 4
    if dsat_weight > sat_weight:
        dsat_scores = [s for s in scores if s <= 3]
        return _round_score(statistics.median(dsat_scores)) if dsat_scores else 3
    return 4 if statistics.mean(scores) >= 3.5 else 3


def _confidence(scores: list[int]) -> tuple[str, dict]:
    span = max(scores) - min(scores)
    sat_votes = sum(1 for s in scores if s >= 4)
    dsat_votes = len(scores) - sat_votes
    all_same_side = sat_votes == 0 or dsat_votes == 0
    if span <= 1 and all_same_side:
        label = "high"
    elif span <= 2 or all_same_side:
        label = "medium"
    else:
        label = "low"
    return label, {
        "score_span": span,
        "sat_votes": sat_votes,
        "dsat_votes": dsat_votes,
    }


def build_ensemble_records(
    specs: list[RunSpec],
    strategy: str,
) -> list[dict]:
    by_run = {spec.name: load_jsonl(spec.path) for spec in specs}
    common_ids = set.intersection(*(set(records) for records in by_run.values()))
    missing = {
        name: len(common_ids.symmetric_difference(set(records)))
        for name, records in by_run.items()
    }
    logger.info(f"Common sample IDs: {len(common_ids)}")
    logger.info(f"Run non-overlap counts: {missing}")

    weights = [spec.weight for spec in specs]
    records: list[dict] = []
    for sample_id in sorted(common_ids):
        base = by_run[specs[0].name][sample_id]
        scores = [int(by_run[spec.name][sample_id]["pred_score"]) for spec in specs]
        score_float: float
        if strategy == "mean":
            score_float = _weighted_mean([float(s) for s in scores], weights)
            pred_score = _round_score(score_float)
        elif strategy == "median":
            score_float = float(statistics.median(scores))
            pred_score = _round_score(score_float)
        elif strategy == "majority_sat":
            pred_score = _majority_sat_score(scores, weights)
            score_float = float(pred_score)
        elif strategy == "mean_if_confident":
            label, _ = _confidence(scores)
            if label == "low":
                continue
            score_float = _weighted_mean([float(s) for s in scores], weights)
            pred_score = _round_score(score_float)
        else:
            raise ValueError(f"Unknown ensemble strategy: {strategy}")

        confidence_label, confidence_info = _confidence(scores)
        record = {
            "sample_id": sample_id,
            "user": base.get("user"),
            "target_task": base.get("target_task"),
            "target_file": base.get("target_file"),
            "turn_idx": base.get("turn_idx", 0),
            "gold_score": int(base["gold_score"]),
            "gold_reason": base.get("gold_reason"),
            "pred_score": pred_score,
            "pred_score_float": score_float,
            "reason_prediction": "ensemble",
            "model": f"urs_ensemble_{strategy}",
            "dataset": "urs",
            "ensemble_strategy": strategy,
            "ensemble_confidence": confidence_label,
            "ensemble_scores": {
                spec.name: int(by_run[spec.name][sample_id]["pred_score"])
                for spec in specs
            },
            "ensemble_weights": {spec.name: spec.weight for spec in specs},
            **confidence_info,
        }
        records.append(record)
    return records


def _evaluate_records(records: list[dict], min_samples: int) -> dict:
    gold = [float(r["gold_score"]) for r in records]
    pred = [float(r["pred_score"]) for r in records]
    users = [str(r.get("user", "unknown")) for r in records]
    global_metrics = compute_global_metrics(gold, pred)
    boundary_metrics = compute_boundary_metrics(records)
    user_aware = compute_user_aware_metrics(gold, pred, users, min_samples=min_samples)
    gold_bin = [to_binary_sat(r["gold_score"]) for r in records]
    pred_bin = [to_binary_sat(r["pred_score"]) for r in records]
    conf = [get_sat_confidence(r) for r in records]
    user_aware_boundary = compute_user_aware_binary_metrics(
        gold_bin,
        pred_bin,
        users,
        conf,
        min_samples=min_samples,
    )
    by_conf: dict[str, int] = Counter(r["ensemble_confidence"] for r in records)
    return {
        "n": len(records),
        "global": global_metrics,
        "boundary": boundary_metrics,
        "user_aware": user_aware,
        "user_aware_boundary": user_aware_boundary,
        "pred_distribution": dict(sorted(Counter(int(r["pred_score"]) for r in records).items())),
        "gold_distribution": dict(sorted(Counter(int(r["gold_score"]) for r in records).items())),
        "confidence_distribution": dict(sorted(by_conf.items())),
        "language_breakdown": _breakdown(records, lambda r: str(r["user"]).split("_", 1)[0], min_samples),
        "task_breakdown": _breakdown(records, lambda r: str(r.get("target_task", "unknown")), min_samples),
    }


def _breakdown(records: list[dict], key_fn, min_samples: int) -> dict[str, dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        groups[key_fn(record)].append(record)
    out: dict[str, dict] = {}
    for key, group in sorted(groups.items()):
        gold = [float(r["gold_score"]) for r in group]
        pred = [float(r["pred_score"]) for r in group]
        gm = compute_global_metrics(gold, pred)
        bm = compute_boundary_metrics(group)
        out[key] = {
            "n": len(group),
            "mae": gm["mae"],
            "pearson": gm["pearson"],
            "spearman": gm["spearman"],
            "qwk": gm["kappa"],
            "f1_dsat": bm["f1_dsat"],
            "pred_distribution": dict(sorted(Counter(int(r["pred_score"]) for r in group).items())),
        }
    return out


def print_summary(results: dict[str, dict]) -> None:
    logger.info("=" * 112)
    logger.info("URS ensemble predictor comparison")
    logger.info("=" * 112)
    logger.info(
        f"{'strategy':20s} {'n':>6s} {'MAE':>8s} {'Pear':>8s} {'Spear':>8s} "
        f"{'QWK':>8s} {'Acc':>8s} {'F1-D':>8s} {'F1-S':>8s} {'conf':>18s}"
    )
    for name, result in results.items():
        gm = result["global"]
        bm = result["boundary"]
        conf = result["confidence_distribution"]
        logger.info(
            f"{name:20s} {result['n']:6d} {gm['mae']:8.4f} {gm['pearson']:8.4f} "
            f"{gm['spearman']:8.4f} {gm['kappa']:8.4f} {bm['accuracy']:8.4f} "
            f"{bm['f1_dsat']:8.4f} {bm['f1_sat']:8.4f} {str(conf):>18s}"
        )


def parse_args() -> ArgumentParser:
    parser = ArgumentParser(description="Evaluate offline URS predictor ensembles.")
    parser.add_argument(
        "--runs",
        type=str,
        nargs="+",
        required=True,
        help="Run specs: name=path or name=weight=path.",
    )
    parser.add_argument(
        "--strategies",
        type=str,
        nargs="+",
        default=["mean", "median", "majority_sat", "mean_if_confident"],
        choices=["mean", "median", "majority_sat", "mean_if_confident"],
    )
    parser.add_argument("--output_json", type=str, default="")
    parser.add_argument("--output_jsonl", type=str, default="")
    parser.add_argument("--min_samples", type=int, default=3)
    return parser


def main() -> None:
    args = parse_args().parse_args()
    specs = parse_run_specs(args.runs)
    results: dict[str, dict] = {}
    all_records: dict[str, list[dict]] = {}
    for strategy in args.strategies:
        records = build_ensemble_records(specs, strategy=strategy)
        all_records[strategy] = records
        results[strategy] = _evaluate_records(records, min_samples=args.min_samples)
    print_summary(results)

    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as fp:
            json.dump(results, fp, ensure_ascii=False, indent=2)
        logger.info(f"Saved: {args.output_json}")
    if args.output_jsonl:
        os.makedirs(os.path.dirname(args.output_jsonl) or ".", exist_ok=True)
        with open(args.output_jsonl, "w", encoding="utf-8") as fp:
            for strategy, records in all_records.items():
                for record in records:
                    out = dict(record)
                    out["ensemble_strategy"] = strategy
                    fp.write(json.dumps(out, ensure_ascii=False) + "\n")
        logger.info(f"Saved: {args.output_jsonl}")


if __name__ == "__main__":
    main()
