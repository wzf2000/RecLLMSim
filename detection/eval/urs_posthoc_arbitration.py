"""Post-hoc boundary arbitration for URS predictor outputs.

This module keeps a strong base predictor as the default and only uses an
auxiliary predictor when the two disagree across the 3/4 SAT/DSAT boundary.
It is intended for sparse URS session-level data where full score-distribution
calibration is too aggressive.
"""

from __future__ import annotations

import json
import os
from argparse import ArgumentParser
from collections import Counter, defaultdict

from loguru import logger

from eval.personalized.boundary_metrics import (
    compute_boundary_metrics,
    get_sat_confidence,
    to_binary_sat,
)
from eval.personalized.global_metrics import compute_global_metrics
from lib.user_aware_metrics import compute_user_aware_binary_metrics, compute_user_aware_metrics


DEFAULT_SELECTED_TASKS = ("leisure", "professional", "text", "other")


def load_jsonl(path: str) -> dict[str, dict]:
    records: dict[str, dict] = {}
    with open(path, "r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            records[obj["sample_id"]] = obj
    return records


def save_jsonl(records: list[dict], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fp:
        for record in records:
            fp.write(json.dumps(record, ensure_ascii=False) + "\n")


def _score(record: dict) -> int:
    return int(record["pred_score"])


def _is_sat(score: int) -> bool:
    return score >= 4


def _boundary_score_from_aux(aux_score: int) -> int:
    return 4 if _is_sat(aux_score) else 3


def _copy_base_record(base: dict) -> dict:
    out = dict(base)
    out["pred_score_raw"] = base.get("pred_score")
    out["reason_prediction_raw"] = base.get("reason_prediction")
    out["analysis_raw"] = base.get("analysis")
    return out


def arbitrate_one(
    base: dict,
    aux: dict,
    strategy: str,
    selected_tasks: set[str],
) -> tuple[dict, bool, str]:
    base_score = _score(base)
    aux_score = _score(aux)
    task = str(base.get("target_task", "unknown"))
    crosses_boundary = _is_sat(base_score) != _is_sat(aux_score)
    selected = task in selected_tasks

    out = _copy_base_record(base)
    out["posthoc_strategy"] = strategy
    out["posthoc_base_score"] = base_score
    out["posthoc_aux_score"] = aux_score
    out["posthoc_aux_prompt_version"] = aux.get("urs_prompt_version")
    out["posthoc_crosses_boundary"] = crosses_boundary
    out["posthoc_selected_task"] = selected

    if not crosses_boundary:
        out["posthoc_action"] = "keep_base_same_boundary"
        return out, False, out["posthoc_action"]

    if strategy == "all_boundary_aux":
        out["pred_score"] = _boundary_score_from_aux(aux_score)
        out["reason_prediction"] = aux.get("reason_prediction")
        out["analysis"] = (
            f"Post-hoc arbitration used auxiliary boundary score because base={base_score} "
            f"and aux={aux_score} crossed SAT/DSAT boundary. Aux analysis: {aux.get('analysis', '')}"
        )
        out["posthoc_action"] = "use_aux_boundary"
        return out, True, out["posthoc_action"]

    if not selected:
        out["posthoc_action"] = "keep_base_unselected_task"
        return out, False, out["posthoc_action"]

    if strategy == "selected_boundary_aux":
        out["pred_score"] = _boundary_score_from_aux(aux_score)
        out["reason_prediction"] = aux.get("reason_prediction")
        out["analysis"] = (
            f"Post-hoc arbitration used selected-task auxiliary boundary score because "
            f"base={base_score} and aux={aux_score} crossed SAT/DSAT boundary. "
            f"Aux analysis: {aux.get('analysis', '')}"
        )
        out["posthoc_action"] = "use_aux_boundary_selected_task"
        return out, True, out["posthoc_action"]

    if strategy == "selected_downgrade":
        if base_score == 4 and aux_score <= 3:
            out["pred_score"] = 3
            out["reason_prediction"] = aux.get("reason_prediction")
            out["analysis"] = (
                f"Post-hoc arbitration downgraded base 4 to 3 for selected task because "
                f"auxiliary predictor judged DSAT. Aux analysis: {aux.get('analysis', '')}"
            )
            out["posthoc_action"] = "downgrade_4_to_3_selected_task"
            return out, True, out["posthoc_action"]
        out["posthoc_action"] = "keep_base_not_selected_downgrade_case"
        return out, False, out["posthoc_action"]

    if strategy == "selected_upgrade":
        if base_score == 3 and aux_score >= 4:
            out["pred_score"] = 4
            out["reason_prediction"] = "满意"
            out["analysis"] = (
                f"Post-hoc arbitration upgraded base 3 to 4 for selected task because "
                f"auxiliary predictor judged SAT. Aux analysis: {aux.get('analysis', '')}"
            )
            out["posthoc_action"] = "upgrade_3_to_4_selected_task"
            return out, True, out["posthoc_action"]
        out["posthoc_action"] = "keep_base_not_selected_upgrade_case"
        return out, False, out["posthoc_action"]

    raise ValueError(f"Unknown strategy: {strategy}")


def build_arbitrated_records(
    base_path: str,
    aux_path: str,
    strategy: str,
    selected_tasks: set[str],
) -> tuple[list[dict], dict]:
    base_records = load_jsonl(base_path)
    aux_records = load_jsonl(aux_path)
    common_ids = sorted(set(base_records) & set(aux_records))
    action_counts: Counter[str] = Counter()
    changed = 0
    out: list[dict] = []
    for sample_id in common_ids:
        record, did_change, action = arbitrate_one(
            base_records[sample_id],
            aux_records[sample_id],
            strategy=strategy,
            selected_tasks=selected_tasks,
        )
        action_counts[action] += 1
        changed += int(did_change)
        out.append(record)
    stats = {
        "strategy": strategy,
        "n_base": len(base_records),
        "n_aux": len(aux_records),
        "n_common": len(common_ids),
        "n_changed": changed,
        "action_counts": dict(sorted(action_counts.items())),
        "selected_tasks": sorted(selected_tasks),
    }
    return out, stats


def _evaluate(records: list[dict], min_samples: int) -> dict:
    gold = [float(r["gold_score"]) for r in records]
    pred = [float(r["pred_score"]) for r in records]
    users = [str(r.get("user", "unknown")) for r in records]
    gold_bin = [to_binary_sat(r["gold_score"]) for r in records]
    pred_bin = [to_binary_sat(r["pred_score"]) for r in records]
    conf = [get_sat_confidence(r) for r in records]
    return {
        "n": len(records),
        "global": compute_global_metrics(gold, pred),
        "boundary": compute_boundary_metrics(records),
        "user_aware": compute_user_aware_metrics(gold, pred, users, min_samples=min_samples),
        "user_aware_boundary": compute_user_aware_binary_metrics(
            gold_bin,
            pred_bin,
            users,
            conf,
            min_samples=min_samples,
        ),
        "pred_distribution": dict(sorted(Counter(int(r["pred_score"]) for r in records).items())),
        "task_breakdown": _breakdown(records, lambda r: str(r.get("target_task", "unknown"))),
    }


def _breakdown(records: list[dict], key_fn) -> dict[str, dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        groups[key_fn(record)].append(record)
    out: dict[str, dict] = {}
    for key, group in sorted(groups.items()):
        gm = compute_global_metrics(
            [float(r["gold_score"]) for r in group],
            [float(r["pred_score"]) for r in group],
        )
        bm = compute_boundary_metrics(group)
        out[key] = {
            "n": len(group),
            "mae": gm["mae"],
            "pearson": gm["pearson"],
            "spearman": gm["spearman"],
            "qwk": gm["kappa"],
            "accuracy": bm["accuracy"],
            "f1_dsat": bm["f1_dsat"],
        }
    return out


def print_summary(results: dict[str, dict], stats_by_strategy: dict[str, dict]) -> None:
    logger.info("=" * 112)
    logger.info("URS post-hoc boundary arbitration")
    logger.info("=" * 112)
    logger.info(
        f"{'strategy':32s} {'chg':>5s} {'MAE':>8s} {'Pear':>8s} {'Spear':>8s} "
        f"{'QWK':>8s} {'Acc':>8s} {'F1-D':>8s} {'F1-S':>8s} {'FalseSAT':>9s}"
    )
    for name, result in results.items():
        gm = result["global"]
        bm = result["boundary"]
        logger.info(
            f"{name:32s} {stats_by_strategy[name]['n_changed']:5d} "
            f"{gm['mae']:8.4f} {gm['pearson']:8.4f} {gm['spearman']:8.4f} "
            f"{gm['kappa']:8.4f} {bm['accuracy']:8.4f} {bm['f1_dsat']:8.4f} "
            f"{bm['f1_sat']:8.4f} {bm['false_sat_rate']:9.4f}"
        )


def parse_args() -> ArgumentParser:
    parser = ArgumentParser(description="Post-hoc boundary arbitration for URS outputs.")
    parser.add_argument("--base_jsonl", type=str, required=True)
    parser.add_argument("--aux_jsonl", type=str, required=True)
    parser.add_argument(
        "--strategies",
        type=str,
        nargs="+",
        default=["selected_downgrade", "selected_boundary_aux", "all_boundary_aux"],
        choices=[
            "selected_downgrade",
            "selected_upgrade",
            "selected_boundary_aux",
            "all_boundary_aux",
        ],
    )
    parser.add_argument(
        "--selected_tasks",
        type=str,
        nargs="+",
        default=list(DEFAULT_SELECTED_TASKS),
    )
    parser.add_argument("--output_dir", type=str, default="")
    parser.add_argument("--output_json", type=str, default="")
    parser.add_argument("--min_samples", type=int, default=3)
    return parser


def main() -> None:
    args = parse_args().parse_args()
    selected_tasks = set(args.selected_tasks)
    output_dir = args.output_dir
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    results: dict[str, dict] = {}
    stats_by_strategy: dict[str, dict] = {}
    for strategy in args.strategies:
        records, stats = build_arbitrated_records(
            args.base_jsonl,
            args.aux_jsonl,
            strategy=strategy,
            selected_tasks=selected_tasks,
        )
        results[strategy] = _evaluate(records, min_samples=args.min_samples)
        stats_by_strategy[strategy] = stats
        if output_dir:
            output_path = os.path.join(output_dir, f"urs_posthoc_{strategy}.jsonl")
            save_jsonl(records, output_path)
            stats["output_jsonl"] = output_path

    print_summary(results, stats_by_strategy)
    combined = {
        name: {
            **results[name],
            "posthoc_stats": stats_by_strategy[name],
        }
        for name in results
    }
    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as fp:
            json.dump(combined, fp, ensure_ascii=False, indent=2)
        logger.info(f"Saved: {args.output_json}")


if __name__ == "__main__":
    main()
