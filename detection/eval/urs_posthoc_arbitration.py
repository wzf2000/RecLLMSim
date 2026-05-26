"""Post-hoc boundary arbitration for URS predictor outputs.

This module keeps a strong base predictor as the default and only uses an
auxiliary predictor when the two disagree across the 3/4 SAT/DSAT boundary.
It is intended for sparse URS session-level data where full score-distribution
calibration is too aggressive.
"""

from __future__ import annotations

import json
import math
import os
import random
from argparse import ArgumentParser
from collections import Counter, defaultdict
from itertools import combinations

from loguru import logger

from eval.personalized.boundary_metrics import (
    compute_boundary_metrics,
    get_sat_confidence,
    to_binary_sat,
)
from eval.personalized.global_metrics import compute_global_metrics
from lib.user_aware_metrics import compute_user_aware_binary_metrics, compute_user_aware_metrics


DEFAULT_SELECTED_TASKS = ("leisure", "professional", "text", "other")
ALL_TASKS = ("advice", "creative", "leisure", "other", "professional", "retrieval", "text")


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
    return build_arbitrated_records_from_maps(
        base_records,
        aux_records,
        common_ids=common_ids,
        strategy=strategy,
        selected_tasks=selected_tasks,
    )


def build_arbitrated_records_from_maps(
    base_records: dict[str, dict],
    aux_records: dict[str, dict],
    common_ids: list[str],
    strategy: str,
    selected_tasks: set[str],
) -> tuple[list[dict], dict]:
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


def _evaluate_light(records: list[dict]) -> dict:
    gold = [float(r["gold_score"]) for r in records]
    pred = [float(r["pred_score"]) for r in records]
    return {
        "n": len(records),
        "global": compute_global_metrics(gold, pred),
        "boundary": compute_boundary_metrics(records),
        "pred_distribution": dict(sorted(Counter(int(r["pred_score"]) for r in records).items())),
    }


def _breakdown(records: list[dict], key_fn) -> dict[str, dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        groups[key_fn(record)].append(record)
    out: dict[str, dict] = {}
    for key, group in sorted(groups.items()):
        gold = [float(r["gold_score"]) for r in group]
        pred = [float(r["pred_score"]) for r in group]
        if len(group) >= 2:
            gm = compute_global_metrics(gold, pred)
        else:
            err = abs(gold[0] - pred[0]) if group else math.nan
            gm = {
                "mae": err,
                "rmse": err,
                "pearson": math.nan,
                "spearman": math.nan,
                "kappa": math.nan,
                "n_samples": len(group),
            }
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


def _metric_value(result: dict, metric: str) -> float:
    def finite(value: float, fallback: float = -1e9) -> float:
        value = float(value)
        return fallback if math.isnan(value) else value

    if metric == "mae":
        return -float(result["global"]["mae"])
    if metric == "rmse":
        return -float(result["global"]["rmse"])
    if metric == "pearson":
        return finite(result["global"]["pearson"])
    if metric == "spearman":
        return finite(result["global"]["spearman"])
    if metric == "qwk":
        return finite(result["global"]["kappa"])
    if metric == "accuracy":
        return float(result["boundary"]["accuracy"])
    if metric == "f1_dsat":
        return float(result["boundary"]["f1_dsat"])
    if metric == "f1_macro":
        return float(result["boundary"]["f1_macro"])
    if metric == "balanced":
        boundary = result["boundary"]
        global_metrics = result["global"]
        return (
            float(boundary["f1_dsat"])
            + float(boundary["accuracy"])
            + finite(global_metrics["spearman"], fallback=0.0)
            + finite(global_metrics["kappa"], fallback=0.0)
            - 0.25 * float(global_metrics["mae"])
        )
    raise ValueError(f"Unknown search_metric: {metric}")


def _task_subsets(tasks: tuple[str, ...], max_tasks: int) -> list[set[str]]:
    out: list[set[str]] = []
    upper = min(max_tasks, len(tasks))
    for size in range(1, upper + 1):
        for combo in combinations(tasks, size):
            out.append(set(combo))
    return out


def run_grid_search(
    base_jsonl: str,
    aux_jsonl: str,
    strategies: list[str],
    candidate_tasks: tuple[str, ...],
    max_tasks: int,
    search_metric: str,
    min_samples: int,
) -> dict:
    trials: list[dict] = []
    for strategy in strategies:
        for selected_tasks in _task_subsets(candidate_tasks, max_tasks=max_tasks):
            records, stats = build_arbitrated_records(
                base_jsonl,
                aux_jsonl,
                strategy=strategy,
                selected_tasks=selected_tasks,
            )
            result = _evaluate_light(records)
            score = _metric_value(result, search_metric)
            trials.append({
                "strategy": strategy,
                "selected_tasks": sorted(selected_tasks),
                "search_metric": search_metric,
                "search_score": score,
                "posthoc_stats": stats,
                "metrics": result,
            })

    trials.sort(
        key=lambda item: (
            item["search_score"],
            item["metrics"]["boundary"]["f1_dsat"],
            item["metrics"]["boundary"]["accuracy"],
            -len(item["selected_tasks"]),
        ),
        reverse=True,
    )
    return {
        "best": trials[0] if trials else None,
        "top_trials": trials[:20],
        "n_trials": len(trials),
        "candidate_tasks": list(candidate_tasks),
        "max_tasks": max_tasks,
        "search_metric": search_metric,
    }


def _make_user_folds(users: list[str], n_folds: int, seed: int) -> list[set[str]]:
    users = list(users)
    random.Random(seed).shuffle(users)
    n_folds = max(2, min(n_folds, len(users)))
    folds = [set() for _ in range(n_folds)]
    for idx, user in enumerate(users):
        folds[idx % n_folds].add(user)
    return folds


def _records_for_ids(
    base_records: dict[str, dict],
    aux_records: dict[str, dict],
    sample_ids: list[str],
    strategy: str,
    selected_tasks: set[str],
) -> tuple[list[dict], dict]:
    return build_arbitrated_records_from_maps(
        base_records,
        aux_records,
        common_ids=sample_ids,
        strategy=strategy,
        selected_tasks=selected_tasks,
    )


def _search_best_on_ids(
    base_records: dict[str, dict],
    aux_records: dict[str, dict],
    train_ids: list[str],
    strategies: list[str],
    candidate_tasks: tuple[str, ...],
    max_tasks: int,
    search_metric: str,
    min_samples: int,
) -> dict:
    trials: list[dict] = []
    for strategy in strategies:
        for selected_tasks in _task_subsets(candidate_tasks, max_tasks=max_tasks):
            records, stats = _records_for_ids(
                base_records,
                aux_records,
                train_ids,
                strategy=strategy,
                selected_tasks=selected_tasks,
            )
            result = _evaluate_light(records)
            trials.append({
                "strategy": strategy,
                "selected_tasks": sorted(selected_tasks),
                "search_metric": search_metric,
                "search_score": _metric_value(result, search_metric),
                "posthoc_stats": stats,
                "metrics": result,
            })
    trials.sort(
        key=lambda item: (
            item["search_score"],
            item["metrics"]["boundary"]["f1_dsat"],
            item["metrics"]["boundary"]["accuracy"],
            -len(item["selected_tasks"]),
        ),
        reverse=True,
    )
    return trials[0]


def _base_records_for_ids(records_by_id: dict[str, dict], sample_ids: list[str]) -> list[dict]:
    return [dict(records_by_id[sample_id]) for sample_id in sample_ids]


def _mean_metric(fold_results: list[dict], section: str, key: str) -> float:
    values = [float(r[section][key]) for r in fold_results]
    values = [v for v in values if not math.isnan(v)]
    return sum(values) / len(values) if values else math.nan


def _summarize_cv_fold_results(fold_results: list[dict]) -> dict:
    return {
        "n_folds": len(fold_results),
        "global": {
            key: _mean_metric(fold_results, "global", key)
            for key in ["mae", "rmse", "pearson", "spearman", "kappa"]
        },
        "boundary": {
            key: _mean_metric(fold_results, "boundary", key)
            for key in [
                "accuracy",
                "f1_macro",
                "f1_sat",
                "f1_dsat",
                "kappa",
                "auc",
                "false_sat_rate",
                "false_dsat_rate",
            ]
        },
    }


def run_user_cv_search(
    base_jsonl: str,
    aux_jsonl: str,
    strategies: list[str],
    candidate_tasks: tuple[str, ...],
    max_tasks: int,
    search_metric: str,
    min_samples: int,
    n_folds: int,
    seed: int,
) -> dict:
    base_records = load_jsonl(base_jsonl)
    aux_records = load_jsonl(aux_jsonl)
    common_ids = sorted(set(base_records) & set(aux_records))
    users = sorted({str(base_records[sample_id].get("user", "unknown")) for sample_id in common_ids})
    folds = _make_user_folds(users, n_folds=n_folds, seed=seed)

    fold_outputs: list[dict] = []
    selected_eval_results: list[dict] = []
    base_eval_results: list[dict] = []
    aux_eval_results: list[dict] = []

    for fold_idx, valid_users in enumerate(folds):
        train_ids = [
            sample_id for sample_id in common_ids
            if str(base_records[sample_id].get("user", "unknown")) not in valid_users
        ]
        valid_ids = [
            sample_id for sample_id in common_ids
            if str(base_records[sample_id].get("user", "unknown")) in valid_users
        ]
        best = _search_best_on_ids(
            base_records,
            aux_records,
            train_ids=train_ids,
            strategies=strategies,
            candidate_tasks=candidate_tasks,
            max_tasks=max_tasks,
            search_metric=search_metric,
            min_samples=min_samples,
        )
        selected_tasks = set(best["selected_tasks"])
        selected_records, selected_stats = _records_for_ids(
            base_records,
            aux_records,
            valid_ids,
            strategy=best["strategy"],
            selected_tasks=selected_tasks,
        )
        selected_eval = _evaluate(selected_records, min_samples=min_samples)
        base_eval = _evaluate(_base_records_for_ids(base_records, valid_ids), min_samples=min_samples)
        aux_eval = _evaluate(_base_records_for_ids(aux_records, valid_ids), min_samples=min_samples)
        selected_eval_results.append(selected_eval)
        base_eval_results.append(base_eval)
        aux_eval_results.append(aux_eval)
        fold_outputs.append({
            "fold": fold_idx,
            "valid_users": sorted(valid_users),
            "n_train": len(train_ids),
            "n_valid": len(valid_ids),
            "selected_rule": {
                "strategy": best["strategy"],
                "selected_tasks": best["selected_tasks"],
                "train_search_score": best["search_score"],
                "train_metrics": best["metrics"],
                "train_posthoc_stats": best["posthoc_stats"],
            },
            "valid_posthoc_stats": selected_stats,
            "valid_metrics": selected_eval,
            "valid_base_metrics": base_eval,
            "valid_aux_metrics": aux_eval,
        })

    summary = {
        "selected": _summarize_cv_fold_results(selected_eval_results),
        "base": _summarize_cv_fold_results(base_eval_results),
        "aux": _summarize_cv_fold_results(aux_eval_results),
    }
    return {
        "n_records": len(common_ids),
        "n_users": len(users),
        "n_folds": len(folds),
        "seed": seed,
        "search_metric": search_metric,
        "candidate_tasks": list(candidate_tasks),
        "max_tasks": max_tasks,
        "strategies": strategies,
        "summary": summary,
        "folds": fold_outputs,
    }


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
    parser.add_argument("--grid_search", action="store_true")
    parser.add_argument("--cv_search", action="store_true")
    parser.add_argument(
        "--candidate_tasks",
        type=str,
        nargs="+",
        default=list(ALL_TASKS),
        choices=list(ALL_TASKS),
    )
    parser.add_argument("--max_tasks", type=int, default=4)
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--cv_seed", type=int, default=42)
    parser.add_argument(
        "--search_metric",
        type=str,
        default="balanced",
        choices=[
            "balanced",
            "mae",
            "rmse",
            "pearson",
            "spearman",
            "qwk",
            "accuracy",
            "f1_dsat",
            "f1_macro",
        ],
    )
    return parser


def main() -> None:
    args = parse_args().parse_args()
    selected_tasks = set(args.selected_tasks)
    output_dir = args.output_dir
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    if args.cv_search:
        result = run_user_cv_search(
            args.base_jsonl,
            args.aux_jsonl,
            strategies=args.strategies,
            candidate_tasks=tuple(args.candidate_tasks),
            max_tasks=args.max_tasks,
            search_metric=args.search_metric,
            min_samples=args.min_samples,
            n_folds=args.n_folds,
            seed=args.cv_seed,
        )
        selected = result["summary"]["selected"]
        base = result["summary"]["base"]
        logger.info(
            "CV selected summary: "
            f"MAE={selected['global']['mae']:.4f}, "
            f"Spearman={selected['global']['spearman']:.4f}, "
            f"QWK={selected['global']['kappa']:.4f}, "
            f"Acc={selected['boundary']['accuracy']:.4f}, "
            f"F1-DSAT={selected['boundary']['f1_dsat']:.4f}, "
            f"FalseSAT={selected['boundary']['false_sat_rate']:.4f}"
        )
        logger.info(
            "CV base summary: "
            f"MAE={base['global']['mae']:.4f}, "
            f"Spearman={base['global']['spearman']:.4f}, "
            f"QWK={base['global']['kappa']:.4f}, "
            f"Acc={base['boundary']['accuracy']:.4f}, "
            f"F1-DSAT={base['boundary']['f1_dsat']:.4f}, "
            f"FalseSAT={base['boundary']['false_sat_rate']:.4f}"
        )
        if args.output_json:
            os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
            with open(args.output_json, "w", encoding="utf-8") as fp:
                json.dump(result, fp, ensure_ascii=False, indent=2)
            logger.info(f"Saved: {args.output_json}")
        return

    if args.grid_search:
        result = run_grid_search(
            args.base_jsonl,
            args.aux_jsonl,
            strategies=args.strategies,
            candidate_tasks=tuple(args.candidate_tasks),
            max_tasks=args.max_tasks,
            search_metric=args.search_metric,
            min_samples=args.min_samples,
        )
        best = result.get("best")
        if best:
            gm = best["metrics"]["global"]
            bm = best["metrics"]["boundary"]
            logger.info(
                "Best arbitration rule: "
                f"strategy={best['strategy']}, tasks={best['selected_tasks']}, "
                f"score={best['search_score']:.4f}, MAE={gm['mae']:.4f}, "
                f"Spearman={gm['spearman']:.4f}, QWK={gm['kappa']:.4f}, "
                f"Acc={bm['accuracy']:.4f}, F1-DSAT={bm['f1_dsat']:.4f}"
            )
        if args.output_json:
            os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
            with open(args.output_json, "w", encoding="utf-8") as fp:
                json.dump(result, fp, ensure_ascii=False, indent=2)
            logger.info(f"Saved: {args.output_json}")
        return

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
