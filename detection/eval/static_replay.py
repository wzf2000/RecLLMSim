"""
Evaluate static replay benchmark scores.

Input is a scored JSONL produced by trace/score_static_replay.py. Gold labels
are optional; the primary benchmark score is the judge-predicted satisfaction
of each candidate model response.
"""

from __future__ import annotations

import json
import math
import os
import random
from argparse import ArgumentParser
from collections import defaultdict

from loguru import logger


def load_jsonl(path: str) -> list[dict]:
    records: list[dict] = []
    with open(path, "r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def _score_distribution(records: list[dict]) -> dict[str, int]:
    out = {str(i): 0 for i in range(1, 6)}
    for r in records:
        score = int(r["pred_score"])
        out[str(score)] += 1
    return out


def _group_means(records: list[dict], key_fn) -> dict[str, float]:
    groups: dict[str, list[float]] = defaultdict(list)
    for r in records:
        groups[key_fn(r)].append(float(r["pred_score"]))
    return {k: _mean(v) for k, v in groups.items()}


def _bootstrap_user_ci(
    records: list[dict],
    n_bootstrap: int,
    seed: int,
) -> dict[str, float]:
    by_user: dict[str, list[float]] = defaultdict(list)
    for r in records:
        by_user[r["user"]].append(float(r["pred_score"]))
    users = sorted(by_user)
    if not users or n_bootstrap <= 0:
        return {"low": float("nan"), "high": float("nan")}
    user_means = {u: _mean(v) for u, v in by_user.items()}
    rng = random.Random(seed)
    boot: list[float] = []
    for _ in range(n_bootstrap):
        sampled = [rng.choice(users) for _ in users]
        boot.append(_mean([user_means[u] for u in sampled]))
    boot.sort()
    low_i = max(0, min(len(boot) - 1, int(0.025 * len(boot))))
    high_i = max(0, min(len(boot) - 1, int(0.975 * len(boot))))
    return {"low": boot[low_i], "high": boot[high_i]}


def summarize_model(records: list[dict], n_bootstrap: int, seed: int) -> dict:
    scores = [float(r["pred_score"]) for r in records]
    sat = [s for s in scores if s >= 4]
    dsat = [s for s in scores if s <= 3]

    user_means = _group_means(records, lambda r: r["user"])
    task_means = _group_means(records, lambda r: r["target_task"])
    block_means = _group_means(records, lambda r: f"{r['user']}__{r['target_task']}")

    out = {
        "n_samples": len(records),
        "n_users": len(user_means),
        "n_tasks": len(task_means),
        "n_blocks": len(block_means),
        "micro_mean": _mean(scores),
        "user_macro_mean": _mean(list(user_means.values())),
        "task_macro_mean": _mean(list(task_means.values())),
        "user_task_macro_mean": _mean(list(block_means.values())),
        "sat_rate": len(sat) / len(scores) if scores else float("nan"),
        "dsat_rate": len(dsat) / len(scores) if scores else float("nan"),
        "score_distribution": _score_distribution(records),
        "task_means": task_means,
        "user_bootstrap_ci": _bootstrap_user_ci(records, n_bootstrap, seed),
    }
    if any(r.get("gold_score") is not None for r in records):
        gold = [float(r["gold_score"]) for r in records if r.get("gold_score") is not None]
        pred = [float(r["pred_score"]) for r in records if r.get("gold_score") is not None]
        if len(gold) == len(pred) and gold:
            out["diagnostic_mae_vs_original_gold"] = _mean([abs(g - p) for g, p in zip(gold, pred)])
    return out


def evaluate(records: list[dict], n_bootstrap: int, seed: int) -> dict:
    by_model: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_model[r["candidate_model"]].append(r)
    return {
        model: summarize_model(model_records, n_bootstrap=n_bootstrap, seed=seed)
        for model, model_records in sorted(by_model.items())
    }


def print_summary(results: dict) -> None:
    logger.info("=" * 96)
    logger.info("Static replay benchmark summary")
    logger.info("=" * 96)
    logger.info(
        f"{'candidate_model':36s} {'n':>6s} {'micro':>8s} {'user':>8s} "
        f"{'task':>8s} {'block':>8s} {'SAT':>8s} {'DSAT':>8s}"
    )
    for model, m in results.items():
        logger.info(
            f"{model[:36]:36s} {m['n_samples']:6d} "
            f"{m['micro_mean']:8.4f} {m['user_macro_mean']:8.4f} "
            f"{m['task_macro_mean']:8.4f} {m['user_task_macro_mean']:8.4f} "
            f"{m['sat_rate']:8.4f} {m['dsat_rate']:8.4f}"
        )


def parse_args() -> ArgumentParser:
    parser = ArgumentParser(description="Evaluate static replay scored responses")
    parser.add_argument("--input_jsonl", type=str, required=True)
    parser.add_argument("--output_json", type=str, default="")
    parser.add_argument("--n_bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    args = parse_args().parse_args()
    records = load_jsonl(args.input_jsonl)
    results = evaluate(records, n_bootstrap=args.n_bootstrap, seed=args.seed)
    print_summary(results)
    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as fp:
            json.dump(results, fp, ensure_ascii=False, indent=2)
        logger.info(f"Saved: {args.output_json}")


if __name__ == "__main__":
    main()
