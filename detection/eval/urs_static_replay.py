"""Evaluate URS static replay benchmark scores."""

from __future__ import annotations

import json
import os
from argparse import ArgumentParser
from collections import defaultdict

from loguru import logger

from eval.static_replay import evaluate, load_jsonl, print_summary


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def _language(user: str) -> str:
    return user.split("_", 1)[0] if "_" in user else "unknown"


def add_urs_breakdowns(records: list[dict], results: dict) -> dict:
    by_model: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        by_model[record["candidate_model"]].append(record)

    for model, model_records in by_model.items():
        lang_groups: dict[str, list[float]] = defaultdict(list)
        granularity_groups: dict[str, list[float]] = defaultdict(list)
        context_groups: dict[str, list[float]] = defaultdict(list)
        for record in model_records:
            score = float(record["pred_score"])
            lang_groups[_language(record["user"])].append(score)
            granularity_groups[record.get("replay_granularity", "unknown")].append(score)
            context_groups[record.get("replay_context_mode", "unknown")].append(score)
        if model not in results:
            continue
        results[model]["language_means"] = {
            key: _mean(values) for key, values in sorted(lang_groups.items())
        }
        results[model]["replay_granularity_means"] = {
            key: _mean(values) for key, values in sorted(granularity_groups.items())
        }
        results[model]["replay_context_means"] = {
            key: _mean(values) for key, values in sorted(context_groups.items())
        }
    return results


def parse_args() -> ArgumentParser:
    parser = ArgumentParser(description="Evaluate URS static replay scored responses")
    parser.add_argument("--input_jsonl", type=str, required=True)
    parser.add_argument("--output_json", type=str, default="")
    parser.add_argument("--n_bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    args = parse_args().parse_args()
    records = load_jsonl(args.input_jsonl)
    results = evaluate(records, n_bootstrap=args.n_bootstrap, seed=args.seed)
    results = add_urs_breakdowns(records, results)
    print_summary(results)
    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as fp:
            json.dump(results, fp, ensure_ascii=False, indent=2)
        logger.info(f"Saved: {args.output_json}")


if __name__ == "__main__":
    main()
