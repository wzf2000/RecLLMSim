"""Pairwise win/tie/lose comparison for static replay scored outputs."""

from __future__ import annotations

import json
import os
import sys
from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path

from loguru import logger

_DETECTION_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _DETECTION_DIR not in sys.path:
    sys.path.insert(0, _DETECTION_DIR)


def load_jsonl(path: str) -> list[dict]:
    records: list[dict] = []
    with open(path, encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def _candidate_name(records: list[dict], fallback: str) -> str:
    names = sorted({str(r.get("candidate_model", "")) for r in records if r.get("candidate_model")})
    return names[0] if len(names) == 1 else fallback


def _by_sample_id(records: list[dict]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for record in records:
        sid = record.get("sample_id")
        if sid is not None:
            out[str(sid)] = record
    return out


def _compare_values(candidate: float, reference: float, tie_margin: float) -> int:
    delta = candidate - reference
    if delta > tie_margin:
        return 1
    if delta < -tie_margin:
        return -1
    return 0


def _score(record: dict) -> float:
    return float(record.get("pred_score", 3))


def _block_key(record: dict) -> str:
    return f"{record.get('user')}__{record.get('target_task')}"


def _ratio_summary(comparisons: list[int]) -> dict[str, float | int]:
    n = len(comparisons)
    wins = sum(1 for v in comparisons if v > 0)
    ties = sum(1 for v in comparisons if v == 0)
    losses = sum(1 for v in comparisons if v < 0)
    decided = wins + losses
    return {
        "n": n,
        "wins": wins,
        "ties": ties,
        "losses": losses,
        "win_rate": wins / n if n else float("nan"),
        "tie_rate": ties / n if n else float("nan"),
        "lose_rate": losses / n if n else float("nan"),
        "non_tie_win_rate": wins / decided if decided else float("nan"),
    }


def _macro_pairwise(
    paired: list[tuple[dict, dict]],
    group_fn,
    tie_margin: float,
) -> dict[str, float | int]:
    groups: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for cand, ref in paired:
        groups[str(group_fn(cand))].append((_score(cand), _score(ref)))

    comparisons: list[int] = []
    deltas: list[float] = []
    for values in groups.values():
        cand_mean = _mean([v[0] for v in values])
        ref_mean = _mean([v[1] for v in values])
        comparisons.append(_compare_values(cand_mean, ref_mean, tie_margin))
        deltas.append(cand_mean - ref_mean)

    out = _ratio_summary(comparisons)
    out["mean_delta"] = _mean(deltas)
    out["n_groups"] = len(groups)
    return out


def compare_records(
    candidate_records: list[dict],
    reference_records: list[dict],
    candidate_name: str,
    reference_name: str,
    tie_margin: float,
) -> dict:
    cand_by_id = _by_sample_id(candidate_records)
    ref_by_id = _by_sample_id(reference_records)
    common_ids = sorted(set(cand_by_id) & set(ref_by_id))
    paired = [(cand_by_id[sid], ref_by_id[sid]) for sid in common_ids]

    sample_comp: list[int] = []
    deltas: list[float] = []
    for cand, ref in paired:
        delta = _score(cand) - _score(ref)
        deltas.append(delta)
        sample_comp.append(_compare_values(_score(cand), _score(ref), tie_margin))

    micro = _ratio_summary(sample_comp)
    micro["mean_delta"] = _mean(deltas)
    micro["candidate_mean"] = _mean([_score(cand) for cand, _ in paired])
    micro["reference_mean"] = _mean([_score(ref) for _, ref in paired])

    return {
        "candidate": candidate_name,
        "reference": reference_name,
        "n_candidate_records": len(candidate_records),
        "n_reference_records": len(reference_records),
        "n_common": len(common_ids),
        "n_candidate_only": len(set(cand_by_id) - set(ref_by_id)),
        "n_reference_only": len(set(ref_by_id) - set(cand_by_id)),
        "tie_margin": tie_margin,
        "micro": micro,
        "user_macro": _macro_pairwise(paired, lambda r: r.get("user"), tie_margin),
        "task_macro": _macro_pairwise(paired, lambda r: r.get("target_task"), tie_margin),
        "block_macro": _macro_pairwise(paired, _block_key, tie_margin),
    }


def _format_ratio(section: dict) -> str:
    return (
        f"W/T/L={section['win_rate']:.4f}/{section['tie_rate']:.4f}/{section['lose_rate']:.4f}, "
        f"non-tie win={section['non_tie_win_rate']:.4f}, delta={section['mean_delta']:.4f}"
    )


def print_summary(results: list[dict]) -> None:
    logger.info("=" * 120)
    logger.info("Static replay pairwise comparison")
    logger.info("=" * 120)
    logger.info(
        f"{'candidate':30s} {'reference':16s} {'n':>5s} "
        f"{'win':>7s} {'tie':>7s} {'lose':>7s} {'nt_win':>7s} "
        f"{'delta':>8s} {'user_nt':>8s} {'block_nt':>8s}"
    )
    for result in results:
        m = result["micro"]
        u = result["user_macro"]
        b = result["block_macro"]
        logger.info(
            f"{result['candidate'][:30]:30s} {result['reference'][:16]:16s} "
            f"{m['n']:5d} {m['win_rate']:7.4f} {m['tie_rate']:7.4f} "
            f"{m['lose_rate']:7.4f} {m['non_tie_win_rate']:7.4f} "
            f"{m['mean_delta']:8.4f} {u['non_tie_win_rate']:8.4f} "
            f"{b['non_tie_win_rate']:8.4f}"
        )


def _parse_named_file(item: str) -> tuple[str, str]:
    if "=" in item:
        name, path = item.split("=", 1)
        return name, path
    path = item
    name = Path(path).name
    return name, path


def parse_args() -> ArgumentParser:
    parser = ArgumentParser(description="Compare static replay outputs pairwise.")
    parser.add_argument("--reference_file", required=True)
    parser.add_argument("--reference_name", default="")
    parser.add_argument(
        "--candidate_files",
        nargs="+",
        required=True,
        help="List of path or name=path entries.",
    )
    parser.add_argument("--tie_margin", type=float, default=0.0)
    parser.add_argument("--output_json", default="")
    return parser


def main() -> None:
    args = parse_args().parse_args()
    reference_records = load_jsonl(args.reference_file)
    reference_name = args.reference_name or _candidate_name(reference_records, "reference")

    results = []
    for item in args.candidate_files:
        fallback_name, path = _parse_named_file(item)
        candidate_records = load_jsonl(path)
        candidate_name = _candidate_name(candidate_records, fallback_name)
        if candidate_name == reference_name and os.path.abspath(path) == os.path.abspath(args.reference_file):
            continue
        results.append(compare_records(
            candidate_records=candidate_records,
            reference_records=reference_records,
            candidate_name=candidate_name,
            reference_name=reference_name,
            tie_margin=args.tie_margin,
        ))

    results.sort(key=lambda r: (
        r["micro"]["non_tie_win_rate"]
        if r["micro"]["non_tie_win_rate"] == r["micro"]["non_tie_win_rate"]
        else -1.0,
        r["micro"]["mean_delta"],
    ), reverse=True)
    print_summary(results)

    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as fp:
            json.dump(results, fp, ensure_ascii=False, indent=2)
        logger.info(f"Saved: {args.output_json}")


if __name__ == "__main__":
    main()
