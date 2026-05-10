"""Reference-based post-hoc calibration for static replay scores.

The ordinary calibrate.py script calibrates only the records present in the
input file. Static replay hard subsets often contain one record per
``(user, target_task)`` block, so block-wise CDF/mean-shift falls back to
identity for many records.

This script uses a full original-response prediction file as reference context,
e.g. Qwen3-8B V2 none on all personalized target turns, so every replay record
can be calibrated relative to the original turns from the same user/task block.
"""

from __future__ import annotations

import json
import math
import os
import sys
from argparse import ArgumentParser
from collections import defaultdict

from loguru import logger

_DETECTION_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _DETECTION_DIR not in sys.path:
    sys.path.insert(0, _DETECTION_DIR)

from eval.calibrate import _hist_cdf, _inv_cdf, load_memory_cache


def load_jsonl(path: str) -> list[dict]:
    rows: list[dict] = []
    with open(path, encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def save_jsonl(records: list[dict], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fp:
        for record in records:
            fp.write(json.dumps(record, ensure_ascii=False) + "\n")


def _clip_round(value: float) -> int:
    return max(1, min(5, int(round(value))))


def _block_key(record: dict) -> tuple[str, str]:
    return str(record["user"]), str(record["target_task"])


def _group_by_block(records: list[dict]) -> dict[tuple[str, str], list[int]]:
    groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    for idx, record in enumerate(records):
        groups[_block_key(record)].append(idx)
    return groups


def _score(record: dict) -> float:
    return float(record.get("pred_score", 3))


def _stable_id(record: dict, fallback: int) -> str:
    return str(record.get("sample_id") or f"idx_{fallback}")


def _memory_model(record: dict, fallback: str) -> str:
    return (
        record.get("memory_model")
        or record.get("judge_memory_model")
        or record.get("model")
        or record.get("judge_model")
        or fallback
    )


def _reference_cdf_block(
    target_records: list[dict],
    target_idxs: list[int],
    reference_records: list[dict],
    cdf: list[tuple[int, float]],
) -> dict[int, int]:
    combined: list[tuple[float, str, str, int]] = []
    for local_idx, record in enumerate(reference_records):
        combined.append((_score(record), "ref", _stable_id(record, local_idx), local_idx))
    for idx in target_idxs:
        record = target_records[idx]
        combined.append((_score(record), "target", _stable_id(record, idx), idx))

    combined.sort(key=lambda item: (item[0], item[2], item[1]))
    n = len(combined)
    out: dict[int, int] = {}
    for rank, (_, source, _, idx) in enumerate(combined):
        if source != "target":
            continue
        q = (rank + 0.5) / n
        out[idx] = _inv_cdf(cdf, q)
    return out


def _reference_mean_shift_block(
    target_records: list[dict],
    target_idxs: list[int],
    reference_records: list[dict],
    hist_mean: float,
) -> dict[int, int]:
    ref_scores = [_score(r) for r in reference_records]
    if not ref_scores:
        return {}
    delta = hist_mean - (sum(ref_scores) / len(ref_scores))
    return {idx: _clip_round(_score(target_records[idx]) + delta) for idx in target_idxs}


def _build_transfer_map(
    reference_raw: list[dict],
    reference_calibrated: list[dict],
) -> dict[tuple[str, str], dict[int, float]]:
    cal_by_id = {
        str(record["sample_id"]): record
        for record in reference_calibrated
        if record.get("sample_id") is not None
    }
    buckets: dict[tuple[str, str], dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for raw_record in reference_raw:
        sid = str(raw_record.get("sample_id"))
        cal_record = cal_by_id.get(sid)
        if cal_record is None:
            continue
        key = _block_key(raw_record)
        raw_score = _clip_round(_score(raw_record))
        buckets[key][raw_score].append(float(cal_record.get("pred_score", raw_score)))

    mapping: dict[tuple[str, str], dict[int, float]] = {}
    for key, score_buckets in buckets.items():
        mapping[key] = {
            score: sum(values) / len(values)
            for score, values in score_buckets.items()
            if values
        }
    return mapping


def _reference_transfer_block(
    target_records: list[dict],
    target_idxs: list[int],
    score_map: dict[int, float],
) -> dict[int, int]:
    if not score_map:
        return {}
    available = sorted(score_map)
    out: dict[int, int] = {}
    for idx in target_idxs:
        raw_score = _clip_round(_score(target_records[idx]))
        if raw_score in score_map:
            mapped = score_map[raw_score]
        else:
            nearest = min(available, key=lambda s: abs(s - raw_score))
            mapped = score_map[nearest]
        out[idx] = _clip_round(mapped)
    return out


def calibrate_static_replay(
    records: list[dict],
    reference_raw: list[dict],
    method: str,
    memory_cache_dir: str,
    memory_model: str,
    reference_calibrated: list[dict] | None = None,
    min_history_turns: int = 5,
) -> tuple[list[dict], dict]:
    reference_groups = _group_by_block(reference_raw)
    target_groups = _group_by_block(records)
    transfer_map = (
        _build_transfer_map(reference_raw, reference_calibrated)
        if reference_calibrated is not None else {}
    )

    out = [dict(record) for record in records]
    stats = {
        "method": method,
        "n_records": len(records),
        "n_blocks": len(target_groups),
        "n_calibrated_records": 0,
        "n_calibrated_blocks": 0,
        "n_fallback_identity_records": 0,
        "fallback_reasons": defaultdict(int),
        "reference_records": len(reference_raw),
    }

    for key, target_idxs in target_groups.items():
        user, task = key
        ref_idxs = reference_groups.get(key, [])
        if not ref_idxs:
            stats["fallback_reasons"]["no_reference_block"] += 1
            stats["n_fallback_identity_records"] += len(target_idxs)
            continue
        ref_records = [reference_raw[idx] for idx in ref_idxs]
        model_for_cache = _memory_model(records[target_idxs[0]], memory_model)
        mem = load_memory_cache(memory_cache_dir, user, task, model_for_cache)
        if mem is None:
            stats["fallback_reasons"]["no_memory_cache"] += 1
            stats["n_fallback_identity_records"] += len(target_idxs)
            continue
        if int(mem.get("n_history_turns", 0)) < min_history_turns:
            stats["fallback_reasons"]["thin_history"] += 1
            stats["n_fallback_identity_records"] += len(target_idxs)
            continue

        predictions: dict[int, int]
        if method == "reference_cdf":
            cdf = _hist_cdf(mem.get("score_distribution") or {})
            if not cdf:
                stats["fallback_reasons"]["empty_history_cdf"] += 1
                stats["n_fallback_identity_records"] += len(target_idxs)
                continue
            predictions = _reference_cdf_block(out, target_idxs, ref_records, cdf)
        elif method == "reference_mean_shift":
            hist_mean = float(mem.get("avg_satisfaction_score", 0.0))
            if not math.isfinite(hist_mean) or hist_mean <= 0:
                stats["fallback_reasons"]["bad_history_mean"] += 1
                stats["n_fallback_identity_records"] += len(target_idxs)
                continue
            predictions = _reference_mean_shift_block(out, target_idxs, ref_records, hist_mean)
        elif method == "reference_transfer":
            predictions = _reference_transfer_block(out, target_idxs, transfer_map.get(key, {}))
            if not predictions:
                stats["fallback_reasons"]["no_transfer_map"] += 1
                stats["n_fallback_identity_records"] += len(target_idxs)
                continue
        else:
            raise ValueError(f"Unsupported method: {method}")

        changed_block = False
        for idx, new_score in predictions.items():
            out[idx]["pred_score_raw"] = records[idx].get("pred_score")
            out[idx]["pred_score"] = new_score
            out[idx]["calibration_method"] = method
            out[idx]["calibration_reference"] = "full_original_qwen3_8b_v2_none"
            changed_block = True
            stats["n_calibrated_records"] += 1
        if changed_block:
            stats["n_calibrated_blocks"] += 1

    stats["fallback_reasons"] = dict(stats["fallback_reasons"])
    return out, stats


def parse_args() -> ArgumentParser:
    parser = ArgumentParser(description="Reference-calibrate static replay scores.")
    parser.add_argument("--input_jsonl", required=True)
    parser.add_argument("--output_jsonl", default="")
    parser.add_argument(
        "--reference_raw_jsonl",
        default="outputs/personalized/Qwen_Qwen3-8B_test_none.jsonl",
    )
    parser.add_argument(
        "--reference_calibrated_jsonl",
        default="",
        help="Required for reference_transfer.",
    )
    parser.add_argument(
        "--method",
        choices=["reference_cdf", "reference_mean_shift", "reference_transfer"],
        default="reference_cdf",
    )
    parser.add_argument("--memory_cache_dir", default="outputs/personalized/memory_cache")
    parser.add_argument("--memory_model", default="Qwen/Qwen3-8B")
    parser.add_argument("--min_history_turns", type=int, default=5)
    return parser


def main() -> None:
    args = parse_args().parse_args()
    records = load_jsonl(args.input_jsonl)
    reference_raw = load_jsonl(args.reference_raw_jsonl)
    reference_calibrated = (
        load_jsonl(args.reference_calibrated_jsonl)
        if args.reference_calibrated_jsonl else None
    )
    if args.method == "reference_transfer" and reference_calibrated is None:
        raise ValueError("--reference_calibrated_jsonl is required for reference_transfer")

    out, stats = calibrate_static_replay(
        records=records,
        reference_raw=reference_raw,
        reference_calibrated=reference_calibrated,
        method=args.method,
        memory_cache_dir=args.memory_cache_dir,
        memory_model=args.memory_model,
        min_history_turns=args.min_history_turns,
    )
    logger.info(f"Reference calibration stats: {json.dumps(stats, ensure_ascii=False)}")

    output_path = args.output_jsonl
    if not output_path:
        base, ext = os.path.splitext(args.input_jsonl)
        output_path = f"{base}_{args.method}{ext or '.jsonl'}"
    save_jsonl(out, output_path)
    logger.info(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
