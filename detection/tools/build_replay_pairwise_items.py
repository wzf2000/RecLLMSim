"""Build pairwise replay items for human validation.

The input files are scored static-replay JSONL files produced by
``trace/score_static_replay.py`` and optional post-hoc calibration scripts.
The output is a blinded pairwise annotation set for the Streamlit app in
``tools/replay_pairwise_validation_app.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any

_DETECTION_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _DETECTION_DIR not in sys.path:
    sys.path.insert(0, _DETECTION_DIR)

from lib.personalized_data import build_personalized_samples


DEFAULT_BUCKET_WEIGHTS = {
    "large_gap": 0.4,
    "small_gap": 0.4,
    "tie": 0.2,
}


def load_jsonl(path: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl(path: str, records: list[dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fp:
        for record in records:
            fp.write(json.dumps(record, ensure_ascii=False) + "\n")


def _parse_named_file(item: str) -> tuple[str | None, str]:
    if "=" in item:
        name, path = item.split("=", 1)
        return name.strip() or None, path
    return None, item


def _model_name(records: list[dict[str, Any]], fallback: str | None, path: str) -> str:
    if fallback:
        return fallback
    names = sorted({str(r.get("candidate_model", "")) for r in records if r.get("candidate_model")})
    if len(names) == 1:
        return names[0]
    return Path(path).name.replace(".jsonl", "")


def _score(record: dict[str, Any], field: str) -> float:
    value = record.get(field)
    if value is None:
        value = record.get("pred_score")
    return float(value)


def _stable_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


def _safe_text(text: object) -> str:
    if text is None:
        return ""
    return str(text).strip()


def _excerpt(text: object, max_chars: int) -> str:
    value = _safe_text(text)
    if len(value) <= max_chars:
        return value
    return value[: max_chars - 1].rstrip() + "..."


def _current_user_request(prefix: list[dict[str, Any]]) -> str:
    for utt in reversed(prefix):
        if utt.get("role") == "user":
            return _safe_text(utt.get("content"))
    return ""


def _history_turns(sample: Any, max_chars: int) -> list[dict[str, Any]]:
    turns: list[dict[str, Any]] = []
    for session in sample.history_sessions:
        source_file = os.path.basename(session.file_path)
        last_user = ""
        assistant_idx = 0
        for utt in session.history:
            role = utt.get("role")
            if role == "user":
                last_user = _safe_text(utt.get("content"))
                continue
            if role != "assistant":
                continue
            if assistant_idx >= len(session.satisfaction_scores):
                assistant_idx += 1
                continue
            score = int(session.satisfaction_scores[assistant_idx])
            reason = (
                session.dissatisfaction_reasons[assistant_idx]
                if assistant_idx < len(session.dissatisfaction_reasons)
                else ""
            )
            turns.append({
                "source_task": session.task,
                "source_file": source_file,
                "turn_idx": assistant_idx,
                "user_request": _excerpt(last_user, max_chars),
                "assistant_response": _excerpt(utt.get("content"), max_chars),
                "score": score,
                "reason": reason,
            })
            assistant_idx += 1
    return turns


def _score_distribution(turns: list[dict[str, Any]]) -> dict[str, Any]:
    scores = [int(t["score"]) for t in turns]
    counts = Counter(scores)
    total = len(scores)
    return {
        "total_turns": total,
        "mean": round(sum(scores) / total, 3) if total else None,
        "counts": {str(score): int(counts.get(score, 0)) for score in range(1, 6)},
        "sat_rate": round(sum(1 for score in scores if score >= 4) / total, 3) if total else None,
        "neutral_rate": round(sum(1 for score in scores if score == 3) / total, 3) if total else None,
        "low_rate": round(sum(1 for score in scores if score <= 2) / total, 3) if total else None,
    }


def _reason_distribution(turns: list[dict[str, Any]]) -> list[dict[str, Any]]:
    reasons = Counter(
        _safe_text(t.get("reason")) or "unspecified"
        for t in turns
        if int(t.get("score", 0)) <= 3
    )
    return [
        {"reason": reason, "count": count}
        for reason, count in reasons.most_common(5)
    ]


def _preference_summary(turns: list[dict[str, Any]], tasks: list[str]) -> list[str]:
    dist = _score_distribution(turns)
    total = int(dist["total_turns"])
    if total == 0:
        return ["No source-history satisfaction evidence is available for this user block."]

    mean = dist["mean"]
    sat_rate = float(dist["sat_rate"])
    neutral_rate = float(dist["neutral_rate"])
    low_rate = float(dist["low_rate"])
    summary: list[str] = [
        (
            f"Source-history evidence covers {total} assistant turns from "
            f"{len(tasks)} other scenario(s): {', '.join(tasks)}."
        ),
        (
            f"Historical mean score is {mean:.2f}; SAT rate is {sat_rate:.1%}, "
            f"Neutral rate is {neutral_rate:.1%}, and score-1/2 rate is {low_rate:.1%}."
        ),
    ]
    if mean is not None and mean >= 4.25 and low_rate <= 0.08:
        summary.append("This user appears relatively easy to satisfy in the source histories, so severe preference signals should be checked against concrete constraints.")
    elif mean is not None and mean <= 3.85:
        summary.append("This user appears relatively strict in the source histories; generic or under-specified answers may be less preferred.")
    else:
        summary.append("This user shows a mixed rating pattern; compare responses against the current request and the anchor examples rather than relying on overall quality alone.")

    reason_items = _reason_distribution(turns)
    if reason_items:
        reason_text = ", ".join(f"{item['reason']} ({item['count']})" for item in reason_items[:3])
        summary.append(f"Common low-side/neutral reasons in source histories: {reason_text}.")
    summary.append("This summary is generated from raw source-history labels and examples; it is not copied from the evaluator memory.")
    return summary


def _select_anchor_examples(
    turns: list[dict[str, Any]],
    max_per_side: int,
) -> dict[str, list[dict[str, Any]]]:
    high = sorted(
        [t for t in turns if int(t["score"]) >= 4],
        key=lambda t: (-int(t["score"]), str(t["source_task"]), str(t["source_file"]), int(t["turn_idx"])),
    )[:max_per_side]
    low_or_neutral = sorted(
        [t for t in turns if int(t["score"]) <= 3],
        key=lambda t: (int(t["score"]), str(t["source_task"]), str(t["source_file"]), int(t["turn_idx"])),
    )[:max_per_side]
    return {
        "high_score": high,
        "low_or_neutral": low_or_neutral,
    }


def _preference_evidence(sample: Any, max_examples_per_side: int, max_chars: int) -> dict[str, Any]:
    turns = _history_turns(sample, max_chars=max_chars)
    tasks = list(dict.fromkeys(session.task for session in sample.history_sessions))
    return {
        "source": "template_from_source_history_labels",
        "source_history_tasks": tasks,
        "score_distribution": _score_distribution(turns),
        "low_side_reasons": _reason_distribution(turns),
        "summary": _preference_summary(turns, tasks),
        "anchor_examples": _select_anchor_examples(
            turns,
            max_per_side=max_examples_per_side,
        ),
    }


def _block_context_maps(
    split: str,
    min_history_sessions: int,
    max_anchor_examples_per_side: int,
    anchor_excerpt_chars: int,
) -> tuple[
    dict[tuple[str, str], dict[str, Any]],
    dict[tuple[str, str], dict[str, Any]],
]:
    samples = build_personalized_samples(
        split=split,
        min_history_sessions=min_history_sessions,
    )
    profile_by_block = {(sample.user, sample.target_task): sample.profile for sample in samples}
    evidence_by_block = {
        (sample.user, sample.target_task): _preference_evidence(
            sample,
            max_examples_per_side=max_anchor_examples_per_side,
            max_chars=anchor_excerpt_chars,
        )
        for sample in samples
    }
    return profile_by_block, evidence_by_block


def _profile_map(split: str, min_history_sessions: int) -> dict[tuple[str, str], dict[str, Any]]:
    profile_by_block, _ = _block_context_maps(
        split=split,
        min_history_sessions=min_history_sessions,
        max_anchor_examples_per_side=2,
        anchor_excerpt_chars=420,
    )
    return profile_by_block


def _load_model_records(
    file_items: list[str],
    score_field: str,
) -> dict[str, dict[str, dict[str, Any]]]:
    by_model: dict[str, dict[str, dict[str, Any]]] = {}
    for item in file_items:
        fallback, path = _parse_named_file(item)
        records = load_jsonl(path)
        model = _model_name(records, fallback=fallback, path=path)
        if model in by_model:
            raise ValueError(f"Duplicate model name: {model}. Use name=path to disambiguate inputs.")
        sample_map: dict[str, dict[str, Any]] = {}
        for record in records:
            sample_id = str(record.get("sample_id", ""))
            if not sample_id:
                continue
            out = dict(record)
            out["_pairwise_model_name"] = model
            out["_pairwise_score"] = _score(record, score_field)
            out["_pairwise_source_type"] = "candidate"
            sample_map[sample_id] = out
        by_model[model] = sample_map
    return by_model


def _source_record_from(candidate_record: dict[str, Any], source_score_field: str) -> dict[str, Any] | None:
    response = _safe_text(candidate_record.get("source_assistant_reply"))
    if not response:
        return None
    score_value = candidate_record.get(source_score_field)
    if score_value is None:
        return None
    out = dict(candidate_record)
    out["_pairwise_model_name"] = f"original-{candidate_record.get('source_chat_model', 'assistant')}"
    out["_pairwise_score"] = float(score_value)
    out["_pairwise_source_type"] = "source_assistant"
    out["_pairwise_input_path"] = "source_assistant"
    out["candidate_response"] = response
    out["candidate_model"] = out["_pairwise_model_name"]
    out["pred_score"] = float(score_value)
    out["pred_score_raw"] = None
    out["reason_prediction"] = candidate_record.get("gold_reason", "")
    out["analysis"] = ""
    return out


def _response_payload(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "response_id": f"{record.get('_pairwise_source_type')}::{record.get('_pairwise_model_name')}",
        "source_type": record.get("_pairwise_source_type"),
        "model": record.get("_pairwise_model_name"),
        "response": _safe_text(record.get("candidate_response")),
        "evaluator_score": record.get("_pairwise_score"),
        "raw_score": record.get("pred_score_raw"),
        "reason_prediction": record.get("reason_prediction"),
        "analysis": record.get("analysis"),
    }


def _bucket(abs_delta: float, tie_margin: float) -> str:
    if abs_delta <= tie_margin:
        return "tie"
    if abs_delta >= 2.0:
        return "large_gap"
    return "small_gap"


def _preference(delta: float, tie_margin: float) -> str:
    if delta > tie_margin:
        return "a"
    if delta < -tie_margin:
        return "b"
    return "tie"


def _make_pair_item(
    record_a: dict[str, Any],
    record_b: dict[str, Any],
    profile_by_block: dict[tuple[str, str], dict[str, Any]],
    evidence_by_block: dict[tuple[str, str], dict[str, Any]],
    tie_margin: float,
    seed: int,
) -> dict[str, Any]:
    sample_id = str(record_a["sample_id"])
    model_a = str(record_a.get("_pairwise_model_name"))
    model_b = str(record_b.get("_pairwise_model_name"))
    source_type_a = str(record_a.get("_pairwise_source_type"))
    source_type_b = str(record_b.get("_pairwise_source_type"))
    pair_key = "||".join([sample_id, source_type_a, model_a, source_type_b, model_b])
    rng = random.Random(f"{seed}:{pair_key}")
    sides = [_response_payload(record_a), _response_payload(record_b)]
    rng.shuffle(sides)

    score_a = float(sides[0]["evaluator_score"])
    score_b = float(sides[1]["evaluator_score"])
    delta = score_a - score_b
    abs_delta = abs(delta)
    user = str(record_a.get("user", ""))
    target_task = str(record_a.get("target_task", ""))
    prefix = record_a.get("dialogue_prefix", [])
    pair_kind = (
        "candidate_source"
        if {source_type_a, source_type_b} == {"candidate", "source_assistant"}
        else "candidate_candidate"
    )
    item_hash = _stable_hash(pair_key)
    return {
        "item_id": f"{sample_id}__pair_{item_hash}",
        "sample_id": sample_id,
        "user": user,
        "target_task": target_task,
        "target_file": record_a.get("target_file"),
        "turn_idx": record_a.get("turn_idx"),
        "pair_kind": pair_kind,
        "pair_bucket": _bucket(abs_delta, tie_margin=tie_margin),
        "task_context": record_a.get("task_context", ""),
        "profile": profile_by_block.get((user, target_task), {}),
        "user_preference_evidence": evidence_by_block.get((user, target_task), {}),
        "dialogue_prefix": prefix,
        "current_user_request": _current_user_request(prefix if isinstance(prefix, list) else []),
        "selection_mode": record_a.get("selection_mode"),
        "selection_score": record_a.get("selection_score"),
        "selection_reasons": record_a.get("selection_reasons", []),
        "gold_score": record_a.get("gold_score"),
        "gold_reason": record_a.get("gold_reason"),
        "source_chat_model": record_a.get("source_chat_model"),
        "side_a": sides[0],
        "side_b": sides[1],
        "evaluator_preference": _preference(delta, tie_margin=tie_margin),
        "evaluator_score_delta": delta,
        "evaluator_abs_delta": abs_delta,
        "evaluator_tie_margin": tie_margin,
        "model_pair": " vs ".join(sorted([model_a, model_b])),
        "source_files": sorted(
            path
            for path in {
                _safe_text(record_a.get("_pairwise_input_path")),
                _safe_text(record_b.get("_pairwise_input_path")),
            }
            if path
        ),
    }


def build_pair_candidates(
    model_records: dict[str, dict[str, dict[str, Any]]],
    profile_by_block: dict[tuple[str, str], dict[str, Any]],
    evidence_by_block: dict[tuple[str, str], dict[str, Any]],
    include_source_assistant: bool,
    source_score_field: str,
    tie_margin: float,
    seed: int,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    all_sample_ids = sorted({sid for records in model_records.values() for sid in records})
    for sample_id in all_sample_ids:
        responses: list[dict[str, Any]] = []
        for records in model_records.values():
            record = records.get(sample_id)
            if record is not None:
                responses.append(record)
        if include_source_assistant and responses:
            source = _source_record_from(responses[0], source_score_field=source_score_field)
            if source is not None:
                responses.append(source)
        if len(responses) < 2:
            continue
        for record_a, record_b in combinations(responses, 2):
            if record_a.get("_pairwise_source_type") == record_b.get("_pairwise_source_type") == "source_assistant":
                continue
            candidates.append(
                _make_pair_item(
                    record_a=record_a,
                    record_b=record_b,
                    profile_by_block=profile_by_block,
                    evidence_by_block=evidence_by_block,
                    tie_margin=tie_margin,
                    seed=seed,
                )
            )
    return candidates


def _parse_bucket_quotas(text: str, sample_size: int, buckets_present: set[str]) -> dict[str, int]:
    if text.strip():
        quotas: dict[str, int] = {}
        for part in text.split(","):
            if not part.strip():
                continue
            key, value = part.split("=", 1)
            quotas[key.strip()] = int(value)
        return quotas

    weights = {
        key: weight
        for key, weight in DEFAULT_BUCKET_WEIGHTS.items()
        if key in buckets_present
    }
    if not weights:
        return {}
    total_weight = sum(weights.values())
    quotas = {
        key: int(sample_size * weight / total_weight)
        for key, weight in weights.items()
    }
    while sum(quotas.values()) < sample_size:
        for key in sorted(quotas):
            quotas[key] += 1
            if sum(quotas.values()) >= sample_size:
                break
    return quotas


def _can_select(
    item: dict[str, Any],
    sample_counts: Counter[str],
    user_counts: Counter[str],
    model_pair_counts: Counter[str],
    max_per_sample: int,
    max_per_user: int,
    max_per_model_pair: int,
) -> bool:
    if max_per_sample > 0 and sample_counts[str(item["sample_id"])] >= max_per_sample:
        return False
    if max_per_user > 0 and user_counts[str(item["user"])] >= max_per_user:
        return False
    if max_per_model_pair > 0 and model_pair_counts[str(item["model_pair"])] >= max_per_model_pair:
        return False
    return True


def select_items(
    candidates: list[dict[str, Any]],
    sample_size: int,
    bucket_quotas_text: str,
    max_per_sample: int,
    max_per_user: int,
    max_per_model_pair: int,
    seed: int,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    by_bucket: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in candidates:
        by_bucket[str(item["pair_bucket"])].append(item)
    for items in by_bucket.values():
        rng.shuffle(items)

    quotas = _parse_bucket_quotas(
        bucket_quotas_text,
        sample_size=sample_size,
        buckets_present=set(by_bucket),
    )
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    sample_counts: Counter[str] = Counter()
    user_counts: Counter[str] = Counter()
    model_pair_counts: Counter[str] = Counter()

    def add(item: dict[str, Any]) -> None:
        selected.append(item)
        selected_ids.add(str(item["item_id"]))
        sample_counts[str(item["sample_id"])] += 1
        user_counts[str(item["user"])] += 1
        model_pair_counts[str(item["model_pair"])] += 1

    for bucket_name, quota in quotas.items():
        for item in by_bucket.get(bucket_name, []):
            if len([x for x in selected if x["pair_bucket"] == bucket_name]) >= quota:
                break
            if str(item["item_id"]) in selected_ids:
                continue
            if _can_select(
                item,
                sample_counts=sample_counts,
                user_counts=user_counts,
                model_pair_counts=model_pair_counts,
                max_per_sample=max_per_sample,
                max_per_user=max_per_user,
                max_per_model_pair=max_per_model_pair,
            ):
                add(item)

    remaining = [item for item in candidates if str(item["item_id"]) not in selected_ids]
    rng.shuffle(remaining)
    for item in remaining:
        if len(selected) >= sample_size:
            break
        if _can_select(
            item,
            sample_counts=sample_counts,
            user_counts=user_counts,
            model_pair_counts=model_pair_counts,
            max_per_sample=max_per_sample,
            max_per_user=max_per_user,
            max_per_model_pair=max_per_model_pair,
        ):
            add(item)

    if len(selected) < sample_size:
        relaxed = [item for item in remaining if str(item["item_id"]) not in selected_ids]
        for item in relaxed:
            if len(selected) >= sample_size:
                break
            add(item)

    selected.sort(key=lambda item: str(item["item_id"]))
    return selected[:sample_size]


def summarize_items(items: list[dict[str, Any]], candidates: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "n_candidates": len(candidates),
        "n_selected": len(items),
        "selected_pair_buckets": dict(Counter(str(item["pair_bucket"]) for item in items)),
        "selected_pair_kinds": dict(Counter(str(item["pair_kind"]) for item in items)),
        "selected_users": len({item["user"] for item in items}),
        "selected_samples": len({item["sample_id"] for item in items}),
        "selected_model_pairs": dict(Counter(str(item["model_pair"]) for item in items)),
        "candidate_pair_buckets": dict(Counter(str(item["pair_bucket"]) for item in candidates)),
        "candidate_pair_kinds": dict(Counter(str(item["pair_kind"]) for item in candidates)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build pairwise human-validation replay items.")
    parser.add_argument(
        "--candidate_files",
        nargs="+",
        required=True,
        help="Scored replay JSONL files. Use name=path to override the displayed model name.",
    )
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--summary_json", default="")
    parser.add_argument("--sample_size", type=int, default=120)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split", default="test", choices=["train", "test", "all"])
    parser.add_argument("--min_history_sessions", type=int, default=1)
    parser.add_argument("--score_field", default="pred_score")
    parser.add_argument("--tie_margin", type=float, default=0.0)
    parser.add_argument("--bucket_quotas", default="")
    parser.add_argument("--max_per_sample", type=int, default=1)
    parser.add_argument("--max_per_user", type=int, default=4)
    parser.add_argument("--max_per_model_pair", type=int, default=20)
    parser.add_argument("--max_anchor_examples_per_side", type=int, default=2)
    parser.add_argument("--anchor_excerpt_chars", type=int, default=420)
    parser.add_argument("--include_source_assistant", action="store_true")
    parser.add_argument("--source_score_field", default="gold_score")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_records = _load_model_records(args.candidate_files, score_field=args.score_field)
    for item in args.candidate_files:
        fallback, path = _parse_named_file(item)
        records = load_jsonl(path)
        model = _model_name(records, fallback=fallback, path=path)
        for record in model_records[model].values():
            record["_pairwise_input_path"] = path

    profile_by_block, evidence_by_block = _block_context_maps(
        split=args.split,
        min_history_sessions=args.min_history_sessions,
        max_anchor_examples_per_side=args.max_anchor_examples_per_side,
        anchor_excerpt_chars=args.anchor_excerpt_chars,
    )
    candidates = build_pair_candidates(
        model_records=model_records,
        profile_by_block=profile_by_block,
        evidence_by_block=evidence_by_block,
        include_source_assistant=args.include_source_assistant,
        source_score_field=args.source_score_field,
        tie_margin=args.tie_margin,
        seed=args.seed,
    )
    selected = select_items(
        candidates=candidates,
        sample_size=args.sample_size,
        bucket_quotas_text=args.bucket_quotas,
        max_per_sample=args.max_per_sample,
        max_per_user=args.max_per_user,
        max_per_model_pair=args.max_per_model_pair,
        seed=args.seed,
    )
    summary = summarize_items(selected, candidates)
    write_jsonl(args.output_jsonl, selected)
    if args.summary_json:
        os.makedirs(os.path.dirname(args.summary_json) or ".", exist_ok=True)
        with open(args.summary_json, "w", encoding="utf-8") as fp:
            json.dump(summary, fp, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
