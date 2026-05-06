"""
Statistical/history baselines for personalized satisfaction prediction.

The script emits JSONL files compatible with eval/personalized.py.  No LLM calls
are used.  Global/task statistics are estimated from train users only; per-user
history baselines use only the source-task history_sessions of each personalized
test block.

Run from detection/:

  python eval/history_baselines.py --split test

or:

  bash scripts/run_history_baselines.sh
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from argparse import ArgumentParser
from collections import Counter, defaultdict
from dataclasses import dataclass
from statistics import median

from loguru import logger

from lib.anchor_retrieval import AnchorRetriever
from lib.personalized_data import (
    PersonalizedSample,
    SessionData,
    build_personalized_samples,
)
from lib.satisfaction_constants import normalize_reason_for_score


DEFAULT_BASELINES: list[str] = [
    "global_mean",
    "global_majority",
    "task_mean",
    "task_majority",
    "user_history_mean",
    "user_history_median",
    "user_history_majority",
    "user_history_cdf_hash",
    "nearest_history_turn",
    "nearest_history_turn_k3",
]


@dataclass
class TrainStats:
    global_scores: list[int]
    task_scores: dict[str, list[int]]


@dataclass
class TargetTurn:
    sample: PersonalizedSample
    session: SessionData
    session_file: str
    turn_idx: int
    user_msg: str
    assistant_reply: str
    gold_score: int
    gold_reason: str

    @property
    def sample_id(self) -> str:
        return (
            f"{self.sample.user}__{self.sample.target_task}__"
            f"{self.session_file}__turn_{self.turn_idx}"
        )


def _round_score(value: float) -> int:
    """Round half up and clip to the legal 1-5 score range."""
    if math.isnan(value):
        return 3
    return max(1, min(5, int(math.floor(float(value) + 0.5))))


def _majority(scores: list[int], fallback: int = 4) -> int:
    if not scores:
        return fallback
    counts = Counter(scores)
    max_count = max(counts.values())
    # Tie-break toward the lower score to avoid optimistic SAT bias.
    return min(score for score, count in counts.items() if count == max_count)


def _flatten_scores(sessions: list[SessionData]) -> list[int]:
    scores: list[int] = []
    for session in sessions:
        scores.extend(int(s) for s in session.satisfaction_scores)
    return scores


def _collect_train_stats(
    split: str,
    train_ratio: float,
    seed: int,
    min_history_sessions: int,
    target_tasks: list[str] | None,
) -> TrainStats:
    train_samples = build_personalized_samples(
        split="train" if split != "train" else "test",
        train_ratio=train_ratio,
        seed=seed,
        min_history_sessions=min_history_sessions,
        target_tasks=target_tasks,
    )
    # If we are generating baselines for train split, use held-out test users as
    # the statistics source to avoid same-record leakage.
    global_scores: list[int] = []
    task_scores: dict[str, list[int]] = defaultdict(list)
    seen_sessions: set[str] = set()

    for sample in train_samples:
        for session in sample.target_sessions:
            key = session.file_path
            if key in seen_sessions:
                continue
            seen_sessions.add(key)
            scores = [int(s) for s in session.satisfaction_scores]
            global_scores.extend(scores)
            task_scores[session.task].extend(scores)

    return TrainStats(global_scores=global_scores, task_scores=dict(task_scores))


def _iter_target_turns(sample: PersonalizedSample) -> list[TargetTurn]:
    turns: list[TargetTurn] = []
    for session in sample.target_sessions:
        session_file = os.path.basename(session.file_path)
        assistant_turn_idx = 0
        last_user_msg = ""
        for utt in session.history:
            role = utt.get("role")
            if role == "user":
                last_user_msg = utt.get("content", "")
                continue
            if role != "assistant":
                continue
            if assistant_turn_idx >= len(session.satisfaction_scores):
                break
            turns.append(
                TargetTurn(
                    sample=sample,
                    session=session,
                    session_file=session_file,
                    turn_idx=assistant_turn_idx,
                    user_msg=last_user_msg,
                    assistant_reply=utt.get("content", ""),
                    gold_score=int(session.satisfaction_scores[assistant_turn_idx]),
                    gold_reason=session.dissatisfaction_reasons[assistant_turn_idx],
                )
            )
            assistant_turn_idx += 1
    return turns


def _inverse_cdf_score(scores: list[int], q: float, fallback: int = 4) -> int:
    if not scores:
        return fallback
    ordered = sorted(int(s) for s in scores)
    idx = min(len(ordered) - 1, max(0, int(math.floor(q * len(ordered)))))
    return ordered[idx]


def _stable_rank_values(turns: list[TargetTurn]) -> dict[str, float]:
    """Return deterministic pseudo-quantiles in (0, 1) for turns in one block."""
    if not turns:
        return {}
    pairs: list[tuple[str, TargetTurn]] = []
    for turn in turns:
        digest = hashlib.sha1(turn.sample_id.encode("utf-8")).hexdigest()
        pairs.append((digest, turn))
    pairs.sort(key=lambda x: x[0])
    n = len(pairs)
    return {turn.sample_id: (i + 0.5) / n for i, (_, turn) in enumerate(pairs)}


def _predict_score(
    baseline: str,
    turn: TargetTurn,
    sample_turns: list[TargetTurn],
    train_stats: TrainStats,
    retriever: AnchorRetriever | None,
) -> tuple[int, str]:
    global_fallback = _round_score(
        sum(train_stats.global_scores) / len(train_stats.global_scores)
        if train_stats.global_scores else 4.0
    )
    history_scores = _flatten_scores(turn.sample.history_sessions)
    task_scores = train_stats.task_scores.get(turn.sample.target_task, [])

    if baseline == "global_mean":
        score = _round_score(sum(train_stats.global_scores) / len(train_stats.global_scores))
        detail = "rounded train-user global mean"
    elif baseline == "global_majority":
        score = _majority(train_stats.global_scores, fallback=global_fallback)
        detail = "train-user global majority score"
    elif baseline == "task_mean":
        source = task_scores or train_stats.global_scores
        score = _round_score(sum(source) / len(source))
        detail = f"rounded train-user task mean for {turn.sample.target_task}"
    elif baseline == "task_majority":
        score = _majority(task_scores, fallback=global_fallback)
        detail = f"train-user task majority for {turn.sample.target_task}"
    elif baseline == "user_history_mean":
        source = history_scores or train_stats.global_scores
        score = _round_score(sum(source) / len(source))
        detail = "rounded source-task user history mean"
    elif baseline == "user_history_median":
        source = history_scores or train_stats.global_scores
        score = _round_score(float(median(source)))
        detail = "source-task user history median"
    elif baseline == "user_history_majority":
        score = _majority(history_scores, fallback=global_fallback)
        detail = "source-task user history majority score"
    elif baseline == "user_history_cdf_hash":
        source = history_scores or train_stats.global_scores
        q = _stable_rank_values(sample_turns).get(turn.sample_id, 0.5)
        score = _inverse_cdf_score(source, q, fallback=global_fallback)
        detail = "source-task user history empirical CDF with stable hash ranks"
    elif baseline == "nearest_history_turn":
        anchors = (
            retriever.retrieve(turn.user_msg, turn.assistant_reply, k=1)
            if retriever is not None else []
        )
        score = int(anchors[0].score) if anchors else _majority(history_scores, fallback=global_fallback)
        detail = "top-1 TF-IDF nearest source-task history turn"
    elif baseline == "nearest_history_turn_k3":
        anchors = (
            retriever.retrieve(turn.user_msg, turn.assistant_reply, k=3)
            if retriever is not None else []
        )
        if anchors:
            score = _round_score(sum(a.score for a in anchors) / len(anchors))
        else:
            score = _majority(history_scores, fallback=global_fallback)
        detail = "rounded mean score of top-3 TF-IDF nearest source-task history turns"
    else:
        raise ValueError(f"Unknown baseline: {baseline}")

    return score, detail


def _make_record(
    baseline: str,
    turn: TargetTurn,
    pred_score: int,
    detail: str,
) -> dict:
    pred_reason = normalize_reason_for_score(pred_score, "", default_reason="其它")
    return {
        "sample_id": turn.sample_id,
        "user": turn.sample.user,
        "target_task": turn.sample.target_task,
        "target_file": turn.session_file,
        "turn_idx": turn.turn_idx,
        "gold_score": turn.gold_score,
        "pred_score": pred_score,
        "gold_reason": turn.gold_reason,
        "reason_prediction": pred_reason,
        "analysis": f"{baseline}: {detail}. No LLM call was used.",
        "model": baseline,
        "baseline_name": baseline,
        "with_memory": baseline.startswith("user_history") or baseline.startswith("nearest_history"),
        "memory_update_mode": "history_baseline",
        "memory_version": "none",
        "turn_eval_prompt_version": "history_baseline",
    }


def generate_baseline_records(
    samples: list[PersonalizedSample],
    baseline: str,
    train_stats: TrainStats,
) -> list[dict]:
    records: list[dict] = []
    for sample in samples:
        sample_turns = _iter_target_turns(sample)
        retriever = None
        if baseline.startswith("nearest_history"):
            retriever = AnchorRetriever(sample.history_sessions)
        for turn in sample_turns:
            pred_score, detail = _predict_score(
                baseline=baseline,
                turn=turn,
                sample_turns=sample_turns,
                train_stats=train_stats,
                retriever=retriever,
            )
            records.append(_make_record(baseline, turn, pred_score, detail))
    return records


def write_jsonl(records: list[dict], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fp:
        for record in records:
            fp.write(json.dumps(record, ensure_ascii=False) + "\n")


def parse_args() -> ArgumentParser:
    parser = ArgumentParser(description="Generate statistical/history baseline JSONL files.")
    parser.add_argument("--split", choices=["train", "test", "all"], default="test")
    parser.add_argument("--train_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min_history_sessions", type=int, default=1)
    parser.add_argument("--target_tasks", nargs="*", default=None)
    parser.add_argument("--limit_users", type=int, default=0)
    parser.add_argument(
        "--baselines",
        nargs="*",
        default=DEFAULT_BASELINES,
        choices=DEFAULT_BASELINES,
        help="Baselines to generate. Defaults to all supported baselines.",
    )
    parser.add_argument(
        "--output_dir",
        default="outputs/personalized/history_baselines",
        help="Directory for one JSONL file per baseline.",
    )
    parser.add_argument(
        "--output_jsonl",
        default="",
        help="Optional output path when generating exactly one baseline.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger.info(f"Building personalized samples: split={args.split}")
    samples = build_personalized_samples(
        split=args.split,
        train_ratio=args.train_ratio,
        seed=args.seed,
        min_history_sessions=args.min_history_sessions,
        target_tasks=args.target_tasks,
    )
    if args.limit_users > 0:
        allowed_users = sorted({s.user for s in samples})[: args.limit_users]
        allowed = set(allowed_users)
        samples = [s for s in samples if s.user in allowed]
        logger.info(f"limit_users={args.limit_users}, selected blocks={len(samples)}")

    logger.info(f"Collecting train statistics with seed={args.seed}")
    train_stats = _collect_train_stats(
        split=args.split,
        train_ratio=args.train_ratio,
        seed=args.seed,
        min_history_sessions=args.min_history_sessions,
        target_tasks=args.target_tasks,
    )
    logger.info(
        f"Train stats: n_scores={len(train_stats.global_scores)}, "
        f"tasks={sorted(train_stats.task_scores)}"
    )

    if args.output_jsonl and len(args.baselines) != 1:
        raise ValueError("--output_jsonl can only be used with exactly one baseline")

    for baseline in args.baselines:
        records = generate_baseline_records(samples, baseline, train_stats)
        out_path = (
            args.output_jsonl
            if args.output_jsonl
            else os.path.join(args.output_dir, f"{baseline}_{args.split}.jsonl")
        )
        write_jsonl(records, out_path)
        logger.info(f"Wrote {len(records)} records: {out_path}")


if __name__ == "__main__":
    main()
