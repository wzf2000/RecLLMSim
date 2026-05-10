"""
Score static replay candidate responses with a user-specific satisfaction judge.

Input is the JSONL produced by collect_static_replay.py. The judge can be an
OpenAI API model or an OpenAI-compatible vLLM endpoint.
"""

from __future__ import annotations

import json
import os
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

from loguru import logger
from openai import OpenAI
from tqdm import tqdm

import trace.collect_personalized as personalized
from lib.memory import build_turn_eval_prompt, build_turn_eval_prompt_no_memory
from lib.personalized_data import PersonalizedSample, build_personalized_samples, dataset_stats
from lib.satisfaction_constants import get_reason_to_id, normalize_reason_for_score


def load_jsonl(path: str) -> list[dict]:
    records: list[dict] = []
    with open(path, "r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def load_finished_ids(path: str) -> set[str]:
    if not os.path.exists(path):
        return set()
    finished: set[str] = set()
    with open(path, "r", encoding="utf-8") as fp:
        for line in fp:
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = f"{obj.get('candidate_model')}::{obj.get('sample_id')}"
            finished.add(key)
    return finished


def _history_window_from_prefix(prefix: list[dict], history_window_size: int) -> list[str]:
    out: list[str] = []
    for utt in prefix[-history_window_size:]:
        role = "用户" if utt["role"] == "user" else "助手"
        out.append(f"{role}：{utt['content']}")
    return out


def _sample_map(samples: list[PersonalizedSample]) -> dict[tuple[str, str], PersonalizedSample]:
    return {(s.user, s.target_task): s for s in samples}


def _select_user_subset(
    samples: list[PersonalizedSample],
    limit_users: int,
    user_offset: int,
) -> list[PersonalizedSample]:
    if limit_users <= 0:
        return samples
    users = list(dict.fromkeys(s.user for s in samples))
    selected = set(users[user_offset:user_offset + limit_users])
    return [s for s in samples if s.user in selected]


def _build_memory_cache(
    records: list[dict],
    samples_by_key: dict[tuple[str, str], PersonalizedSample],
    judge_model: str,
    memory_model: str,
    memory_cache_dir: str,
    memory_version: personalized.MemoryVersion,
    with_memory: bool,
) -> dict[tuple[str, str], object]:
    if not with_memory:
        return {}
    needed = sorted({(r["user"], r["target_task"]) for r in records})
    out: dict[tuple[str, str], object] = {}
    for key in tqdm(needed, desc="memory"):
        sample = samples_by_key.get(key)
        if sample is None:
            logger.warning(f"No PersonalizedSample for {key}; scoring will skip matching records.")
            continue
        out[key] = personalized.build_user_memory(
            sample=sample,
            model=memory_model,
            memory_cache_dir=memory_cache_dir,
            memory_version=memory_version,
        )
    return out


def score_record(
    record: dict,
    memory_by_key: dict[tuple[str, str], object],
    samples_by_key: dict[tuple[str, str], PersonalizedSample],
    judge_model: str,
    memory_model: str,
    judge_config: str,
    turn_eval_prompt_version: str,
    memory_version: personalized.MemoryVersion,
    history_window_size: int,
    with_memory: bool,
    default_reason: str,
) -> dict:
    key = (record["user"], record["target_task"])
    memory = memory_by_key.get(key)
    sample = samples_by_key.get(key)
    if sample is None:
        raise RuntimeError(f"missing sample for {key}")
    history_window = _history_window_from_prefix(
        record.get("dialogue_prefix", []),
        history_window_size=history_window_size,
    )

    if with_memory:
        if memory is None:
            raise RuntimeError(f"missing memory for {(record['user'], record['target_task'])}")
        prompt = build_turn_eval_prompt(
            memory=memory,
            profile=sample.profile,
            task_context=record.get("task_context", ""),
            history_window=history_window,
            assistant_reply=record["candidate_response"],
            prompt_version=turn_eval_prompt_version,  # type: ignore[arg-type]
        )
    else:
        prompt = build_turn_eval_prompt_no_memory(
            profile=sample.profile,
            task_context=record.get("task_context", ""),
            history_window=history_window,
            assistant_reply=record["candidate_response"],
        )

    pred = personalized._call_predict_turn(
        prompt=prompt,
        model=judge_model,
        prompt_version=turn_eval_prompt_version,
        debug_context=f"static_replay__{record['candidate_model']}__{record['sample_id']}",
    )
    pred_reason = normalize_reason_for_score(
        int(pred.classification),
        pred.reason.strip(),
        default_reason=default_reason,
    )

    out = dict(record)
    out.update({
        "judge_model": judge_model,
        "judge_memory_model": memory_model if with_memory else "none",
        "judge_config": judge_config,
        "judge_memory_version": memory_version if with_memory else "none",
        "judge_turn_eval_prompt_version": turn_eval_prompt_version,
        "pred_score": int(pred.classification),
        "reason_prediction": pred_reason,
        "analysis": pred.analysis,
    })
    return out


def score_all(
    records: list[dict],
    output_jsonl: str,
    judge_model: str,
    memory_model: str,
    judge_config: str,
    memory_version: personalized.MemoryVersion,
    turn_eval_prompt_version: str,
    history_window_size: int,
    max_workers: int,
    memory_cache_dir: str,
    with_memory: bool,
    samples: list[PersonalizedSample],
) -> None:
    os.makedirs(os.path.dirname(output_jsonl) or ".", exist_ok=True)
    finished = load_finished_ids(output_jsonl)
    reason_to_id = get_reason_to_id()
    default_reason = "其它" if "其它" in reason_to_id else next(iter(reason_to_id))

    samples_by_key = _sample_map(samples)
    pending = [
        r for r in records
        if f"{r.get('candidate_model')}::{r.get('sample_id')}" not in finished
    ]
    logger.info(f"Already scored: {len(records) - len(pending)} / {len(records)}")

    memory_by_key = _build_memory_cache(
        records=pending,
        samples_by_key=samples_by_key,
        judge_model=judge_model,
        memory_model=memory_model,
        memory_cache_dir=memory_cache_dir,
        memory_version=memory_version,
        with_memory=with_memory,
    )
    output_lock = Lock()
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                score_record,
                record,
                memory_by_key,
                samples_by_key,
                judge_model,
                memory_model,
                judge_config,
                turn_eval_prompt_version,
                memory_version,
                history_window_size,
                with_memory,
                default_reason,
            ): record
            for record in pending
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="turns"):
            record = futures[future]
            try:
                scored = future.result()
            except Exception as e:
                logger.error(f"Scoring failed: {record.get('sample_id')} ({record.get('candidate_model')}): {e}")
                continue
            with output_lock:
                with open(output_jsonl, "a", encoding="utf-8") as fp:
                    fp.write(json.dumps(scored, ensure_ascii=False) + "\n")


def parse_args() -> ArgumentParser:
    parser = ArgumentParser(description="Score static replay responses")
    parser.add_argument("--input_jsonl", type=str, required=True)
    parser.add_argument("--output_jsonl", type=str, default="")
    parser.add_argument("--judge_model", type=str, required=True)
    parser.add_argument(
        "--memory_model",
        type=str,
        default="",
        help="Model used to build/load judge memory. Defaults to --judge_model.",
    )
    parser.add_argument("--judge_base_url", type=str, default="")
    parser.add_argument("--judge_api_key", type=str, default="")
    parser.add_argument(
        "--memory_base_url",
        type=str,
        default="",
        help=(
            "OpenAI-compatible endpoint used for memory_model. If omitted, only "
            "defaults to judge_base_url when memory_model equals judge_model."
        ),
    )
    parser.add_argument("--memory_api_key", type=str, default="")
    parser.add_argument("--judge_config", type=str, default="")
    parser.add_argument("--split", type=str, default="test", choices=["train", "test", "all"])
    parser.add_argument("--train_ratio", type=float, default=0.2)
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--target_tasks", type=str, nargs="+", default=None)
    parser.add_argument("--min_history_sessions", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--limit_users", type=int, default=0)
    parser.add_argument("--user_offset", type=int, default=0)
    parser.add_argument("--max_workers", type=int, default=4)
    parser.add_argument("--history_window_size", type=int, default=5)
    parser.add_argument("--memory_cache_dir", type=str, default="outputs/personalized/memory_cache")
    parser.add_argument("--memory_version", type=str, default="v2", choices=["v2", "v3"])
    parser.add_argument("--turn_eval_prompt_version", type=str, default="v2")
    parser.add_argument("--no_memory", action="store_true")
    return parser


def main() -> None:
    args = parse_args().parse_args()
    memory_model = args.memory_model or args.judge_model

    if args.judge_base_url:
        personalized.client = OpenAI(
            base_url=args.judge_base_url,
            api_key=args.judge_api_key or "EMPTY",
        )
        personalized._is_vllm = True
    if args.memory_base_url:
        memory_base_url = args.memory_base_url
        memory_api_key = args.memory_api_key or "EMPTY"
    elif memory_model == args.judge_model:
        memory_base_url = args.judge_base_url
        memory_api_key = args.judge_api_key or "EMPTY"
    else:
        memory_base_url = ""
        memory_api_key = ""
    if memory_base_url:
        personalized.memory_client = OpenAI(
            base_url=memory_base_url,
            api_key=memory_api_key,
        )

    records = load_jsonl(args.input_jsonl)
    if args.limit > 0:
        records = records[:args.limit]

    if not args.output_jsonl:
        base, ext = os.path.splitext(args.input_jsonl)
        judge_tag = args.judge_model.replace("/", "_").replace(":", "_")
        memory_tag = ""
        if memory_model != args.judge_model and not args.no_memory:
            memory_tag = "_memmodel_" + memory_model.replace("/", "_").replace(":", "_")
        args.output_jsonl = f"{base}_scored_by_{judge_tag}{memory_tag}{ext or '.jsonl'}"
    judge_config = args.judge_config or (
        f"{args.judge_model}_mem{args.memory_version}_{args.turn_eval_prompt_version}"
        if memory_model == args.judge_model else
        f"{args.judge_model}_mem{args.memory_version}_{args.turn_eval_prompt_version}"
        f"_memmodel_{memory_model}"
    )

    samples = build_personalized_samples(
        split=args.split,
        train_ratio=args.train_ratio,
        seed=args.split_seed,
        min_history_sessions=args.min_history_sessions,
        target_tasks=args.target_tasks,
    )
    samples = _select_user_subset(samples, args.limit_users, args.user_offset)
    sample_keys = {(s.user, s.target_task) for s in samples}
    before_filter = len(records)
    records = [
        r for r in records
        if (r.get("user"), r.get("target_task")) in sample_keys
    ]
    if len(records) != before_filter:
        logger.info(f"Filtered input records by selected samples: {len(records)} / {before_filter}")

    logger.info(f"Input records: {len(records)}")
    logger.info(f"Judge model: {args.judge_model}")
    logger.info(f"Memory model: {memory_model if not args.no_memory else 'none'}")
    logger.info(f"Judge backend: {'custom @ ' + args.judge_base_url if args.judge_base_url else 'default OpenAI API'}")
    if memory_base_url:
        logger.info(f"Memory backend: custom @ {memory_base_url}")
    logger.info(f"Output: {args.output_jsonl}")
    logger.info(f"Dataset stats: {dataset_stats(samples)}")

    score_all(
        records=records,
        output_jsonl=args.output_jsonl,
        judge_model=args.judge_model,
        memory_model=memory_model,
        judge_config=judge_config,
        memory_version=args.memory_version,  # type: ignore[arg-type]
        turn_eval_prompt_version=args.turn_eval_prompt_version,
        history_window_size=args.history_window_size,
        max_workers=args.max_workers,
        memory_cache_dir=args.memory_cache_dir,
        with_memory=not args.no_memory,
        samples=samples,
    )


if __name__ == "__main__":
    main()
