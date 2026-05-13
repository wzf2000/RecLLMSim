"""Score URS static replay responses with the URS session-level predictor."""

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
from lib.personalized_data import PersonalizedSample, SessionData
from lib.satisfaction_constants import get_reason_to_id
from lib.urs_data import (
    URS_INTENT_LIST,
    build_urs_personalized_samples,
    urs_dataset_stats,
)
from trace.urs.memory import build_user_memory_urs
from trace.urs.session_eval import evaluate_urs_session


def load_jsonl(path: str) -> list[dict]:
    records: list[dict] = []
    with open(path, "r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if line:
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
            finished.add(f"{obj.get('candidate_model')}::{obj.get('sample_id')}")
    return finished


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
    memory_model: str,
    memory_cache_dir: str,
    with_memory: bool,
    judge_client: OpenAI | None,
    memory_client: OpenAI | None,
) -> dict[tuple[str, str], object]:
    if not with_memory:
        return {}
    needed = sorted({(r["user"], r["target_task"]) for r in records})
    out: dict[tuple[str, str], object] = {}
    original_client = personalized.client
    for key in tqdm(needed, desc="urs_memory"):
        sample = samples_by_key.get(key)
        if sample is None:
            logger.warning(f"No URS sample for {key}; related records will be skipped.")
            continue
        if memory_client is not None:
            personalized.client = memory_client
        out[key] = build_user_memory_urs(
            sample=sample,
            model=memory_model,
            memory_cache_dir=memory_cache_dir,
        )
    if judge_client is not None:
        personalized.client = judge_client
    else:
        personalized.client = original_client
    return out


def _synthetic_session(record: dict) -> SessionData:
    history = [
        {"role": m.get("role", ""), "content": m.get("content", "")}
        for m in record.get("replay_history", [])
        if m.get("role") in {"user", "assistant"}
    ]
    if not history:
        history = list(record.get("dialogue_prefix", [])) + [
            {"role": "assistant", "content": record.get("candidate_response", "")}
        ]
    return SessionData(
        user=record["user"],
        task=record["target_task"],
        file_path=f"urs_replay::{record['sample_id']}.json",
        task_context=record.get("task_context", ""),
        profile={},
        history=history,
        satisfaction_scores=[int(record.get("gold_score") or 3)],
        dissatisfaction_reasons=[record.get("gold_reason") or "其它"],
        chat_model=record.get("candidate_model", "unknown"),
    )


def score_record(
    record: dict,
    samples_by_key: dict[tuple[str, str], PersonalizedSample],
    memory_by_key: dict[tuple[str, str], object],
    judge_model: str,
    memory_model: str,
    judge_config: str,
    with_memory: bool,
    default_reason: str,
    urs_prompt_version: str,
) -> dict:
    key = (record["user"], record["target_task"])
    sample = samples_by_key.get(key)
    if sample is None:
        raise RuntimeError(f"missing URS sample for {key}")
    memory = memory_by_key.get(key)
    if with_memory and memory is None:
        raise RuntimeError(f"missing URS memory for {key}")
    session = _synthetic_session(record)
    session_results = evaluate_urs_session(
        memory=memory if with_memory else None,
        session=session,
        model=judge_model,
        valid_reasons=set(get_reason_to_id().keys()),
        default_reason=default_reason,
        block_id=f"urs_static_replay__{record['candidate_model']}__{record['sample_id']}",
        prompt_version=urs_prompt_version,
    )
    pred = session_results[0]
    out = dict(record)
    out.update({
        "judge_model": judge_model,
        "judge_memory_model": memory_model if with_memory else "none",
        "judge_config": judge_config,
        "judge_memory_version": "urs_v2" if with_memory else "none",
        "pred_score": int(pred["pred_score"]),
        "reason_prediction": pred["pred_reason"],
        "analysis": pred["analysis"],
        "with_judge_memory": with_memory,
        "urs_prompt_version": urs_prompt_version,
    })
    return out


def score_all(
    records: list[dict],
    output_jsonl: str,
    judge_model: str,
    memory_model: str,
    judge_config: str,
    max_workers: int,
    memory_cache_dir: str,
    with_memory: bool,
    samples: list[PersonalizedSample],
    judge_client: OpenAI | None,
    memory_client: OpenAI | None,
    urs_prompt_version: str,
) -> None:
    os.makedirs(os.path.dirname(output_jsonl) or ".", exist_ok=True)
    finished = load_finished_ids(output_jsonl)
    pending = [
        r for r in records
        if f"{r.get('candidate_model')}::{r.get('sample_id')}" not in finished
    ]
    logger.info(f"Already scored: {len(records) - len(pending)} / {len(records)}")
    samples_by_key = _sample_map(samples)
    reason_to_id = get_reason_to_id()
    default_reason = "其它" if "其它" in reason_to_id else next(iter(reason_to_id))
    memory_by_key = _build_memory_cache(
        records=pending,
        samples_by_key=samples_by_key,
        memory_model=memory_model,
        memory_cache_dir=memory_cache_dir,
        with_memory=with_memory,
        judge_client=judge_client,
        memory_client=memory_client,
    )
    if judge_client is not None:
        personalized.client = judge_client

    output_lock = Lock()
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                score_record,
                record,
                samples_by_key,
                memory_by_key,
                judge_model,
                memory_model,
                judge_config,
                with_memory,
                default_reason,
                urs_prompt_version,
            ): record
            for record in pending
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="urs_replay_score"):
            record = futures[future]
            try:
                scored = future.result()
            except Exception as e:
                logger.error(
                    f"Scoring failed: {record.get('sample_id')} "
                    f"({record.get('candidate_model')}): {e}"
                )
                continue
            with output_lock:
                with open(output_jsonl, "a", encoding="utf-8") as fp:
                    fp.write(json.dumps(scored, ensure_ascii=False) + "\n")


def parse_args() -> ArgumentParser:
    parser = ArgumentParser(description="Score URS static replay responses")
    parser.add_argument("--input_jsonl", type=str, required=True)
    parser.add_argument("--output_jsonl", type=str, default="")
    parser.add_argument("--judge_model", type=str, required=True)
    parser.add_argument("--memory_model", type=str, default="")
    parser.add_argument("--judge_base_url", type=str, default="")
    parser.add_argument("--judge_api_key", type=str, default="")
    parser.add_argument("--memory_base_url", type=str, default="")
    parser.add_argument("--memory_api_key", type=str, default="")
    parser.add_argument("--judge_config", type=str, default="")
    parser.add_argument(
        "--urs_prompt_version",
        type=str,
        default="v2",
        choices=["v2", "urs_v2_calibrated", "urs_v2_memory_guarded"],
    )
    parser.add_argument("--split", type=str, default="test", choices=["train", "test", "all"])
    parser.add_argument("--train_ratio", type=float, default=0.2)
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--languages", type=str, nargs="+", default=["zh", "en"], choices=["zh", "en"])
    parser.add_argument("--target_intents", type=str, nargs="+", default=None, choices=URS_INTENT_LIST)
    parser.add_argument("--min_history_sessions", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--limit_users", type=int, default=0)
    parser.add_argument("--user_offset", type=int, default=0)
    parser.add_argument("--max_workers", type=int, default=4)
    parser.add_argument("--memory_cache_dir", type=str, default="outputs/urs/memory_cache")
    parser.add_argument("--no_memory", action="store_true")
    return parser


def main() -> None:
    args = parse_args().parse_args()
    memory_model = args.memory_model or args.judge_model

    judge_client = None
    memory_client = None
    if args.judge_base_url:
        judge_client = OpenAI(base_url=args.judge_base_url, api_key=args.judge_api_key or "EMPTY")
        personalized.client = judge_client
    if args.memory_base_url:
        memory_client = OpenAI(base_url=args.memory_base_url, api_key=args.memory_api_key or "EMPTY")
    elif memory_model == args.judge_model:
        memory_client = judge_client

    records = load_jsonl(args.input_jsonl)
    if args.limit > 0:
        records = records[:args.limit]

    samples = build_urs_personalized_samples(
        split=args.split,
        train_ratio=args.train_ratio,
        seed=args.split_seed,
        min_history_sessions=args.min_history_sessions,
        target_intents=args.target_intents,
        languages=tuple(args.languages),
    )
    samples = _select_user_subset(samples, args.limit_users, args.user_offset)
    sample_keys = {(s.user, s.target_task) for s in samples}
    before = len(records)
    records = [r for r in records if (r.get("user"), r.get("target_task")) in sample_keys]
    if len(records) != before:
        logger.info(f"Filtered input records by selected URS samples: {len(records)} / {before}")

    if not args.output_jsonl:
        base, ext = os.path.splitext(args.input_jsonl)
        judge_tag = args.judge_model.replace("/", "_").replace(":", "_")
        args.output_jsonl = f"{base}_scored_by_{judge_tag}{ext or '.jsonl'}"
    judge_config = args.judge_config or (
        f"urs_{args.judge_model}_mem_{memory_model}"
        if not args.no_memory else
        f"urs_{args.judge_model}_no_memory"
    )

    logger.info(f"Input records: {len(records)}")
    logger.info(f"Judge model: {args.judge_model}")
    logger.info(f"Memory model: {memory_model if not args.no_memory else 'none'}")
    logger.info(f"Judge backend: {'custom @ ' + args.judge_base_url if args.judge_base_url else 'default OpenAI API'}")
    if args.memory_base_url:
        logger.info(f"Memory backend: custom @ {args.memory_base_url}")
    logger.info(f"Output: {args.output_jsonl}")
    logger.info(f"Dataset stats: {urs_dataset_stats(samples)}")

    score_all(
        records=records,
        output_jsonl=args.output_jsonl,
        judge_model=args.judge_model,
        memory_model=memory_model,
        judge_config=judge_config,
        max_workers=args.max_workers,
        memory_cache_dir=args.memory_cache_dir,
        with_memory=not args.no_memory,
        samples=samples,
        judge_client=judge_client,
        memory_client=memory_client,
        urs_prompt_version=args.urs_prompt_version,
    )


if __name__ == "__main__":
    main()
