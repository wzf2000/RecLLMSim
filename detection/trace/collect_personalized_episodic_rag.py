"""Collect personalized satisfaction predictions with raw episodic-memory RAG."""

from __future__ import annotations

import json
import os
import sys
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

from loguru import logger
from openai import OpenAI
from pydantic import BaseModel
from tenacity import before_sleep_log, retry, stop_after_attempt, wait_fixed
from tqdm import tqdm

_DETECTION_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _DETECTION_DIR not in sys.path:
    sys.path.insert(0, _DETECTION_DIR)

from lib.episodic_rag import RetrievalStrategy
from lib.llm import client as _default_client
from lib.personalized_data import (
    PersonalizedSample,
    build_personalized_samples,
    dataset_stats,
)
from trace.episodic_rag_predictions import (
    EpisodicRagBoundaryFirstPrediction,
    EpisodicRagTurnPrediction,
)
from trace.episodic_rag_runner import run_episodic_rag_on_sample
from trace.personalized_collect import load_finished_ids
from trace.structured_output import StructuredOutputError, structured_parse

client = _default_client


def _structured_parse(
    prompt: str,
    model: str,
    response_model: type[BaseModel],
    temperature: float = 0.25,
    timeout: int = 120,
    system_msg: str = "You are a skilled conversational analyst.",
) -> BaseModel:
    return structured_parse(
        client=client,
        prompt=prompt,
        model=model,
        response_model=response_model,
        temperature=temperature,
        timeout=timeout,
        system_msg=system_msg,
    )


@retry(
    stop=stop_after_attempt(3),
    wait=wait_fixed(5),
    before_sleep=before_sleep_log(logger, log_level=40),
)
def call_predict_turn(
    *,
    prompt: str,
    model: str,
    prompt_version: str = "episodic_rag",
    debug_context: str = "",
) -> EpisodicRagTurnPrediction | EpisodicRagBoundaryFirstPrediction:
    response_model: type[BaseModel]
    if prompt_version == "episodic_rag":
        response_model = EpisodicRagTurnPrediction
        temperature = 0.25
    elif prompt_version == "episodic_rag_boundary_first":
        response_model = EpisodicRagBoundaryFirstPrediction
        temperature = 0.2
    else:
        raise ValueError(f"Unsupported LLM episodic prompt version: {prompt_version}")
    try:
        return _structured_parse(
            prompt=prompt,
            model=model,
            response_model=response_model,
            temperature=temperature,
            timeout=90,
            system_msg="You are a skilled conversational analyst.",
        )
    except Exception as e:
        dump_dir = "outputs/personalized/parse_failures"
        os.makedirs(dump_dir, exist_ok=True)
        safe_context = "".join(
            c if c.isalnum() or c in {"_", "-", "."} else "_"
            for c in (debug_context or "unknown_context")
        )[:160]
        prefix = os.path.join(dump_dir, f"{safe_context}__{prompt_version}")
        meta = {
            "debug_context": debug_context,
            "model": model,
            "prompt_version": prompt_version,
            "prompt_length": len(prompt),
            "response_model": response_model.__name__,
            "error": str(e),
        }
        try:
            with open(prefix + ".json", "w", encoding="utf-8") as fp:
                json.dump(meta, fp, ensure_ascii=False, indent=2)
            with open(prefix + ".prompt.txt", "w", encoding="utf-8") as fp:
                fp.write(prompt)
            if isinstance(e, StructuredOutputError) and e.raw_text:
                with open(prefix + ".raw.txt", "w", encoding="utf-8") as fp:
                    fp.write(e.raw_text)
        except Exception as dump_err:
            logger.warning(f"Failed to dump parse debug info for {debug_context}: {dump_err}")
        raise


def _limit_by_users(
    samples: list[PersonalizedSample],
    limit_users: int,
    user_offset: int,
) -> list[PersonalizedSample]:
    if limit_users <= 0:
        return samples
    users = sorted({s.user for s in samples})
    selected_users = set(users[user_offset:user_offset + limit_users])
    return [s for s in samples if s.user in selected_users]


def _default_output_path(
    model: str,
    split: str,
    retrieval_strategy: str,
    top_k: int,
    prompt_version: str,
) -> str:
    model_tag = model.replace("/", "_").replace(":", "_")
    return (
        f"outputs/personalized/{model_tag}_{split}_"
        f"{prompt_version}_{retrieval_strategy}_k{top_k}.jsonl"
    )


def collect_all_episodic_rag(
    *,
    samples: list[PersonalizedSample],
    model: str,
    output_jsonl: str,
    max_workers: int,
    retrieval_strategy: RetrievalStrategy,
    top_k: int,
    history_window_size: int,
    turn_eval_prompt_version: str,
) -> None:
    os.makedirs(os.path.dirname(output_jsonl) or ".", exist_ok=True)
    finished_ids = load_finished_ids(output_jsonl)
    logger.info(f"Already finished turn IDs: {len(finished_ids)}")
    output_lock = Lock()

    def process_sample(sample: PersonalizedSample) -> list[dict]:
        expected_ids = {
            f"{sample.user}__{sample.target_task}__{os.path.basename(s.file_path)}__turn_{t}"
            for s in sample.target_sessions
            for t in range(s.assistant_turns)
        }
        if expected_ids and expected_ids.issubset(finished_ids):
            return []
        return run_episodic_rag_on_sample(
            sample=sample,
            model=model,
            call_predict_fn=call_predict_turn,
            retrieval_strategy=retrieval_strategy,
            top_k=top_k,
            history_window_size=history_window_size,
            turn_eval_prompt_version=turn_eval_prompt_version,
        )

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_sample, s): s for s in samples}
        for future in tqdm(as_completed(futures), total=len(futures), desc="blocks"):
            sample = futures[future]
            try:
                records = future.result()
                if not records:
                    continue
                with output_lock:
                    with open(output_jsonl, "a", encoding="utf-8") as fp:
                        for record in records:
                            fp.write(json.dumps(record, ensure_ascii=False) + "\n")
            except Exception as e:
                logger.error(f"Block {sample.block_id} failed: {e}")


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--split", choices=["train", "test", "all"], default="test")
    parser.add_argument("--train_ratio", type=float, default=0.2)
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--target_tasks", nargs="*", default=None)
    parser.add_argument("--min_history_sessions", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--limit_users", type=int, default=0)
    parser.add_argument("--user_offset", type=int, default=0)
    parser.add_argument("--output_jsonl", default="")
    parser.add_argument("--max_workers", type=int, default=4)
    parser.add_argument("--history_window_size", type=int, default=5)
    parser.add_argument(
        "--retrieval_strategy",
        choices=["topk_similar", "score_balanced", "boundary_paired", "nearest"],
        default="boundary_paired",
    )
    parser.add_argument("--top_k", type=int, default=6)
    parser.add_argument(
        "--turn_eval_prompt_version",
        choices=[
            "episodic_rag",
            "episodic_rag_boundary_first",
            "episodic_rag_nearest",
        ],
        default="episodic_rag",
    )
    parser.add_argument("--vllm_base_url", default="")
    parser.add_argument("--vllm_api_key", default="EMPTY")
    return parser.parse_args()


def main() -> None:
    global client
    args = parse_args()
    if args.vllm_base_url:
        client = OpenAI(base_url=args.vllm_base_url, api_key=args.vllm_api_key)
        logger.info(f"Using vLLM/OpenAI-compatible endpoint: {args.vllm_base_url}")

    samples = build_personalized_samples(
        split=args.split,
        train_ratio=args.train_ratio,
        seed=args.split_seed,
        min_history_sessions=args.min_history_sessions,
        target_tasks=args.target_tasks,
    )
    samples = _limit_by_users(samples, args.limit_users, args.user_offset)
    if args.limit > 0:
        samples = samples[:args.limit]

    output_jsonl = args.output_jsonl or _default_output_path(
        model=args.model,
        split=args.split,
        retrieval_strategy=args.retrieval_strategy,
        top_k=args.top_k,
        prompt_version=args.turn_eval_prompt_version,
    )
    logger.info(f"Dataset stats: {dataset_stats(samples)}")
    logger.info(
        "Episodic RAG config: "
        f"strategy={args.retrieval_strategy}, top_k={args.top_k}, "
        f"prompt={args.turn_eval_prompt_version}, output={output_jsonl}"
    )
    collect_all_episodic_rag(
        samples=samples,
        model=args.model,
        output_jsonl=output_jsonl,
        max_workers=args.max_workers,
        retrieval_strategy=args.retrieval_strategy,
        top_k=args.top_k,
        history_window_size=args.history_window_size,
        turn_eval_prompt_version=args.turn_eval_prompt_version,
    )


if __name__ == "__main__":
    main()
