"""Collect URS predictions with raw episodic retrieval memory."""

from __future__ import annotations

import json
import os
import sys
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

from loguru import logger
from openai import OpenAI
from tqdm import tqdm

_DETECTION_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _DETECTION_DIR not in sys.path:
    sys.path.insert(0, _DETECTION_DIR)

from lib.personalized_data import PersonalizedSample
from lib.urs_data import URS_INTENT_LIST, build_urs_personalized_samples, urs_dataset_stats
from lib.urs_episodic import UrsRetrievalStrategy
from trace import collect_personalized as _base
from trace.urs.collect import load_finished_ids
from trace.urs.episodic_runner import run_episodic_retrieval_on_urs_sample


def _limit_by_users(
    samples: list[PersonalizedSample],
    limit_users: int,
    user_offset: int,
) -> list[PersonalizedSample]:
    if limit_users <= 0:
        return samples
    users_in_order = list(dict.fromkeys(s.user for s in samples))
    selected_users = set(users_in_order[user_offset:user_offset + limit_users])
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
        f"outputs/urs/{model_tag}_{split}_{prompt_version}_"
        f"{retrieval_strategy}_k{top_k}.jsonl"
    )


def collect_all_urs_episodic(
    *,
    samples: list[PersonalizedSample],
    model: str,
    output_jsonl: str,
    max_workers: int,
    retrieval_strategy: UrsRetrievalStrategy,
    top_k: int,
    prompt_version: str,
    max_dialogue_chars: int,
) -> None:
    os.makedirs(os.path.dirname(output_jsonl) or ".", exist_ok=True)
    finished_ids = load_finished_ids(output_jsonl)
    logger.info(f"Already finished sample IDs: {len(finished_ids)}")
    output_lock = Lock()

    def process_sample(sample: PersonalizedSample) -> list[dict]:
        expected_ids = {
            f"{sample.user}__{sample.target_task}__{os.path.basename(s.file_path)}__turn_0"
            for s in sample.target_sessions
        }
        if expected_ids and expected_ids.issubset(finished_ids):
            return []
        return run_episodic_retrieval_on_urs_sample(
            sample=sample,
            model=model,
            retrieval_strategy=retrieval_strategy,
            top_k=top_k,
            prompt_version=prompt_version,
            max_dialogue_chars=max_dialogue_chars,
        )

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_sample, s): s for s in samples}
        for future in tqdm(as_completed(futures), total=len(futures), desc="urs_episodic_blocks"):
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


def parse_args() -> ArgumentParser:
    parser = ArgumentParser(
        description="URS session-level satisfaction prediction with episodic retrieval memory"
    )
    parser.add_argument("--model", type=str, default="gpt-4o")
    parser.add_argument("--split", type=str, default="test", choices=["train", "dev", "test", "all"])
    parser.add_argument("--train_ratio", type=float, default=0.2)
    parser.add_argument("--dev_ratio", type=float, default=0.5)
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument(
        "--urs_prompt_version",
        type=str,
        default="urs_episodic_task_guarded",
        choices=[
            "urs_episodic_task_guarded",
            "urs_episodic_task_guarded_dsat_first",
            "urs_episodic_task_guarded_dsat_twostage",
        ],
    )
    parser.add_argument(
        "--retrieval_strategy",
        type=str,
        default="boundary_paired",
        choices=["topk_similar", "score_balanced", "boundary_paired", "nearest"],
    )
    parser.add_argument("--top_k", type=int, default=4)
    parser.add_argument("--max_dialogue_chars", type=int, default=900)
    parser.add_argument("--output_jsonl", type=str, default="")
    parser.add_argument("--max_workers", type=int, default=4)
    parser.add_argument(
        "--target_intents",
        type=str,
        nargs="+",
        default=None,
        choices=URS_INTENT_LIST,
    )
    parser.add_argument(
        "--languages",
        type=str,
        nargs="+",
        default=["zh", "en"],
        choices=["zh", "en"],
    )
    parser.add_argument("--min_history_sessions", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--limit_users", type=int, default=0)
    parser.add_argument("--user_offset", type=int, default=0)
    parser.add_argument("--vllm_base_url", type=str, default="")
    parser.add_argument("--vllm_api_key", type=str, default="EMPTY")
    return parser


def main() -> None:
    parser = parse_args()
    args = parser.parse_args()

    if args.vllm_base_url:
        _base.client = OpenAI(base_url=args.vllm_base_url, api_key=args.vllm_api_key)
        _base._is_vllm = True
        logger.info(f"vLLM mode: base_url={args.vllm_base_url}")

    samples = build_urs_personalized_samples(
        split=args.split,
        train_ratio=args.train_ratio,
        dev_ratio=args.dev_ratio,
        seed=args.split_seed,
        min_history_sessions=args.min_history_sessions,
        target_intents=args.target_intents,
        languages=tuple(args.languages),
    )
    samples = _limit_by_users(samples, args.limit_users, args.user_offset)
    if args.limit > 0:
        samples = samples[:args.limit]

    output_jsonl = args.output_jsonl or _default_output_path(
        model=args.model,
        split=args.split,
        retrieval_strategy=args.retrieval_strategy,
        top_k=args.top_k,
        prompt_version=args.urs_prompt_version,
    )

    logger.info(f"Model:              {args.model}")
    logger.info(f"Backend:            {'vLLM @ ' + args.vllm_base_url if args.vllm_base_url else 'OpenAI API'}")
    logger.info(f"Split:              {args.split} (train_ratio={args.train_ratio})")
    logger.info(f"Languages:          {args.languages}")
    logger.info(f"Prompt version:     {args.urs_prompt_version}")
    logger.info(f"Retrieval strategy: {args.retrieval_strategy}")
    logger.info(f"Top-k:              {args.top_k}")
    logger.info(f"Output:             {output_jsonl}")
    logger.info(f"Dataset stats:      {urs_dataset_stats(samples)}")

    collect_all_urs_episodic(
        samples=samples,
        model=args.model,
        output_jsonl=output_jsonl,
        max_workers=args.max_workers,
        retrieval_strategy=args.retrieval_strategy,
        top_k=args.top_k,
        prompt_version=args.urs_prompt_version,
        max_dialogue_chars=args.max_dialogue_chars,
    )
    logger.info(f"Done. Results saved to: {output_jsonl}")


if __name__ == "__main__":
    main()
