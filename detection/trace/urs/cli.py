from __future__ import annotations

import os
from argparse import ArgumentParser

from loguru import logger
from openai import OpenAI

from lib.urs_data import URS_INTENT_LIST, build_urs_personalized_samples, urs_dataset_stats
from trace import collect_personalized as _base
from .collect import collect_all_urs


def parse_args() -> ArgumentParser:
    parser = ArgumentParser(
        description="URS session-level 满意度感知 Agent 推理（training-free）"
    )
    parser.add_argument("--model", type=str, default="gpt-4o")
    parser.add_argument(
        "--split", type=str, default="test", choices=["train", "test", "all"],
    )
    parser.add_argument("--train_ratio", type=float, default=0.2)
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument(
        "--memory_update_mode", type=str, default="per_session",
        choices=["none", "per_session", "per_session_oracle"],
        help="per_turn 在 URS 上语义不适用，已禁用",
    )
    parser.add_argument(
        "--urs_prompt_version",
        type=str,
        default="v2",
        choices=["v2", "urs_v2_calibrated", "urs_v2_memory_guarded"],
        help="URS session-level scoring prompt version.",
    )
    parser.add_argument("--output_jsonl", type=str, default="")
    parser.add_argument("--max_workers", type=int, default=8)
    parser.add_argument("--save_memory_snapshots", action="store_true")
    parser.add_argument(
        "--memory_cache_dir", type=str,
        default="outputs/urs/memory_cache",
    )
    parser.add_argument(
        "--target_intents", type=str, nargs="+", default=None,
        choices=URS_INTENT_LIST,
    )
    parser.add_argument(
        "--languages", type=str, nargs="+", default=["zh", "en"],
        choices=["zh", "en"],
        help="数据语言子集（默认 zh+en 一起；可单独跑 zh-only 或 en-only 做消融）",
    )
    parser.add_argument(
        "--min_history_sessions", type=int, default=1,
        help="过滤：历史 session 数量至少为该值（默认 1）",
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--limit_users", type=int, default=0)
    parser.add_argument("--user_offset", type=int, default=0)
    parser.add_argument("--no_memory", action="store_true")
    # vLLM 支持
    parser.add_argument("--vllm_base_url", type=str, default="")
    parser.add_argument("--vllm_api_key", type=str, default="EMPTY")
    return parser


def main() -> None:
    parser = parse_args()
    args = parser.parse_args()

    # vLLM client 切换——改的是 collect_personalized 模块里的 client（_structured_parse 通过它发请求）
    if args.vllm_base_url:
        _base.client = OpenAI(base_url=args.vllm_base_url, api_key=args.vllm_api_key)
        _base._is_vllm = True
        logger.info(f"vLLM mode: base_url={args.vllm_base_url}")

    with_memory = not args.no_memory

    if not args.output_jsonl:
        model_tag = args.model.replace("/", "_").replace(":", "_")
        mode_tag = "no_memory" if not with_memory else args.memory_update_mode
        lang_tag = "_" + "+".join(sorted(args.languages)) if len(args.languages) == 1 else ""
        args.output_jsonl = (
            f"outputs/urs/{model_tag}_{args.split}_{mode_tag}{lang_tag}.jsonl"
        )

    logger.info(f"Model:              {args.model}")
    logger.info(f"Backend:            {'vLLM @ ' + args.vllm_base_url if args.vllm_base_url else 'OpenAI API'}")
    logger.info(f"Split:              {args.split} (train_ratio={args.train_ratio})")
    logger.info(f"Languages:          {args.languages}")
    logger.info(f"With memory:        {with_memory}")
    logger.info(f"Prompt version:     {args.urs_prompt_version}")
    if with_memory:
        logger.info(f"Memory update mode: {args.memory_update_mode}")
    logger.info(f"Output:             {args.output_jsonl}")
    if with_memory:
        logger.info(f"Memory cache:       {args.memory_cache_dir}")

    samples = build_urs_personalized_samples(
        split=args.split,
        train_ratio=args.train_ratio,
        seed=args.split_seed,
        min_history_sessions=args.min_history_sessions,
        target_intents=args.target_intents,
        languages=tuple(args.languages),
    )

    stats = urs_dataset_stats(samples)
    logger.info(f"Dataset stats: {stats}")

    if args.limit_users > 0:
        users_in_order = list(dict.fromkeys(s.user for s in samples))
        start = max(args.user_offset, 0)
        end = start + args.limit_users
        selected_users = set(users_in_order[start:end])
        samples = [s for s in samples if s.user in selected_users]
        logger.info(f"Limiting to {len(selected_users)} users → {len(samples)} blocks")

    if args.limit > 0:
        samples = samples[: args.limit]
        logger.info(f"Limiting to {len(samples)} blocks for debugging.")

    if with_memory:
        os.makedirs(args.memory_cache_dir, exist_ok=True)

    collect_all_urs(
        samples=samples,
        model=args.model,
        memory_update_mode=args.memory_update_mode,
        output_jsonl=args.output_jsonl,
        max_workers=args.max_workers,
        save_memory_snapshots=args.save_memory_snapshots,
        memory_cache_dir=args.memory_cache_dir if with_memory else None,
        with_memory=with_memory,
        prompt_version=args.urs_prompt_version,
    )

    logger.info(f"Done. Results saved to: {args.output_jsonl}")
