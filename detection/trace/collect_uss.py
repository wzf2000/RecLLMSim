"""
USS Cross-Dataset 个性化满意度感知 Agent — training-free 推理

对应 cross_dataset_feasibility.md §2 的两条降级路线：

  --mode warmup       (R1)  per-dialogue warm-up CDF
                            history = 该 dialogue 前 n_warm 个 assistant turn
                            target  = 该 dialogue 全部 turn
                            最终输出过滤掉 turn_idx < n_warm 的 warm-up 部分

  --mode population   (R2)  per-subset population rubric
                            history = 从 train split 随机采 K 条 dialogue
                            target  = 该 subset 所有 test dialogue
                            cache key = `{subset}_population` —— 同一 subset 内
                            所有 test dialogue 共享一份 memory

输出 JSONL 字段与 trace/collect_personalized.py 完全一致，可直接喂给
eval/personalized.py、eval/calibrate.py、eval/diagnose_confusion.py。

sample_id 格式：
  R1: `{dialogue_id}__{subset}__uss::{subset}::{dialogue_id}::target.json__turn_{i}`
  R2: `{subset}_population__{subset}__uss::{subset}::{dialogue_id}::target.json__turn_{i}`

运行（从 detection/ 目录）：
  python trace/collect_uss.py --mode warmup --model gpt-4o \\
      --subsets CCPE SGD --warmup_turns 5 \\
      --output_jsonl outputs/uss/gpt-4o_test_warmup.jsonl
"""

from __future__ import annotations

import json
import os
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from typing import Literal

from loguru import logger
from openai import OpenAI
from tqdm import tqdm

from lib.personalized_data import PersonalizedSample
from lib.uss_pipeline import (
    build_uss_population_samples,
    build_uss_warmup_samples,
    uss_dataset_stats,
)
from trace.collect_personalized import (
    load_finished_ids,
    run_agent_on_sample,
)
from trace import collect_personalized as _base

Mode = Literal["warmup", "population"]
MemoryUpdateMode = Literal["none", "per_session", "per_session_oracle", "per_turn"]


# ──────────────────────────────────────────────────────────────────────────────
# Block 处理：调用现有 run_agent_on_sample，再按 mode 过滤
# ──────────────────────────────────────────────────────────────────────────────

def process_uss_sample(
    sample: PersonalizedSample,
    mode: Mode,
    warmup_turns: int,
    model: str,
    memory_update_mode: MemoryUpdateMode,
    history_window_size: int,
    save_memory_snapshots: bool,
    memory_cache_dir: str | None,
    with_memory: bool,
    n_anchors: int,
    turn_eval_prompt_version: str,
) -> list[dict]:
    records = run_agent_on_sample(
        sample=sample,
        model=model,
        memory_update_mode=memory_update_mode,
        history_window_size=history_window_size,
        save_memory_snapshots=save_memory_snapshots,
        memory_cache_dir=memory_cache_dir,
        with_memory=with_memory,
        n_anchors=n_anchors,
        turn_eval_prompt_version=turn_eval_prompt_version,
    )

    if mode == "warmup":
        # 只保留真正的目标 turn（>= warmup_turns），warm-up 部分仅用于 memory + CDF
        records = [r for r in records if r.get("turn_idx", 0) >= warmup_turns]

    # 标注 dataset/mode/subset 便于后续分析
    for r in records:
        r["dataset"] = "uss"
        r["uss_mode"] = mode
        r["uss_subset"] = sample.target_task
    return records


# ──────────────────────────────────────────────────────────────────────────────
# 主推理流程
# ──────────────────────────────────────────────────────────────────────────────

def collect_all_uss(
    samples: list[PersonalizedSample],
    mode: Mode,
    warmup_turns: int,
    model: str,
    memory_update_mode: MemoryUpdateMode,
    history_window_size: int,
    output_jsonl: str,
    max_workers: int,
    save_memory_snapshots: bool,
    memory_cache_dir: str | None,
    with_memory: bool,
    n_anchors: int,
    turn_eval_prompt_version: str,
) -> None:
    os.makedirs(os.path.dirname(output_jsonl) or ".", exist_ok=True)
    finished_ids = load_finished_ids(output_jsonl)
    logger.info(f"Already finished turn IDs: {len(finished_ids)}")

    output_lock = Lock()

    def _process(sample: PersonalizedSample) -> list[dict]:
        # 断点续跑：如果该 block 在 mode 过滤后所有 expected_id 都已落盘，就跳过
        expected_ids = set()
        for tgt in sample.target_sessions:
            file_stub = os.path.basename(tgt.file_path)
            for t_idx in range(tgt.assistant_turns):
                if mode == "warmup" and t_idx < warmup_turns:
                    continue
                expected_ids.add(
                    f"{sample.user}__{sample.target_task}__{file_stub}__turn_{t_idx}"
                )
        if expected_ids and expected_ids.issubset(finished_ids):
            return []

        return process_uss_sample(
            sample=sample,
            mode=mode,
            warmup_turns=warmup_turns,
            model=model,
            memory_update_mode=memory_update_mode,
            history_window_size=history_window_size,
            save_memory_snapshots=save_memory_snapshots,
            memory_cache_dir=memory_cache_dir,
            with_memory=with_memory,
            n_anchors=n_anchors,
            turn_eval_prompt_version=turn_eval_prompt_version,
        )

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_process, s): s for s in samples}
        for future in tqdm(as_completed(futures), total=len(futures), desc=f"uss_{mode}"):
            sample = futures[future]
            try:
                records = future.result()
                if records:
                    with output_lock:
                        with open(output_jsonl, "a", encoding="utf-8") as fp:
                            for r in records:
                                fp.write(json.dumps(r, ensure_ascii=False) + "\n")
            except Exception as e:
                logger.error(f"Block {sample.block_id} failed: {e}")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> ArgumentParser:
    parser = ArgumentParser(
        description="USS cross-dataset 满意度感知 Agent 推理（training-free）"
    )
    parser.add_argument("--model", type=str, default="gpt-4o")
    parser.add_argument(
        "--mode", type=str, default="warmup",
        choices=["warmup", "population"],
        help="warmup = R1 per-dialogue warm-up；population = R2 subset 级人群 rubric",
    )
    parser.add_argument(
        "--subsets", type=str, nargs="+",
        default=["JDDC", "SGD", "MWOZ", "ReDial", "CCPE"],
        choices=["JDDC", "SGD", "MWOZ", "ReDial", "CCPE"],
    )
    parser.add_argument(
        "--test_split", type=str, default="test",
        choices=["train", "val", "test"],
        help="作为目标的 split（默认 test；R2 的 train_split 单独控制）",
    )
    parser.add_argument(
        "--train_split", type=str, default="train",
        help="R2 模式下作为人群基底的 train split",
    )
    # R1 参数
    parser.add_argument("--warmup_turns", type=int, default=5)
    parser.add_argument("--min_warmup_turns", type=int, default=3)
    parser.add_argument("--min_target_turns", type=int, default=1)
    # R2 参数
    parser.add_argument("--n_population_dialogues", type=int, default=8)
    parser.add_argument("--population_seed", type=int, default=42)
    # 通用
    parser.add_argument(
        "--memory_update_mode", type=str, default="none",
        choices=["none", "per_session", "per_session_oracle", "per_turn"],
        help="USS 默认 none —— warm-up 模式只用初始 memory 即可，per_session 在 R1 下意义不大",
    )
    parser.add_argument("--history_window_size", type=int, default=5)
    parser.add_argument("--output_jsonl", type=str, default="")
    parser.add_argument("--max_workers", type=int, default=8)
    parser.add_argument("--save_memory_snapshots", action="store_true")
    parser.add_argument(
        "--memory_cache_dir", type=str, default="outputs/uss/memory_cache",
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--limit_blocks", type=int, default=0)
    parser.add_argument("--no_memory", action="store_true")
    parser.add_argument(
        "--turn_eval_prompt_version", type=str, default="v2",
        help="复用 collect_personalized 的 prompt 版本枚举（v2 / qwen_short / boundary_34_* / ...）",
    )
    parser.add_argument("--n_anchors", type=int, default=0)
    # vLLM
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

    with_memory = not args.no_memory

    if args.mode == "warmup":
        samples = build_uss_warmup_samples(
            subsets=args.subsets,
            splits=(args.test_split,),
            warmup_turns=args.warmup_turns,
            min_warmup_turns=args.min_warmup_turns,
            min_target_turns=args.min_target_turns,
        )
    else:
        samples = build_uss_population_samples(
            subsets=args.subsets,
            train_splits=(args.train_split,),
            test_splits=(args.test_split,),
            n_population_dialogues=args.n_population_dialogues,
            seed=args.population_seed,
        )

    stats = uss_dataset_stats(samples)
    logger.info(f"USS samples: {stats}")

    if args.limit_blocks > 0:
        samples = samples[: args.limit_blocks]
        logger.info(f"Limit blocks → {len(samples)}")
    if args.limit > 0:
        samples = samples[: args.limit]

    if not args.output_jsonl:
        model_tag = args.model.replace("/", "_").replace(":", "_")
        mode_tag = "no_memory" if not with_memory else args.memory_update_mode
        subsets_tag = "+".join(args.subsets) if len(args.subsets) <= 2 else "all5"
        prompt_tag = (
            f"_{args.turn_eval_prompt_version}"
            if args.turn_eval_prompt_version != "v2" else ""
        )
        args.output_jsonl = (
            f"outputs/uss/{model_tag}_{args.mode}_{subsets_tag}"
            f"_{args.test_split}_{mode_tag}{prompt_tag}.jsonl"
        )

    logger.info(f"Mode:               {args.mode}")
    logger.info(f"Model:              {args.model}")
    logger.info(f"Subsets:            {args.subsets}")
    logger.info(f"With memory:        {with_memory}")
    if args.mode == "warmup":
        logger.info(f"Warmup turns:       {args.warmup_turns}")
    else:
        logger.info(f"Population K:       {args.n_population_dialogues} (seed={args.population_seed})")
    logger.info(f"Output:             {args.output_jsonl}")
    if with_memory:
        logger.info(f"Memory cache:       {args.memory_cache_dir}")

    if with_memory:
        os.makedirs(args.memory_cache_dir, exist_ok=True)

    collect_all_uss(
        samples=samples,
        mode=args.mode,
        warmup_turns=args.warmup_turns,
        model=args.model,
        memory_update_mode=args.memory_update_mode,
        history_window_size=args.history_window_size,
        output_jsonl=args.output_jsonl,
        max_workers=args.max_workers,
        save_memory_snapshots=args.save_memory_snapshots,
        memory_cache_dir=args.memory_cache_dir if with_memory else None,
        with_memory=with_memory,
        n_anchors=args.n_anchors,
        turn_eval_prompt_version=args.turn_eval_prompt_version,
    )

    logger.info(f"Done. Results saved to: {args.output_jsonl}")


if __name__ == "__main__":
    main()
