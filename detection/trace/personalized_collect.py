from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from typing import Callable

from loguru import logger
from tqdm import tqdm

from lib.personalized_data import PersonalizedSample
from trace.personalized_runner import (
    MemoryUpdateMode,
    MemoryUpdatePromptVersion,
    MemoryVersion,
)


def load_finished_ids(output_jsonl: str) -> set[str]:
    finished: set[str] = set()
    if not os.path.exists(output_jsonl):
        return finished
    with open(output_jsonl, "r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                finished.add(obj["sample_id"])
            except Exception:
                continue
    return finished


def collect_all(
    samples: list[PersonalizedSample],
    model: str,
    memory_update_mode: MemoryUpdateMode,
    memory_version: MemoryVersion,
    memory_update_prompt_version: MemoryUpdatePromptVersion,
    history_window_size: int,
    output_jsonl: str,
    max_workers: int,
    save_memory_snapshots: bool,
    memory_cache_dir: str | None,
    run_agent_on_sample_fn: Callable[..., list[dict]],
    with_memory: bool = True,
    n_anchors: int = 0,
    turn_eval_prompt_version: str = "v2",
    memory_model: str | None = None,
) -> None:
    """对所有样本并发执行 agent 推理，结果写入 output_jsonl。"""
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

        return run_agent_on_sample_fn(
            sample=sample,
            model=model,
            memory_update_mode=memory_update_mode,
            memory_version=memory_version,
            memory_update_prompt_version=memory_update_prompt_version,
            history_window_size=history_window_size,
            save_memory_snapshots=save_memory_snapshots,
            memory_cache_dir=memory_cache_dir,
            with_memory=with_memory,
            n_anchors=n_anchors,
            turn_eval_prompt_version=turn_eval_prompt_version,
            memory_model=memory_model,
        )

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_sample, s): s for s in samples}
        for future in tqdm(as_completed(futures), total=len(futures), desc="blocks"):
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
