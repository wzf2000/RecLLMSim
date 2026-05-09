from __future__ import annotations

import json
import os
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

from loguru import logger
from tqdm import tqdm

from lib.satisfaction_constants import (
    get_reason_to_id,
    is_reason_valid_for_score,
    normalize_reason_for_score,
)
from .data import get_rows_from_split
from .io import load_finished_indices
from .llm import predict_with_parse
from .prompts import build_messages, build_prompt


def collect_traces(
    model: str,
    sample_size: int,
    output_jsonl: str,
    seed: int = 42,
    max_workers: int = 8,
    data_split: str = "train",
    split_seed: int = 42,
) -> None:
    rows = get_rows_from_split(split=data_split, split_seed=split_seed)
    rng = random.Random(seed)

    total_available = len(rows)
    if sample_size <= 0 or sample_size > total_available:
        sample_size = total_available
    sampled_rows = rng.sample(rows, k=sample_size)

    os.makedirs(os.path.dirname(output_jsonl) or ".", exist_ok=True)
    finished_indices = load_finished_indices(output_jsonl)
    output_lock = Lock()
    reason_to_id = get_reason_to_id()
    valid_reasons = set(reason_to_id.keys())
    default_reason = "其它" if "其它" in reason_to_id else next(iter(reason_to_id.keys()))

    logger.info(f"Total candidate rows: {total_available}")
    logger.info(f"Sampled rows: {len(sampled_rows)}")
    logger.info(f"Already finished: {len(finished_indices)}")
    logger.info(f"Output path: {output_jsonl}")

    def process_one(sample_id: int, row: dict) -> tuple[int, dict | None]:
        if sample_id in finished_indices:
            return sample_id, None

        prompt = build_prompt(row)
        messages = build_messages(prompt, model)
        parsed_answer, reasoning_content, raw_content = predict_with_parse(messages, model)
        pred_score = int(parsed_answer.classification)
        pred_reason = parsed_answer.reason.strip()
        normalized_reason = normalize_reason_for_score(
            pred_score,
            pred_reason,
            default_reason=default_reason,
        )
        if not is_reason_valid_for_score(pred_score, pred_reason):
            logger.warning(
                f"Normalized invalid reason/score pair for sample {sample_id}: "
                f"score={pred_score}, raw_reason={pred_reason} -> {normalized_reason}"
            )
        pred_reason = normalized_reason
        if pred_reason not in valid_reasons:
            pred_reason = default_reason

        record = {
            "sample_id": sample_id,
            "model": model,
            "prompt": prompt,
            "messages": messages,
            "prediction": int(pred_score),
            "reason_prediction": pred_reason,
            "reason_prediction_id": int(reason_to_id.get(pred_reason, reason_to_id[default_reason])),
            "analysis": parsed_answer.analysis,
            "gold_score": int(row["gold_score"]),
            "gold_reason": row["gold_reason"],
            "gold_reason_id": int(reason_to_id.get(row["gold_reason"], reason_to_id[default_reason])),
            "meta": {
                "user": row["user"],
                "persona": row["persona"],
                "task_context": row["task_context"],
                "history": row["history"],
                "assistant_reply": row["assistant_reply"],
            },
            "reasoning_content": reasoning_content,
            "raw_content": raw_content,
        }
        return sample_id, record

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_one, i, row): i
            for i, row in enumerate(sampled_rows)
        }
        for future in tqdm(as_completed(futures), total=len(futures)):
            idx = futures[future]
            try:
                _, record = future.result()
                if record is None:
                    continue
                with output_lock:
                    with open(output_jsonl, "a", encoding="utf-8") as f:
                        f.write(json.dumps(record, ensure_ascii=False) + "\n")
            except Exception as e:
                logger.error(f"Sample {idx} failed: {e}")


