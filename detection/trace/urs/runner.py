from __future__ import annotations

import os
from typing import Literal

from loguru import logger

from lib.personalized_data import PersonalizedSample
from lib.satisfaction_constants import get_reason_to_id
from .memory import build_user_memory_urs, update_memory_urs
from .session_eval import evaluate_urs_session

MemoryUpdateMode = Literal["none", "per_session", "per_session_oracle"]


def run_agent_on_urs_sample(
    sample: PersonalizedSample,
    model: str,
    memory_update_mode: MemoryUpdateMode = "per_session",
    save_memory_snapshots: bool = False,
    memory_cache_dir: str | None = None,
    with_memory: bool = True,
    prompt_version: str = "v2",
) -> list[dict]:
    reason_to_id = get_reason_to_id()
    valid_reasons = set(reason_to_id.keys())
    default_reason = "其它" if "其它" in reason_to_id else next(iter(reason_to_id))

    memory = (
        build_user_memory_urs(sample, model, memory_cache_dir=memory_cache_dir)
        if with_memory else None
    )

    all_records: list[dict] = []

    for session in sample.target_sessions:
        session_file = os.path.basename(session.file_path)
        session_results = evaluate_urs_session(
            memory=memory,
            session=session,
            model=model,
            valid_reasons=valid_reasons,
            default_reason=default_reason,
            block_id=sample.block_id,
            prompt_version=prompt_version,
        )

        memory_snapshot = memory.model_dump() if (save_memory_snapshots and memory is not None) else None
        for r in session_results:
            record = {
                "sample_id": f"{sample.user}__{sample.target_task}__{session_file}__turn_{r['turn_idx']}",
                "user": sample.user,
                "target_task": sample.target_task,
                "target_file": session_file,
                "turn_idx": r["turn_idx"],
                "gold_score": r["gold_score"],
                "pred_score": r["pred_score"],
                "gold_reason": r["gold_reason"],
                "reason_prediction": r["pred_reason"],
                "analysis": r["analysis"],
                "model": model,
                "with_memory": with_memory,
                "memory_update_mode": memory_update_mode if with_memory else "no_memory",
                "urs_prompt_version": prompt_version,
                "dataset": "urs",
                "chat_model": session.chat_model,
            }
            if memory_snapshot is not None:
                record["memory_snapshot"] = memory_snapshot
            all_records.append(record)

        if with_memory and memory_update_mode in ("per_session", "per_session_oracle"):
            use_oracle = memory_update_mode == "per_session_oracle"
            try:
                memory = update_memory_urs(
                    memory=memory,
                    session=session,
                    turn_predictions=session_results,
                    model=model,
                    use_oracle_labels=use_oracle,
                )
            except Exception as e:
                logger.warning(
                    f"Memory update failed for {sample.user}/{session_file}: {e}, "
                    f"keeping existing memory."
                )

    return all_records


# ──────────────────────────────────────────────────────────────────────────────
# 主推理流程 + 断点续跑
# ──────────────────────────────────────────────────────────────────────────────
