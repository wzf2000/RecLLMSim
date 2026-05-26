"""Runner for URS session-level episodic-retrieval satisfaction prediction."""

from __future__ import annotations

import os

from lib.personalized_data import PersonalizedSample
from lib.satisfaction_constants import get_reason_to_id
from lib.urs_episodic import UrsEpisodicMemoryIndex, UrsRetrievalStrategy
from .session_eval import evaluate_urs_session_episodic


def run_episodic_retrieval_on_urs_sample(
    sample: PersonalizedSample,
    model: str,
    retrieval_strategy: UrsRetrievalStrategy = "boundary_paired",
    top_k: int = 4,
    prompt_version: str = "urs_episodic_task_guarded",
    max_dialogue_chars: int = 900,
) -> list[dict]:
    reason_to_id = get_reason_to_id()
    valid_reasons = set(reason_to_id.keys())
    default_reason = "其它" if "其它" in reason_to_id else next(iter(reason_to_id))

    index = UrsEpisodicMemoryIndex(
        user=sample.user,
        sessions=sample.history_sessions,
        max_dialogue_chars=max_dialogue_chars,
    )

    all_records: list[dict] = []
    for session in sample.target_sessions:
        session_file = os.path.basename(session.file_path)
        retrieved = index.retrieve(
            session=session,
            k=top_k,
            strategy=retrieval_strategy,
        )
        session_results = evaluate_urs_session_episodic(
            retrieved_memories=retrieved,
            session=session,
            model=model,
            valid_reasons=valid_reasons,
            default_reason=default_reason,
            block_id=sample.block_id,
            prompt_version=prompt_version,
        )

        for r in session_results:
            all_records.append({
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
                "with_memory": True,
                "memory_update_mode": "episodic_retrieval",
                "memory_version": "urs_episodic_retrieval",
                "urs_prompt_version": prompt_version,
                "dataset": "urs",
                "chat_model": session.chat_model,
                "retrieval_strategy": retrieval_strategy,
                "retrieval_top_k": top_k,
                "retrieved_memory_ids": r["retrieved_memory_ids"],
                "retrieved_scores": r["retrieved_scores"],
                "retrieved_tasks": r["retrieved_tasks"],
                "retrieved_roles": r["retrieved_roles"],
                "retrieved_similarities": r["retrieved_similarities"],
                "dsat_failure_check": r.get("dsat_failure_check"),
                "dsat_gate_applied": r.get("dsat_gate_applied", False),
                "raw_pred_score_before_dsat_gate": r.get(
                    "raw_pred_score_before_dsat_gate",
                    r["pred_score"],
                ),
            })
    return all_records
