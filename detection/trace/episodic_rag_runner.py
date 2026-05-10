"""Runner for raw episodic-memory RAG satisfaction prediction."""

from __future__ import annotations

import os
from collections.abc import Callable

from lib.episodic_rag import (
    EpisodicMemoryIndex,
    EpisodicMemoryRecord,
    RetrievalStrategy,
    predict_score_from_retrieval,
)
from lib.episodic_rag_prompts import build_episodic_rag_turn_prompt
from lib.personalized_data import PersonalizedSample, SessionData
from lib.satisfaction_constants import SATISFIED_REASON, normalize_reason_for_score
from trace.episodic_rag_predictions import EpisodicRagTurnPrediction


def _retrieval_metadata(retrieved: list[EpisodicMemoryRecord]) -> dict:
    return {
        "retrieved_memory_ids": [r.memory_id for r in retrieved],
        "retrieved_scores": [r.score for r in retrieved],
        "retrieved_reasons": [r.reason for r in retrieved],
        "retrieved_tasks": [r.source_task for r in retrieved],
        "retrieved_roles": [r.evidence_role for r in retrieved],
        "retrieved_similarities": [round(r.similarity, 6) for r in retrieved],
    }


def _predict_one_turn(
    *,
    index: EpisodicMemoryIndex,
    profile: dict,
    session: SessionData,
    history_window: list[dict],
    last_user_msg: str,
    assistant_reply: str,
    model: str,
    call_predict_fn: Callable[..., EpisodicRagTurnPrediction],
    retrieval_strategy: RetrievalStrategy,
    top_k: int,
    prompt_version: str,
    debug_context: str,
) -> tuple[int, str, str, dict]:
    retrieved = index.retrieve(
        task_context=session.task_context,
        last_user_msg=last_user_msg,
        assistant_reply=assistant_reply,
        k=top_k,
        strategy=retrieval_strategy,
    )
    metadata = _retrieval_metadata(retrieved)
    metadata.update({
        "retrieval_strategy": retrieval_strategy,
        "retrieval_top_k": top_k,
    })

    if prompt_version == "episodic_rag_nearest":
        pred_score = predict_score_from_retrieval(retrieved)
        pred_reason = normalize_reason_for_score(pred_score, "")
        analysis = (
            "Nearest-neighbor episodic baseline using weighted retrieved scores."
        )
        metadata.update({
            "episodic_boundary_side": "sat" if pred_score >= 4 else "dsat",
            "episodic_evidence_confidence": "low" if not retrieved else "medium",
        })
        return pred_score, pred_reason, analysis, metadata

    prompt = build_episodic_rag_turn_prompt(
        profile=profile,
        task_context=session.task_context,
        history_window=history_window,
        assistant_reply=assistant_reply,
        retrieved_memories=retrieved,
        prompt_version=prompt_version,
    )
    pred = call_predict_fn(
        prompt=prompt,
        model=model,
        prompt_version=prompt_version,
        debug_context=debug_context,
    )
    pred_score = int(pred.classification)
    pred_reason = normalize_reason_for_score(pred_score, pred.reason)
    metadata.update({
        "episodic_boundary_side": getattr(
            pred,
            "boundary_side",
            getattr(pred, "boundary_decision", ""),
        ),
        "episodic_evidence_confidence": getattr(
            pred,
            "evidence_confidence",
            getattr(pred, "boundary_confidence", ""),
        ),
    })
    if hasattr(pred, "boundary_decision"):
        metadata.update({
            "episodic_boundary_decision": pred.boundary_decision,
            "episodic_boundary_confidence": pred.boundary_confidence,
            "episodic_score_refinement": pred.score_refinement,
            "episodic_key_evidence_ids": pred.key_evidence_ids,
        })
    return pred_score, pred_reason, pred.analysis, metadata


def evaluate_session_episodic_rag(
    *,
    index: EpisodicMemoryIndex,
    sample: PersonalizedSample,
    session: SessionData,
    model: str,
    call_predict_fn: Callable[..., EpisodicRagTurnPrediction],
    retrieval_strategy: RetrievalStrategy,
    top_k: int,
    history_window_size: int,
    prompt_version: str,
) -> list[dict]:
    records: list[dict] = []
    dialogue_so_far: list[dict] = []
    last_user_msg = ""
    assistant_idx = 0

    for utt in session.history:
        role = utt.get("role", "")
        content = utt.get("content", "")
        if role == "user":
            last_user_msg = content
            dialogue_so_far.append({"role": role, "content": content})
            continue
        if role != "assistant":
            dialogue_so_far.append({"role": role, "content": content})
            continue
        if assistant_idx >= len(session.satisfaction_scores):
            dialogue_so_far.append({"role": role, "content": content})
            continue

        sample_id = (
            f"{sample.user}__{sample.target_task}__"
            f"{os.path.basename(session.file_path)}__turn_{assistant_idx}"
        )
        history_window = dialogue_so_far[-history_window_size:]
        debug_context = sample_id
        pred_score, pred_reason, analysis, metadata = _predict_one_turn(
            index=index,
            profile=sample.profile,
            session=session,
            history_window=history_window,
            last_user_msg=last_user_msg,
            assistant_reply=content,
            model=model,
            call_predict_fn=call_predict_fn,
            retrieval_strategy=retrieval_strategy,
            top_k=top_k,
            prompt_version=prompt_version,
            debug_context=debug_context,
        )
        gold_score = int(session.satisfaction_scores[assistant_idx])
        gold_reason = normalize_reason_for_score(
            gold_score,
            session.dissatisfaction_reasons[assistant_idx],
        )
        records.append({
            "sample_id": sample_id,
            "user": sample.user,
            "target_task": sample.target_task,
            "target_file": os.path.basename(session.file_path),
            "turn_idx": assistant_idx,
            "model": model,
            "with_memory": True,
            "memory_update_mode": "episodic_rag",
            "memory_version": "episodic_rag",
            "memory_update_prompt_version": "none",
            "turn_eval_prompt_version": prompt_version,
            "gold_score": gold_score,
            "pred_score": pred_score,
            "gold_reason": gold_reason,
            "reason_prediction": pred_reason,
            "analysis": analysis,
            "task_context": session.task_context,
            "last_user_msg": last_user_msg,
            "assistant_reply": content,
            **metadata,
        })

        dialogue_so_far.append({"role": role, "content": content})
        assistant_idx += 1

    return records


def run_episodic_rag_on_sample(
    *,
    sample: PersonalizedSample,
    model: str,
    call_predict_fn: Callable[..., EpisodicRagTurnPrediction],
    retrieval_strategy: RetrievalStrategy = "topk_similar",
    top_k: int = 6,
    history_window_size: int = 5,
    turn_eval_prompt_version: str = "episodic_rag",
) -> list[dict]:
    index = EpisodicMemoryIndex(user=sample.user, sessions=sample.history_sessions)
    records: list[dict] = []
    for session in sample.target_sessions:
        records.extend(
            evaluate_session_episodic_rag(
                index=index,
                sample=sample,
                session=session,
                model=model,
                call_predict_fn=call_predict_fn,
                retrieval_strategy=retrieval_strategy,
                top_k=top_k,
                history_window_size=history_window_size,
                prompt_version=turn_eval_prompt_version,
            )
        )
    return records


def default_heuristic_prediction(*args, **kwargs) -> EpisodicRagTurnPrediction:
    return EpisodicRagTurnPrediction(
        classification=4,
        reason=SATISFIED_REASON,
        analysis="unused",
        boundary_side="sat",
        evidence_confidence="low",
    )
