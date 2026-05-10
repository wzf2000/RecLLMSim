"""Session-level loop for personalized turn evaluation."""

from __future__ import annotations

import os
from typing import Callable

from pydantic import BaseModel

from lib.anchor_retrieval import AnchorRetriever, AnchorTurn
from lib.memory import UserMemory
from lib.personalized_data import SessionData
from lib.satisfaction_constants import get_reason_to_id
from trace.personalized_predictions import (
    _anchor_metadata,
    _retrieve_anchor_turns,
)
from trace.personalized_turn_selective import predict_turn_with_optional_selective_refute
from trace.personalized_turn_two_stage import (
    predict_turn_fullscale_from_boundary_v2,
    predict_turn_v3_two_stage,
    predict_turn_v3_two_stage_v2,
)

CallPredictFn = Callable[..., BaseModel]

def evaluate_session(
    memory: UserMemory | None,
    session: SessionData,
    model: str,
    call_predict_fn: CallPredictFn,
    history_window_size: int = 5,
    valid_reasons: set[str] | None = None,
    default_reason: str = "其它",
    retriever: AnchorRetriever | None = None,
    n_anchors: int = 0,
    turn_eval_prompt_version: str = "v2",
    block_id: str = "",
) -> list[dict]:
    """
    Predict satisfaction turn by turn for one target session.

    Returns one result per assistant turn, including prediction, gold label,
    reason, analysis, and optional diagnostics.
    """
    if valid_reasons is None:
        reason_to_id = get_reason_to_id()
        valid_reasons = set(reason_to_id.keys())

    results: list[dict] = []
    history_window: list[str] = []
    assistant_turn_idx = 0
    last_user_msg: str = ""

    for utt in session.history:
        if utt["role"] == "user":
            last_user_msg = utt["content"]
        if utt["role"] == "assistant":
            anchors: list[AnchorTurn] | None = None
            if memory is not None and retriever is not None and n_anchors > 0:
                anchors = _retrieve_anchor_turns(
                    retriever=retriever,
                    query_user_msg=last_user_msg,
                    query_assistant_reply=utt["content"],
                    k=n_anchors,
                    turn_eval_prompt_version=turn_eval_prompt_version,
                )

            debug_context = (
                f"{block_id}__{os.path.basename(session.file_path)}__turn_{assistant_turn_idx}"
                if block_id else
                f"{os.path.basename(session.file_path)}__turn_{assistant_turn_idx}"
            )
            if turn_eval_prompt_version == "boundary_34_selective_refute_v2_fullscale":
                pred_result = predict_turn_fullscale_from_boundary_v2(
                    memory=memory,
                    session=session,
                    model=model,
                    history_window=history_window,
                    assistant_reply=utt["content"],
                    debug_context=debug_context,
                    default_reason=default_reason,
                    call_predict_fn=call_predict_fn,
                    anchors=anchors,
                )
            elif turn_eval_prompt_version == "v3_two_stage":
                pred_result = predict_turn_v3_two_stage(
                    memory=memory,
                    session=session,
                    model=model,
                    history_window=history_window,
                    assistant_reply=utt["content"],
                    debug_context=debug_context,
                    default_reason=default_reason,
                    call_predict_fn=call_predict_fn,
                    anchors=anchors,
                )
            elif turn_eval_prompt_version == "v3_two_stage_v2":
                pred_result = predict_turn_v3_two_stage_v2(
                    memory=memory,
                    session=session,
                    model=model,
                    history_window=history_window,
                    assistant_reply=utt["content"],
                    debug_context=debug_context,
                    default_reason=default_reason,
                    call_predict_fn=call_predict_fn,
                    anchors=anchors,
                )
            else:
                pred_result = predict_turn_with_optional_selective_refute(
                    memory=memory,
                    session=session,
                    model=model,
                    history_window=history_window,
                    assistant_reply=utt["content"],
                    turn_eval_prompt_version=turn_eval_prompt_version,
                    debug_context=debug_context,
                    default_reason=default_reason,
                    call_predict_fn=call_predict_fn,
                    anchors=anchors,
                )
            pred_reason = pred_result["pred_reason"].strip()
            if pred_reason not in valid_reasons:
                pred_reason = default_reason

            gold_score = session.satisfaction_scores[assistant_turn_idx]
            gold_reason = session.dissatisfaction_reasons[assistant_turn_idx]

            turn_result = {
                "turn_idx": assistant_turn_idx,
                "pred_score": pred_result["pred_score"],
                "pred_reason": pred_reason,
                "gold_score": gold_score,
                "gold_reason": gold_reason,
                "analysis": pred_result["analysis"],
            }
            if anchors is not None:
                turn_result.update(_anchor_metadata(anchors))
            for optional_key in (
                "analysis_first_pass",
                "analysis_refute",
                "selective_refute_triggered",
                "selective_refute_applied",
                "selective_refute_initial_score",
                "selective_refute_initial_reason",
                "selective_refute_model_flag",
                "analysis_router",
                "analysis_sat_refine",
                "analysis_dsat_refine",
                "fullscale_router_score",
                "fullscale_router_reason",
                "fullscale_router_analysis",
                "fullscale_router_triggered",
                "fullscale_router_applied",
                "fullscale_router_initial_score",
                "fullscale_router_initial_reason",
                "fullscale_router_model_flag",
                "fullscale_branch",
                "fullscale_refine_applied",
                "analysis_gate",
                "two_stage_gate_score",
                "two_stage_gate_reason",
                "two_stage_gate_analysis",
                "two_stage_branch",
                "two_stage_refine_applied",
                "two_stage_gate_model_flag",
                "two_stage_gate_triggered",
                "two_stage_gate_refute_applied",
                "analysis_gate_first_pass",
                "analysis_gate_followup",
                "history_prior_score",
                "delta_label",
                "delta_score",
                "passes_satisfaction_boundary",
                "boundary_score",
                "delta_confidence",
                "boundary_confidence",
                "strong_failure_evidence",
                "strong_excellence_evidence",
                "dsat_signal_votes",
                "pred_boundary_score",
                "history_prior_delta_raw_score",
                "n_anchors_retrieved",
                "anchor_scores",
                "anchor_tasks",
                "anchor_evidence_roles",
                "analysis_episodic_refine",
                "episodic_refine_triggered",
                "episodic_refine_applied",
                "episodic_refine_initial_score",
                "episodic_refine_initial_reason",
                "episodic_refine_first_pass_dsat_votes",
                "episodic_closest_evidence_side",
                "episodic_evidence_match_confidence",
                "episodic_refine_boundary_score",
                "episodic_refine_reason",
            ):
                if optional_key in pred_result:
                    turn_result[optional_key] = pred_result[optional_key]
            results.append(turn_result)
            assistant_turn_idx += 1

        role_label = "用户" if utt["role"] == "user" else "助手"
        history_window.append(f"{role_label}：{utt['content']}")
        while len(history_window) > history_window_size:
            history_window.pop(0)

    return results
