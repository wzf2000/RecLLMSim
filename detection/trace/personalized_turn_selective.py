"""Selective-refute turn prediction flow."""

from __future__ import annotations

from typing import Callable

from loguru import logger
from pydantic import BaseModel

from lib.anchor_retrieval import AnchorTurn
from lib.memory import (
    UserMemory,
    build_turn_eval_prompt,
    build_turn_eval_prompt_no_memory,
    build_turn_eval_refute_followup_prompt,
)
from lib.personalized_data import SessionData
from trace.personalized_predictions import (
    BoundaryTurnPrediction,
    HistoryPriorDeltaPrediction,
    HistoryPriorDeltaV2Prediction,
    SelectiveBoundaryTurnPrediction,
    _history_prior_delta_v3_dsat_votes,
    _normalize_pred_reason,
    _reconstruct_history_prior_delta_score,
    _reconstruct_history_prior_delta_v2_score,
    _reconstruct_history_prior_delta_v3_1_score,
    _reconstruct_history_prior_delta_v3_score,
)

CallPredictFn = Callable[..., BaseModel]

def should_trigger_selective_refute(
    pred: SelectiveBoundaryTurnPrediction,
    prompt_version: str,
) -> bool:
    if not pred.needs_refute_review:
        return False

    if prompt_version == "boundary_34_selective_refute":
        return True

    reason = pred.reason.strip()
    if prompt_version == "boundary_34_selective_refute_v2":
        if pred.classification == 3:
            return reason in {"不够细致", "其它"}
        if pred.classification == 4:
            return True
        return False
    if prompt_version == "v3_two_stage_v2_gate":
        if pred.classification == 3:
            return reason in {"不够细致", "其它"}
        if pred.classification == 4:
            return True
        return False
    if prompt_version == "boundary_34_selective_refute_v3":
        if pred.classification == 3:
            return reason in {"不够细致", "其它"}
        if pred.classification == 4:
            return True
        return False
    if prompt_version == "boundary_34_selective_refute_v4":
        if pred.classification == 3:
            return reason in {"不够细致", "其它"}
        if pred.classification == 4:
            return True
        return False

    return False


def predict_turn_with_optional_selective_refute(
    memory: UserMemory | None,
    session: SessionData,
    model: str,
    history_window: list[str],
    assistant_reply: str,
    turn_eval_prompt_version: str,
    debug_context: str,
    default_reason: str,
    call_predict_fn: CallPredictFn,
    anchors: list[AnchorTurn] | None = None,
) -> dict:
    """Handle one turn prediction, including optional selective refute."""
    if memory is None:
        if turn_eval_prompt_version in {
            "history_prior_delta",
            "history_prior_delta_v2",
            "history_prior_delta_v3",
            "history_prior_delta_v3_1",
            "history_prior_delta_v3_episodic",
        }:
            raise ValueError(
                f"{turn_eval_prompt_version} requires with_memory=True because it uses user history priors."
            )
        prompt = build_turn_eval_prompt_no_memory(
            profile=session.profile,
            task_context=session.task_context,
            history_window=list(history_window),
            assistant_reply=assistant_reply,
        )
        pred = call_predict_fn(
            prompt,
            model,
            prompt_version=turn_eval_prompt_version,
            debug_context=debug_context,
        )
        pred_reason = _normalize_pred_reason(
            pred.classification,
            pred.reason.strip(),
            default_reason=default_reason,
            debug_context=debug_context,
        )
        return {
            "pred_score": pred.classification,
            "pred_reason": pred_reason,
            "analysis": pred.analysis,
        }

    first_prompt = build_turn_eval_prompt(
        memory=memory,
        profile=session.profile,
        task_context=session.task_context,
        history_window=list(history_window),
        assistant_reply=assistant_reply,
        anchor_turns=anchors,
        prompt_version=turn_eval_prompt_version,
    )
    pred = call_predict_fn(
        first_prompt,
        model,
        prompt_version=turn_eval_prompt_version,
        debug_context=debug_context,
    )
    if turn_eval_prompt_version == "history_prior_delta":
        assert isinstance(pred, HistoryPriorDeltaPrediction)
        final_score = _reconstruct_history_prior_delta_score(pred)
        pred_reason = _normalize_pred_reason(
            final_score,
            pred.reason.strip(),
            default_reason=default_reason,
            debug_context=debug_context,
        )
        return {
            "pred_score": final_score,
            "pred_reason": pred_reason,
            "analysis": pred.analysis,
            "history_prior_score": pred.history_prior_score,
            "delta_label": pred.delta_label,
            "delta_score": pred.delta_score,
            "passes_satisfaction_boundary": pred.passes_satisfaction_boundary,
            "boundary_score": pred.boundary_score,
            "history_prior_delta_raw_score": pred.classification,
        }
    if turn_eval_prompt_version in {"history_prior_delta_v2", "history_prior_delta_v3", "history_prior_delta_v3_1", "history_prior_delta_v3_episodic"}:
        assert isinstance(pred, HistoryPriorDeltaV2Prediction)
        if turn_eval_prompt_version in {"history_prior_delta_v3", "history_prior_delta_v3_episodic"}:
            final_score = _reconstruct_history_prior_delta_v3_score(pred)
            dsat_votes = _history_prior_delta_v3_dsat_votes(pred)
            dsat_triggered = dsat_votes >= 2
        elif turn_eval_prompt_version == "history_prior_delta_v3_1":
            final_score = _reconstruct_history_prior_delta_v3_1_score(pred)
            dsat_votes = _history_prior_delta_v3_dsat_votes(pred)
            dsat_triggered = dsat_votes >= 3
        else:
            final_score = _reconstruct_history_prior_delta_v2_score(pred)
            dsat_votes = None
            dsat_triggered = False
        pred_reason = _normalize_pred_reason(
            final_score,
            pred.reason.strip(),
            default_reason=default_reason,
            debug_context=debug_context,
        )
        return {
            "pred_score": final_score,
            "pred_reason": pred_reason,
            "analysis": pred.analysis,
            "history_prior_score": pred.history_prior_score,
            "delta_label": pred.delta_label,
            "delta_score": pred.delta_score,
            "delta_confidence": pred.delta_confidence,
            "passes_satisfaction_boundary": pred.passes_satisfaction_boundary,
            "boundary_score": pred.boundary_score,
            "boundary_confidence": pred.boundary_confidence,
            "strong_failure_evidence": pred.strong_failure_evidence,
            "strong_excellence_evidence": pred.strong_excellence_evidence,
            "history_prior_delta_raw_score": pred.classification,
            **({"dsat_signal_votes": dsat_votes} if dsat_votes is not None else {}),
            **({"pred_boundary_score": 3 if dsat_triggered else 4} if dsat_votes is not None else {}),
        }

    pred_reason = _normalize_pred_reason(
        pred.classification,
        pred.reason.strip(),
        default_reason=default_reason,
        debug_context=debug_context,
    )

    if turn_eval_prompt_version not in {
        "boundary_34_selective_refute",
        "boundary_34_selective_refute_v2",
        "boundary_34_selective_refute_v3",
        "boundary_34_selective_refute_v4",
    }:
        return {
            "pred_score": pred.classification,
            "pred_reason": pred_reason,
            "analysis": pred.analysis,
        }

    assert isinstance(pred, SelectiveBoundaryTurnPrediction)
    should_trigger = should_trigger_selective_refute(pred, turn_eval_prompt_version)
    result = {
        "pred_score": pred.classification,
        "pred_reason": pred_reason,
        "analysis": pred.analysis,
        "analysis_first_pass": pred.analysis,
        "selective_refute_triggered": should_trigger,
        "selective_refute_applied": False,
        "selective_refute_initial_score": pred.classification,
        "selective_refute_initial_reason": pred_reason,
        "selective_refute_model_flag": pred.needs_refute_review,
    }

    if not should_trigger:
        return result

    followup_prompt = build_turn_eval_refute_followup_prompt(
        memory=memory,
        profile=session.profile,
        task_context=session.task_context,
        history_window=list(history_window),
        assistant_reply=assistant_reply,
        initial_classification=pred.classification,
        initial_reason=pred_reason,
        initial_analysis=pred.analysis,
        prompt_version=turn_eval_prompt_version,
    )

    followup_prompt_version = (
        "boundary_34_selective_refute_v2_followup"
        if turn_eval_prompt_version in {
            "boundary_34_selective_refute_v2",
            "boundary_34_selective_refute_v3",
            "boundary_34_selective_refute_v4",
        }
        else "boundary_34_selective_refute_followup"
    )
    try:
        followup = call_predict_fn(
            followup_prompt,
            model,
            prompt_version=followup_prompt_version,
            debug_context=f"{debug_context}__refute",
        )
        assert isinstance(followup, BoundaryTurnPrediction)
        followup_reason = _normalize_pred_reason(
            followup.classification,
            followup.reason.strip(),
            default_reason=default_reason,
            debug_context=f"{debug_context}__refute",
        )
        result.update(
            {
                "pred_score": followup.classification,
                "pred_reason": followup_reason,
                "analysis": (
                    f"[first_pass] {pred.analysis}\n"
                    f"[refute] {followup.analysis}"
                ),
                "analysis_refute": followup.analysis,
                "selective_refute_applied": True,
            }
        )
    except Exception as e:
        logger.warning(
            f"Selective refute follow-up failed for {debug_context}: {e}; "
            "keeping first-pass decision."
        )
        result["analysis"] = (
            f"[first_pass] {pred.analysis}\n"
            "[refute] follow-up failed, keep first-pass decision"
        )

    return result
