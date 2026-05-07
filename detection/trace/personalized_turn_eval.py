"""Turn-level evaluation flow for personalized trace collection."""

from __future__ import annotations

import json
import os
from typing import Callable

from loguru import logger
from pydantic import BaseModel
from tenacity import before_sleep_log, retry, stop_after_attempt, wait_fixed

from lib.anchor_retrieval import AnchorRetriever, AnchorTurn
from lib.memory import (
    UserMemory,
    UserMemoryV3,
    build_turn_eval_fullscale_dsat_refinement_prompt,
    build_turn_eval_fullscale_sat_refinement_prompt,
    build_turn_eval_refute_followup_prompt,
    build_turn_eval_prompt,
    build_turn_eval_prompt_no_memory,
    build_turn_eval_v3_two_stage_dsat_refinement_prompt,
    build_turn_eval_v3_two_stage_gate_prompt,
    build_turn_eval_v3_two_stage_sat_refinement_prompt,
    build_turn_eval_v3_two_stage_v2_gate_followup_prompt,
    build_turn_eval_v3_two_stage_v2_gate_prompt,
)
from lib.personalized_data import SessionData
from lib.satisfaction_constants import get_reason_to_id
from trace.personalized_predictions import (
    BoundaryTurnPrediction,
    DsatRefinementPrediction,
    HistoryPriorDeltaPrediction,
    HistoryPriorDeltaV2Prediction,
    SatRefinementPrediction,
    SelectiveBoundaryTurnPrediction,
    TurnPrediction,
    _anchor_metadata,
    _history_prior_delta_v3_dsat_votes,
    _normalize_pred_reason,
    _reconstruct_history_prior_delta_score,
    _reconstruct_history_prior_delta_v2_score,
    _reconstruct_history_prior_delta_v3_1_score,
    _reconstruct_history_prior_delta_v3_score,
    _retrieve_anchor_turns,
)
from trace.structured_output import StructuredOutputError

StructuredParseFn = Callable[..., BaseModel]
CallPredictFn = Callable[..., BaseModel]


@retry(
    stop=stop_after_attempt(3),
    wait=wait_fixed(5),
    before_sleep=before_sleep_log(logger, log_level=40),
)
def call_predict_turn(
    prompt: str,
    model: str,
    parse_fn: StructuredParseFn,
    raw_parse_fn: StructuredParseFn,
    prompt_version: str = "v2",
    debug_context: str = "",
) -> (
    TurnPrediction
    | BoundaryTurnPrediction
    | SelectiveBoundaryTurnPrediction
    | HistoryPriorDeltaPrediction
    | HistoryPriorDeltaV2Prediction
    | SatRefinementPrediction
    | DsatRefinementPrediction
):
    is_selective_prompt = prompt_version in {
        "boundary_34_selective_refute",
        "boundary_34_selective_refute_v2",
        "boundary_34_selective_refute_v3",
        "boundary_34_selective_refute_v4",
        "v3_two_stage_v2_gate",
    }
    if prompt_version == "boundary_34_selective_refute_v2_fullscale_sat_refine":
        response_model = SatRefinementPrediction
        is_boundary_prompt = True
    elif prompt_version == "boundary_34_selective_refute_v2_fullscale_dsat_refine":
        response_model = DsatRefinementPrediction
        is_boundary_prompt = True
    elif prompt_version == "v3_two_stage_sat_refine":
        response_model = SatRefinementPrediction
        is_boundary_prompt = True
    elif prompt_version == "v3_two_stage_dsat_refine":
        response_model = DsatRefinementPrediction
        is_boundary_prompt = True
    elif prompt_version == "history_prior_delta":
        response_model = HistoryPriorDeltaPrediction
        is_boundary_prompt = True
    elif prompt_version == "history_prior_delta_v2":
        response_model = HistoryPriorDeltaV2Prediction
        is_boundary_prompt = False
    elif prompt_version == "history_prior_delta_v3":
        response_model = HistoryPriorDeltaV2Prediction
        is_boundary_prompt = False
    elif prompt_version == "history_prior_delta_v3_1":
        response_model = HistoryPriorDeltaV2Prediction
        is_boundary_prompt = False
    elif prompt_version == "history_prior_delta_v3_episodic":
        response_model = HistoryPriorDeltaV2Prediction
        is_boundary_prompt = False
    else:
        is_boundary_prompt = prompt_version in {
            "v3_two_stage_gate",
            "v3_two_stage_v2_gate",
            "boundary_34",
            "boundary_34_refute",
            "boundary_34_refute_v2",
            "boundary_34_selective_refute",
            "boundary_34_selective_refute_v2",
            "boundary_34_selective_refute_v3",
            "boundary_34_selective_refute_v4",
            "boundary_34_selective_refute_followup",
            "boundary_34_selective_refute_v2_followup",
            "v3_two_stage_v2_gate_followup",
        }
        if is_selective_prompt:
            response_model = SelectiveBoundaryTurnPrediction
        elif is_boundary_prompt:
            response_model = BoundaryTurnPrediction
        else:
            response_model = TurnPrediction
    temperature = (
        0.2 if prompt_version == "boundary_34_refute" else
        0.2 if prompt_version == "boundary_34_selective_refute_followup" else
        0.2 if prompt_version == "boundary_34_selective_refute_v2_followup" else
        0.2 if prompt_version == "v3_two_stage_gate" else
        0.2 if prompt_version == "v3_two_stage_v2_gate_followup" else
        0.25 if prompt_version == "boundary_34_selective_refute_v2_fullscale_sat_refine" else
        0.25 if prompt_version == "boundary_34_selective_refute_v2_fullscale_dsat_refine" else
        0.25 if prompt_version == "v3_two_stage_sat_refine" else
        0.25 if prompt_version == "v3_two_stage_dsat_refine" else
        0.25 if prompt_version == "boundary_34_refute_v2" else
        0.25 if prompt_version == "boundary_34_selective_refute" else
        0.25 if prompt_version == "boundary_34_selective_refute_v2" else
        0.25 if prompt_version == "boundary_34_selective_refute_v3" else
        0.25 if prompt_version == "boundary_34_selective_refute_v4" else
        0.25 if prompt_version == "v3_two_stage_v2_gate" else
        0.25 if prompt_version == "history_prior_delta" else
        0.25 if prompt_version == "history_prior_delta_v2" else
        0.25 if prompt_version == "history_prior_delta_v3" else
        0.25 if prompt_version == "history_prior_delta_v3_1" else
        0.25 if prompt_version == "history_prior_delta_v3_episodic" else
        0.3 if prompt_version == "boundary_34" else
        0.6
    )

    try:
        if is_boundary_prompt:
            return raw_parse_fn(
                prompt,
                model,
                response_model,
                temperature=temperature,
                timeout=60,
                system_msg="You are a skilled conversational analyst.",
            )
        return parse_fn(
            prompt,
            model,
            response_model,
            temperature=temperature,
            timeout=60,
            system_msg="You are a skilled conversational analyst.",
        )
    except Exception as e:
        dump_dir = "outputs/personalized/parse_failures"
        os.makedirs(dump_dir, exist_ok=True)
        safe_context = "".join(
            c if c.isalnum() or c in {"_", "-", "."} else "_"
            for c in (debug_context or "unknown_context")
        )[:160]
        prefix = os.path.join(dump_dir, f"{safe_context}__{prompt_version}")
        meta = {
            "debug_context": debug_context,
            "model": model,
            "prompt_version": prompt_version,
            "prompt_length": len(prompt),
            "temperature": temperature,
            "response_model": response_model.__name__,
            "parse_route": "raw_text" if is_boundary_prompt else "sdk_parse",
            "error": str(e),
        }
        try:
            with open(prefix + ".json", "w", encoding="utf-8") as fp:
                json.dump(meta, fp, ensure_ascii=False, indent=2)
            with open(prefix + ".prompt.txt", "w", encoding="utf-8") as fp:
                fp.write(prompt)
            if isinstance(e, StructuredOutputError) and e.raw_text:
                with open(prefix + ".raw.txt", "w", encoding="utf-8") as fp:
                    fp.write(e.raw_text)
        except Exception as dump_err:
            logger.warning(f"Failed to dump parse debug info for {debug_context}: {dump_err}")

        logger.error(
            "Turn prediction parse failed: "
            f"context={debug_context}, prompt_version={prompt_version}, "
            f"prompt_len={len(prompt)}, temperature={temperature}, error={e}"
        )
        raise


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


def predict_turn_fullscale_from_boundary_v2(
    memory: UserMemory | None,
    session: SessionData,
    model: str,
    history_window: list[str],
    assistant_reply: str,
    debug_context: str,
    default_reason: str,
    call_predict_fn: CallPredictFn,
    anchors: list[AnchorTurn] | None = None,
) -> dict:
    """
    Hierarchical 1-5 pipeline:
    1. Use boundary_34_selective_refute_v2 as the 3/4 router.
    2. Refine SAT to 4/5 and DSAT to 1/2/3.
    """
    if memory is None:
        prompt = build_turn_eval_prompt_no_memory(
            profile=session.profile,
            task_context=session.task_context,
            history_window=list(history_window),
            assistant_reply=assistant_reply,
        )
        pred = call_predict_fn(
            prompt,
            model,
            prompt_version="v2",
            debug_context=debug_context,
        )
        assert isinstance(pred, TurnPrediction)
        return {
            "pred_score": pred.classification,
            "pred_reason": _normalize_pred_reason(
                pred.classification,
                pred.reason.strip(),
                default_reason=default_reason,
                debug_context=debug_context,
            ),
            "analysis": pred.analysis,
            "fullscale_router_score": None,
            "fullscale_router_reason": "",
            "fullscale_router_analysis": "",
            "fullscale_branch": "no_memory_fallback",
            "fullscale_refine_applied": False,
        }

    router_result = predict_turn_with_optional_selective_refute(
        memory=memory,
        session=session,
        model=model,
        history_window=history_window,
        assistant_reply=assistant_reply,
        turn_eval_prompt_version="boundary_34_selective_refute_v2",
        debug_context=f"{debug_context}__router",
        default_reason=default_reason,
        call_predict_fn=call_predict_fn,
        anchors=anchors,
    )

    router_score = int(router_result["pred_score"])
    router_reason = router_result["pred_reason"].strip()
    router_analysis = router_result["analysis"]
    base_result = {
        "fullscale_router_score": router_score,
        "fullscale_router_reason": router_reason,
        "fullscale_router_analysis": router_analysis,
        "fullscale_router_triggered": router_result.get("selective_refute_triggered", False),
        "fullscale_router_applied": router_result.get("selective_refute_applied", False),
        "fullscale_router_initial_score": router_result.get("selective_refute_initial_score"),
        "fullscale_router_initial_reason": router_result.get("selective_refute_initial_reason"),
        "fullscale_router_model_flag": router_result.get("selective_refute_model_flag"),
        "analysis_router": router_analysis,
    }

    if router_score >= 4:
        refine_prompt = build_turn_eval_fullscale_sat_refinement_prompt(
            memory=memory,
            profile=session.profile,
            task_context=session.task_context,
            history_window=list(history_window),
            assistant_reply=assistant_reply,
            router_reason=router_reason,
            router_analysis=router_analysis,
            anchor_turns=anchors,
        )
        refine = call_predict_fn(
            refine_prompt,
            model,
            prompt_version="boundary_34_selective_refute_v2_fullscale_sat_refine",
            debug_context=f"{debug_context}__sat_refine",
        )
        assert isinstance(refine, SatRefinementPrediction)
        final_reason = _normalize_pred_reason(
            refine.classification,
            refine.reason.strip(),
            default_reason=default_reason,
            debug_context=f"{debug_context}__sat_refine",
        )
        return {
            "pred_score": refine.classification,
            "pred_reason": final_reason,
            "analysis": f"[router] {router_analysis}\n[sat_refine] {refine.analysis}",
            "analysis_sat_refine": refine.analysis,
            "fullscale_branch": "sat_45",
            "fullscale_refine_applied": True,
            **base_result,
        }

    refine_prompt = build_turn_eval_fullscale_dsat_refinement_prompt(
        memory=memory,
        profile=session.profile,
        task_context=session.task_context,
        history_window=list(history_window),
        assistant_reply=assistant_reply,
        router_reason=router_reason,
        router_analysis=router_analysis,
        anchor_turns=anchors,
    )
    refine = call_predict_fn(
        refine_prompt,
        model,
        prompt_version="boundary_34_selective_refute_v2_fullscale_dsat_refine",
        debug_context=f"{debug_context}__dsat_refine",
    )
    assert isinstance(refine, DsatRefinementPrediction)
    final_reason = _normalize_pred_reason(
        refine.classification,
        refine.reason.strip(),
        default_reason=default_reason,
        debug_context=f"{debug_context}__dsat_refine",
    )
    return {
        "pred_score": refine.classification,
        "pred_reason": final_reason,
        "analysis": f"[router] {router_analysis}\n[dsat_refine] {refine.analysis}",
        "analysis_dsat_refine": refine.analysis,
        "fullscale_branch": "dsat_123",
        "fullscale_refine_applied": True,
        **base_result,
    }


def predict_turn_v3_two_stage(
    memory: UserMemory | UserMemoryV3 | None,
    session: SessionData,
    model: str,
    history_window: list[str],
    assistant_reply: str,
    debug_context: str,
    default_reason: str,
    call_predict_fn: CallPredictFn,
    anchors: list[AnchorTurn] | None = None,
) -> dict:
    """
    Memory v3 two-stage 1-5 pipeline:
    1. Stage 1: SAT gate (3/4).
    2. Stage 2: SAT refines to 4/5, DSAT refines to 1/2/3.
    """
    if memory is None:
        prompt = build_turn_eval_prompt_no_memory(
            profile=session.profile,
            task_context=session.task_context,
            history_window=list(history_window),
            assistant_reply=assistant_reply,
        )
        pred = call_predict_fn(
            prompt,
            model,
            prompt_version="v2",
            debug_context=debug_context,
        )
        assert isinstance(pred, TurnPrediction)
        return {
            "pred_score": pred.classification,
            "pred_reason": _normalize_pred_reason(
                pred.classification,
                pred.reason.strip(),
                default_reason=default_reason,
                debug_context=debug_context,
            ),
            "analysis": pred.analysis,
            "two_stage_gate_score": None,
            "two_stage_gate_reason": "",
            "two_stage_gate_analysis": "",
            "two_stage_branch": "no_memory_fallback",
            "two_stage_refine_applied": False,
        }

    gate_prompt = build_turn_eval_v3_two_stage_gate_prompt(
        memory=memory,
        profile=session.profile,
        task_context=session.task_context,
        history_window=list(history_window),
        assistant_reply=assistant_reply,
        anchor_turns=anchors,
    )
    gate_pred = call_predict_fn(
        gate_prompt,
        model,
        prompt_version="v3_two_stage_gate",
        debug_context=f"{debug_context}__gate",
    )
    assert isinstance(gate_pred, BoundaryTurnPrediction)
    gate_reason = _normalize_pred_reason(
        gate_pred.classification,
        gate_pred.reason.strip(),
        default_reason=default_reason,
        debug_context=f"{debug_context}__gate",
    )
    gate_analysis = gate_pred.analysis
    base_result = {
        "two_stage_gate_score": gate_pred.classification,
        "two_stage_gate_reason": gate_reason,
        "two_stage_gate_analysis": gate_analysis,
        "analysis_gate": gate_analysis,
    }

    if gate_pred.classification >= 4:
        refine_prompt = build_turn_eval_v3_two_stage_sat_refinement_prompt(
            memory=memory,
            profile=session.profile,
            task_context=session.task_context,
            history_window=list(history_window),
            assistant_reply=assistant_reply,
            gate_reason=gate_reason,
            gate_analysis=gate_analysis,
            anchor_turns=anchors,
        )
        refine = call_predict_fn(
            refine_prompt,
            model,
            prompt_version="v3_two_stage_sat_refine",
            debug_context=f"{debug_context}__sat_refine",
        )
        assert isinstance(refine, SatRefinementPrediction)
        final_reason = _normalize_pred_reason(
            refine.classification,
            refine.reason.strip(),
            default_reason=default_reason,
            debug_context=f"{debug_context}__sat_refine",
        )
        return {
            "pred_score": refine.classification,
            "pred_reason": final_reason,
            "analysis": f"[gate] {gate_analysis}\n[sat_refine] {refine.analysis}",
            "analysis_sat_refine": refine.analysis,
            "two_stage_branch": "sat_45",
            "two_stage_refine_applied": True,
            **base_result,
        }

    refine_prompt = build_turn_eval_v3_two_stage_dsat_refinement_prompt(
        memory=memory,
        profile=session.profile,
        task_context=session.task_context,
        history_window=list(history_window),
        assistant_reply=assistant_reply,
        gate_reason=gate_reason,
        gate_analysis=gate_analysis,
        anchor_turns=anchors,
    )
    refine = call_predict_fn(
        refine_prompt,
        model,
        prompt_version="v3_two_stage_dsat_refine",
        debug_context=f"{debug_context}__dsat_refine",
    )
    assert isinstance(refine, DsatRefinementPrediction)
    final_reason = _normalize_pred_reason(
        refine.classification,
        refine.reason.strip(),
        default_reason=default_reason,
        debug_context=f"{debug_context}__dsat_refine",
    )
    return {
        "pred_score": refine.classification,
        "pred_reason": final_reason,
        "analysis": f"[gate] {gate_analysis}\n[dsat_refine] {refine.analysis}",
        "analysis_dsat_refine": refine.analysis,
        "two_stage_branch": "dsat_123",
        "two_stage_refine_applied": True,
        **base_result,
    }


def predict_turn_v3_two_stage_v2(
    memory: UserMemory | UserMemoryV3 | None,
    session: SessionData,
    model: str,
    history_window: list[str],
    assistant_reply: str,
    debug_context: str,
    default_reason: str,
    call_predict_fn: CallPredictFn,
    anchors: list[AnchorTurn] | None = None,
) -> dict:
    """
    Memory v3 two-stage v2:
    1. First layer uses a selective-refute style SAT gate.
    2. Second layer keeps the existing 4/5 and 1/2/3 refinement prompts.
    """
    if memory is None:
        return predict_turn_v3_two_stage(
            memory=memory,
            session=session,
            model=model,
            history_window=history_window,
            assistant_reply=assistant_reply,
            debug_context=debug_context,
            default_reason=default_reason,
            call_predict_fn=call_predict_fn,
            anchors=anchors,
        )

    gate_prompt = build_turn_eval_v3_two_stage_v2_gate_prompt(
        memory=memory,
        profile=session.profile,
        task_context=session.task_context,
        history_window=list(history_window),
        assistant_reply=assistant_reply,
        anchor_turns=anchors,
    )
    gate_pred = call_predict_fn(
        gate_prompt,
        model,
        prompt_version="v3_two_stage_v2_gate",
        debug_context=f"{debug_context}__gate",
    )
    assert isinstance(gate_pred, SelectiveBoundaryTurnPrediction)
    gate_reason = _normalize_pred_reason(
        gate_pred.classification,
        gate_pred.reason.strip(),
        default_reason=default_reason,
        debug_context=f"{debug_context}__gate",
    )
    should_trigger_gate_refute = should_trigger_selective_refute(
        gate_pred,
        "v3_two_stage_v2_gate",
    )
    gate_score = gate_pred.classification
    gate_analysis = gate_pred.analysis
    gate_reason_final = gate_reason
    gate_followup_analysis = ""

    if should_trigger_gate_refute:
        followup_prompt = build_turn_eval_v3_two_stage_v2_gate_followup_prompt(
            memory=memory,
            profile=session.profile,
            task_context=session.task_context,
            history_window=list(history_window),
            assistant_reply=assistant_reply,
            initial_classification=gate_pred.classification,
            initial_reason=gate_reason,
            initial_analysis=gate_pred.analysis,
        )
        followup = call_predict_fn(
            followup_prompt,
            model,
            prompt_version="v3_two_stage_v2_gate_followup",
            debug_context=f"{debug_context}__gate_followup",
        )
        assert isinstance(followup, BoundaryTurnPrediction)
        gate_score = followup.classification
        gate_reason_final = _normalize_pred_reason(
            followup.classification,
            followup.reason.strip(),
            default_reason=default_reason,
            debug_context=f"{debug_context}__gate_followup",
        )
        gate_followup_analysis = followup.analysis
        gate_analysis = f"[first_pass] {gate_pred.analysis}\n[gate_followup] {followup.analysis}"

    base_result = {
        "two_stage_gate_score": gate_score,
        "two_stage_gate_reason": gate_reason_final,
        "two_stage_gate_analysis": gate_analysis,
        "analysis_gate": gate_analysis,
        "two_stage_gate_model_flag": gate_pred.needs_refute_review,
        "two_stage_gate_triggered": should_trigger_gate_refute,
        "two_stage_gate_refute_applied": should_trigger_gate_refute,
        "analysis_gate_first_pass": gate_pred.analysis,
        "analysis_gate_followup": gate_followup_analysis,
    }

    if gate_score >= 4:
        refine_prompt = build_turn_eval_v3_two_stage_sat_refinement_prompt(
            memory=memory,
            profile=session.profile,
            task_context=session.task_context,
            history_window=list(history_window),
            assistant_reply=assistant_reply,
            gate_reason=gate_reason_final,
            gate_analysis=gate_analysis,
            anchor_turns=anchors,
        )
        refine = call_predict_fn(
            refine_prompt,
            model,
            prompt_version="v3_two_stage_sat_refine",
            debug_context=f"{debug_context}__sat_refine",
        )
        assert isinstance(refine, SatRefinementPrediction)
        final_reason = _normalize_pred_reason(
            refine.classification,
            refine.reason.strip(),
            default_reason=default_reason,
            debug_context=f"{debug_context}__sat_refine",
        )
        return {
            "pred_score": refine.classification,
            "pred_reason": final_reason,
            "analysis": f"[gate] {gate_analysis}\n[sat_refine] {refine.analysis}",
            "analysis_sat_refine": refine.analysis,
            "two_stage_branch": "sat_45",
            "two_stage_refine_applied": True,
            **base_result,
        }

    refine_prompt = build_turn_eval_v3_two_stage_dsat_refinement_prompt(
        memory=memory,
        profile=session.profile,
        task_context=session.task_context,
        history_window=list(history_window),
        assistant_reply=assistant_reply,
        gate_reason=gate_reason_final,
        gate_analysis=gate_analysis,
        anchor_turns=anchors,
    )
    refine = call_predict_fn(
        refine_prompt,
        model,
        prompt_version="v3_two_stage_dsat_refine",
        debug_context=f"{debug_context}__dsat_refine",
    )
    assert isinstance(refine, DsatRefinementPrediction)
    final_reason = _normalize_pred_reason(
        refine.classification,
        refine.reason.strip(),
        default_reason=default_reason,
        debug_context=f"{debug_context}__dsat_refine",
    )
    return {
        "pred_score": refine.classification,
        "pred_reason": final_reason,
        "analysis": f"[gate] {gate_analysis}\n[dsat_refine] {refine.analysis}",
        "analysis_dsat_refine": refine.analysis,
        "two_stage_branch": "dsat_123",
        "two_stage_refine_applied": True,
        **base_result,
    }


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
