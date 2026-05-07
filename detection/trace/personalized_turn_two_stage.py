"""Fullscale and two-stage personalized turn prediction flows."""

from __future__ import annotations

from typing import Callable

from pydantic import BaseModel

from lib.anchor_retrieval import AnchorTurn
from lib.memory import (
    UserMemory,
    UserMemoryV3,
    build_turn_eval_fullscale_dsat_refinement_prompt,
    build_turn_eval_fullscale_sat_refinement_prompt,
    build_turn_eval_prompt_no_memory,
    build_turn_eval_v3_two_stage_dsat_refinement_prompt,
    build_turn_eval_v3_two_stage_gate_prompt,
    build_turn_eval_v3_two_stage_sat_refinement_prompt,
    build_turn_eval_v3_two_stage_v2_gate_followup_prompt,
    build_turn_eval_v3_two_stage_v2_gate_prompt,
)
from lib.personalized_data import SessionData
from trace.personalized_predictions import (
    BoundaryTurnPrediction,
    DsatRefinementPrediction,
    SatRefinementPrediction,
    SelectiveBoundaryTurnPrediction,
    TurnPrediction,
    _normalize_pred_reason,
)
from trace.personalized_turn_selective import (
    predict_turn_with_optional_selective_refute,
    should_trigger_selective_refute,
)

CallPredictFn = Callable[..., BaseModel]

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
