from __future__ import annotations

import os
from typing import Callable, Literal

from loguru import logger

from lib.anchor_retrieval import AnchorRetriever, AnchorTurn
from lib.memory import UserMemory, UserMemoryV3
from lib.personalized_data import PersonalizedSample, SessionData
from lib.satisfaction_constants import get_reason_to_id
from trace.personalized_predictions import _anchor_metadata, _retrieve_anchor_turns

MemoryUpdateMode = Literal["none", "per_session", "per_session_oracle", "per_turn"]
MemoryVersion = Literal["v2", "v3"]
MemoryUpdatePromptVersion = Literal["auto", "v2", "v2_1", "v2_2", "v2_3", "v2_4", "v2_5", "v3"]

OPTIONAL_TURN_KEYS = (
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
)


def _copy_optional_turn_keys(source: dict, target: dict) -> None:
    for optional_key in OPTIONAL_TURN_KEYS:
        if optional_key in source:
            target[optional_key] = source[optional_key]


def _predict_one_turn(
    memory: UserMemory | UserMemoryV3 | None,
    session: SessionData,
    model: str,
    history_window: list[str],
    assistant_reply: str,
    turn_eval_prompt_version: str,
    debug_context: str,
    default_reason: str,
    anchors: list[AnchorTurn] | None,
    predict_turn_with_optional_selective_refute_fn: Callable[..., dict],
    predict_turn_fullscale_from_boundary_v2_fn: Callable[..., dict],
    predict_turn_v3_two_stage_fn: Callable[..., dict],
    predict_turn_v3_two_stage_v2_fn: Callable[..., dict],
) -> dict:
    if turn_eval_prompt_version == "boundary_34_selective_refute_v2_fullscale":
        return predict_turn_fullscale_from_boundary_v2_fn(
            memory=memory,
            session=session,
            model=model,
            history_window=history_window,
            assistant_reply=assistant_reply,
            debug_context=debug_context,
            default_reason=default_reason,
            anchors=anchors,
        )
    if turn_eval_prompt_version == "v3_two_stage":
        return predict_turn_v3_two_stage_fn(
            memory=memory,
            session=session,
            model=model,
            history_window=history_window,
            assistant_reply=assistant_reply,
            debug_context=debug_context,
            default_reason=default_reason,
            anchors=anchors,
        )
    if turn_eval_prompt_version == "v3_two_stage_v2":
        return predict_turn_v3_two_stage_v2_fn(
            memory=memory,
            session=session,
            model=model,
            history_window=history_window,
            assistant_reply=assistant_reply,
            debug_context=debug_context,
            default_reason=default_reason,
            anchors=anchors,
        )
    return predict_turn_with_optional_selective_refute_fn(
        memory=memory,
        session=session,
        model=model,
        history_window=history_window,
        assistant_reply=assistant_reply,
        turn_eval_prompt_version=turn_eval_prompt_version,
        debug_context=debug_context,
        default_reason=default_reason,
        anchors=anchors,
    )


def evaluate_session_per_turn_update(
    memory: UserMemory | UserMemoryV3,
    session: SessionData,
    model: str,
    memory_model: str,
    history_window_size: int,
    valid_reasons: set[str],
    default_reason: str,
    update_memory_fn: Callable[..., UserMemory | UserMemoryV3],
    predict_turn_with_optional_selective_refute_fn: Callable[..., dict],
    predict_turn_fullscale_from_boundary_v2_fn: Callable[..., dict],
    predict_turn_v3_two_stage_fn: Callable[..., dict],
    predict_turn_v3_two_stage_v2_fn: Callable[..., dict],
    retriever: AnchorRetriever | None = None,
    n_anchors: int = 0,
    turn_eval_prompt_version: str = "v2",
    block_id: str = "",
    memory_version: MemoryVersion = "v2",
    memory_update_prompt_version: MemoryUpdatePromptVersion = "auto",
) -> list[dict]:
    """
    per_turn 模式：每预测一轮后立即更新记忆。
    由于需要顺序执行，不能并行化。
    """
    results: list[dict] = []
    history_window: list[str] = []
    history_window_dicts: list[dict] = []
    assistant_turn_idx = 0
    last_user_msg: str = ""

    for utt in session.history:
        if utt["role"] == "user":
            last_user_msg = utt["content"]
        if utt["role"] == "assistant":
            anchors: list[AnchorTurn] | None = None
            if retriever is not None and n_anchors > 0:
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
            pred_result = _predict_one_turn(
                memory=memory,
                session=session,
                model=model,
                history_window=history_window,
                assistant_reply=utt["content"],
                turn_eval_prompt_version=turn_eval_prompt_version,
                debug_context=debug_context,
                default_reason=default_reason,
                anchors=anchors,
                predict_turn_with_optional_selective_refute_fn=predict_turn_with_optional_selective_refute_fn,
                predict_turn_fullscale_from_boundary_v2_fn=predict_turn_fullscale_from_boundary_v2_fn,
                predict_turn_v3_two_stage_fn=predict_turn_v3_two_stage_fn,
                predict_turn_v3_two_stage_v2_fn=predict_turn_v3_two_stage_v2_fn,
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
            _copy_optional_turn_keys(pred_result, turn_result)
            results.append(turn_result)

            mini_session = SessionData(
                user=session.user,
                task=session.task,
                file_path=session.file_path,
                task_context=session.task_context,
                profile=session.profile,
                history=list(history_window_dicts) + [utt],
                satisfaction_scores=[gold_score],
                dissatisfaction_reasons=[gold_reason],
                chat_model=session.chat_model,
            )
            try:
                memory = update_memory_fn(
                    memory=memory,
                    session=mini_session,
                    turn_predictions=[turn_result],
                    model=memory_model,
                    use_oracle_labels=False,
                    memory_version=memory_version,
                    memory_update_prompt_version=memory_update_prompt_version,
                )
            except Exception as e:
                logger.warning(f"Per-turn memory update failed at turn {assistant_turn_idx}: {e}")

            assistant_turn_idx += 1

        role_label = "用户" if utt["role"] == "user" else "助手"
        history_window.append(f"{role_label}：{utt['content']}")
        history_window_dicts.append(utt)
        while len(history_window) > history_window_size:
            history_window.pop(0)
        while len(history_window_dicts) > history_window_size:
            history_window_dicts.pop(0)

    return results


def run_agent_on_sample(
    sample: PersonalizedSample,
    model: str,
    build_user_memory_fn: Callable[..., UserMemory | UserMemoryV3],
    evaluate_session_fn: Callable[..., list[dict]],
    update_memory_fn: Callable[..., UserMemory | UserMemoryV3],
    predict_turn_with_optional_selective_refute_fn: Callable[..., dict],
    predict_turn_fullscale_from_boundary_v2_fn: Callable[..., dict],
    predict_turn_v3_two_stage_fn: Callable[..., dict],
    predict_turn_v3_two_stage_v2_fn: Callable[..., dict],
    memory_update_mode: MemoryUpdateMode = "per_session",
    memory_version: MemoryVersion = "v2",
    memory_update_prompt_version: MemoryUpdatePromptVersion = "auto",
    history_window_size: int = 5,
    save_memory_snapshots: bool = False,
    memory_cache_dir: str | None = None,
    with_memory: bool = True,
    n_anchors: int = 0,
    turn_eval_prompt_version: str = "v2",
    memory_model: str | None = None,
) -> list[dict]:
    reason_to_id = get_reason_to_id()
    valid_reasons = set(reason_to_id.keys())
    default_reason = "其它" if "其它" in reason_to_id else next(iter(reason_to_id))

    memory_model_effective = memory_model or model
    memory = (
        build_user_memory_fn(
            sample,
            memory_model_effective,
            memory_cache_dir=memory_cache_dir,
            memory_version=memory_version,
        )
        if with_memory
        else None
    )

    retriever: AnchorRetriever | None = None
    if with_memory and n_anchors > 0:
        retriever = AnchorRetriever(sample.history_sessions)

    all_turn_records: list[dict] = []

    for session in sample.target_sessions:
        session_file = os.path.basename(session.file_path)

        if memory_update_mode == "per_turn":
            session_results = evaluate_session_per_turn_update(
                memory=memory,
                session=session,
                model=model,
                memory_model=memory_model_effective,
                history_window_size=history_window_size,
                valid_reasons=valid_reasons,
                default_reason=default_reason,
                update_memory_fn=update_memory_fn,
                predict_turn_with_optional_selective_refute_fn=predict_turn_with_optional_selective_refute_fn,
                predict_turn_fullscale_from_boundary_v2_fn=predict_turn_fullscale_from_boundary_v2_fn,
                predict_turn_v3_two_stage_fn=predict_turn_v3_two_stage_fn,
                predict_turn_v3_two_stage_v2_fn=predict_turn_v3_two_stage_v2_fn,
                retriever=retriever,
                n_anchors=n_anchors,
                turn_eval_prompt_version=turn_eval_prompt_version,
                block_id=sample.block_id,
                memory_version=memory_version,
                memory_update_prompt_version=memory_update_prompt_version,
            )
        else:
            session_results = evaluate_session_fn(
                memory=memory,
                session=session,
                model=model,
                history_window_size=history_window_size,
                valid_reasons=valid_reasons,
                default_reason=default_reason,
                retriever=retriever,
                n_anchors=n_anchors,
                turn_eval_prompt_version=turn_eval_prompt_version,
                block_id=sample.block_id,
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
                "memory_model": memory_model_effective if with_memory else "none",
                "with_memory": with_memory,
                "memory_update_mode": memory_update_mode if with_memory else "no_memory",
                "memory_version": memory.memory_version if memory is not None else "none",
                "memory_update_prompt_version": memory_update_prompt_version if with_memory else "none",
                "turn_eval_prompt_version": turn_eval_prompt_version,
            }
            _copy_optional_turn_keys(r, record)
            if memory_snapshot is not None:
                record["memory_snapshot"] = memory_snapshot
            all_turn_records.append(record)

        if with_memory and memory_update_mode in ("per_session", "per_session_oracle"):
            use_oracle = memory_update_mode == "per_session_oracle"
            try:
                memory = update_memory_fn(
                    memory=memory,
                    session=session,
                    turn_predictions=session_results,
                    model=memory_model_effective,
                    use_oracle_labels=use_oracle,
                    memory_version=memory_version,
                    memory_update_prompt_version=memory_update_prompt_version,
                )
            except Exception as e:
                logger.warning(
                    f"Memory update failed for {sample.user}/{session_file}: {e}, "
                    f"keeping existing memory."
                )

    return all_turn_records
