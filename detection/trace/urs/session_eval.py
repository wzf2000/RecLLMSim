from __future__ import annotations

import json
import os
from typing import Literal

from loguru import logger
from pydantic import BaseModel, Field
from tenacity import before_sleep_log, retry, stop_after_attempt, wait_fixed

from lib.memory import UserMemory
from lib.personalized_data import SessionData
from lib.satisfaction_constants import is_reason_valid_for_score, normalize_reason_for_score
from lib.urs_episodic import UrsEpisodicMemoryRecord
from lib.urs_episodic_prompts import (
    build_urs_dsat_failure_check_prompt,
    build_urs_episodic_task_guarded_prompt,
    build_urs_episodic_twostage_score_prompt,
)
from lib.urs_memory import build_session_eval_prompt, build_session_eval_prompt_no_memory
from trace.collect_personalized import StructuredOutputError, TurnPrediction, _structured_parse


class DsatFailureCheckPrediction(BaseModel):
    same_failure_as_dsat_evidence: bool
    matched_evidence_ids: list[str] = Field(default_factory=list)
    failure_type: Literal[
        "core_missing",
        "missing_constraint",
        "generic_unusable",
        "unverifiable_or_wrong",
        "format_mismatch",
        "off_task",
        "none",
        "other",
    ]
    confidence: Literal["low", "medium", "high"]
    analysis: str


def _normalize_pred_reason(
    pred_score: int,
    pred_reason: str,
    default_reason: str,
    debug_context: str = "",
) -> str:
    normalized = normalize_reason_for_score(
        pred_score,
        pred_reason,
        default_reason=default_reason,
    )
    if not is_reason_valid_for_score(pred_score, pred_reason):
        context = f" for {debug_context}" if debug_context else ""
        logger.warning(
            f"Normalized invalid reason/score pair{context}: "
            f"score={pred_score}, raw_reason={pred_reason} -> {normalized}"
        )
    return normalized


@retry(
    stop=stop_after_attempt(3),
    wait=wait_fixed(5),
    before_sleep=before_sleep_log(logger, log_level=40),
)
def _call_predict_session(
    prompt: str,
    model: str,
    debug_context: str = "",
) -> TurnPrediction:
    try:
        return _structured_parse(
            prompt,
            model,
            TurnPrediction,
            temperature=0.3,
            timeout=120,
            system_msg="You are a skilled conversational analyst.",
        )
    except Exception as e:
        dump_dir = "outputs/urs/parse_failures"
        os.makedirs(dump_dir, exist_ok=True)
        safe_context = "".join(
            c if c.isalnum() or c in {"_", "-", "."} else "_"
            for c in (debug_context or "unknown_context")
        )[:160]
        prefix = os.path.join(dump_dir, safe_context)
        meta = {
            "debug_context": debug_context,
            "model": model,
            "prompt_length": len(prompt),
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
        raise


@retry(
    stop=stop_after_attempt(3),
    wait=wait_fixed(5),
    before_sleep=before_sleep_log(logger, log_level=40),
)
def _call_dsat_failure_check(
    prompt: str,
    model: str,
    debug_context: str = "",
) -> DsatFailureCheckPrediction:
    try:
        return _structured_parse(
            prompt,
            model,
            DsatFailureCheckPrediction,
            temperature=0.2,
            timeout=120,
            system_msg="You are a strict failure-evidence arbiter.",
        )
    except Exception as e:
        dump_dir = "outputs/urs/parse_failures"
        os.makedirs(dump_dir, exist_ok=True)
        safe_context = "".join(
            c if c.isalnum() or c in {"_", "-", "."} else "_"
            for c in (debug_context or "unknown_context")
        )[:160]
        prefix = os.path.join(dump_dir, safe_context)
        meta = {
            "debug_context": debug_context,
            "model": model,
            "prompt_length": len(prompt),
            "response_model": "DsatFailureCheckPrediction",
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
        raise


def evaluate_urs_session(
    memory: UserMemory | None,
    session: SessionData,
    model: str,
    valid_reasons: set[str],
    default_reason: str,
    block_id: str = "",
    prompt_version: str = "v2",
) -> list[dict]:
    """对单个 URS session 出 1 条预测。返回形式与 evaluate_session 保持一致（单元素）。"""
    if memory is None:
        prompt = build_session_eval_prompt_no_memory(
            profile=session.profile,
            task_context=session.task_context,
            session_history=session.history,
            prompt_version=prompt_version,
        )
    else:
        prompt = build_session_eval_prompt(
            memory=memory,
            profile=session.profile,
            task_context=session.task_context,
            session_history=session.history,
            prompt_version=prompt_version,
        )

    debug_context = (
        f"{block_id}__{os.path.basename(session.file_path)}"
        if block_id else os.path.basename(session.file_path)
    )
    pred = _call_predict_session(prompt, model, debug_context=debug_context)

    pred_reason = _normalize_pred_reason(
        pred.classification,
        pred.reason.strip(),
        default_reason=default_reason,
        debug_context=debug_context,
    )
    if pred_reason not in valid_reasons:
        pred_reason = default_reason

    gold_score = session.satisfaction_scores[0]
    gold_reason = session.dissatisfaction_reasons[0]

    return [{
        "turn_idx": 0,
        "pred_score": pred.classification,
        "pred_reason": pred_reason,
        "gold_score": gold_score,
        "gold_reason": gold_reason,
        "analysis": pred.analysis,
    }]


def evaluate_urs_session_episodic(
    retrieved_memories: list[UrsEpisodicMemoryRecord],
    session: SessionData,
    model: str,
    valid_reasons: set[str],
    default_reason: str,
    block_id: str = "",
    prompt_version: str = "urs_episodic_task_guarded",
) -> list[dict]:
    """Evaluate one URS session with raw episodic retrieval memory."""
    debug_context = (
        f"{block_id}__{os.path.basename(session.file_path)}__{prompt_version}"
        if block_id else f"{os.path.basename(session.file_path)}__{prompt_version}"
    )
    failure_check: DsatFailureCheckPrediction | None = None
    if prompt_version == "urs_episodic_task_guarded_dsat_twostage":
        check_prompt = build_urs_dsat_failure_check_prompt(
            profile=session.profile,
            task_context=session.task_context,
            session_history=session.history,
            retrieved_memories=retrieved_memories,
        )
        failure_check = _call_dsat_failure_check(
            check_prompt,
            model,
            debug_context=f"{debug_context}__stage1_dsat_check",
        )
        prompt = build_urs_episodic_twostage_score_prompt(
            profile=session.profile,
            task_context=session.task_context,
            session_history=session.history,
            retrieved_memories=retrieved_memories,
            failure_check=failure_check.model_dump(),
        )
    else:
        prompt = build_urs_episodic_task_guarded_prompt(
            profile=session.profile,
            task_context=session.task_context,
            session_history=session.history,
            retrieved_memories=retrieved_memories,
            prompt_version=prompt_version,
        )
    pred = _call_predict_session(prompt, model, debug_context=debug_context)

    raw_pred_score = pred.classification
    gate_applied = False
    if (
        failure_check is not None
        and failure_check.same_failure_as_dsat_evidence
        and failure_check.confidence in {"medium", "high"}
        and pred.classification >= 4
    ):
        pred.classification = 3
        gate_applied = True

    pred_reason = _normalize_pred_reason(
        pred.classification,
        pred.reason.strip(),
        default_reason=default_reason,
        debug_context=debug_context,
    )
    if pred_reason not in valid_reasons:
        pred_reason = default_reason

    gold_score = session.satisfaction_scores[0]
    gold_reason = session.dissatisfaction_reasons[0]

    return [{
        "turn_idx": 0,
        "pred_score": pred.classification,
        "pred_reason": pred_reason,
        "gold_score": gold_score,
        "gold_reason": gold_reason,
        "analysis": pred.analysis,
        "retrieved_memory_ids": [m.memory_id for m in retrieved_memories],
        "retrieved_scores": [m.score for m in retrieved_memories],
        "retrieved_tasks": [m.source_task for m in retrieved_memories],
        "retrieved_roles": [m.evidence_role for m in retrieved_memories],
        "retrieved_similarities": [m.similarity for m in retrieved_memories],
        "dsat_failure_check": failure_check.model_dump() if failure_check is not None else None,
        "dsat_gate_applied": gate_applied,
        "raw_pred_score_before_dsat_gate": raw_pred_score,
    }]


# ──────────────────────────────────────────────────────────────────────────────
# Memory update（复用 personalized 的 update_memory，但 per_turn 不支持）
# ──────────────────────────────────────────────────────────────────────────────
