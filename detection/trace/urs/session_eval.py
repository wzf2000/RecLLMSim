from __future__ import annotations

import json
import os

from loguru import logger
from tenacity import before_sleep_log, retry, stop_after_attempt, wait_fixed

from lib.memory import UserMemory
from lib.personalized_data import SessionData
from lib.satisfaction_constants import is_reason_valid_for_score, normalize_reason_for_score
from lib.urs_memory import build_session_eval_prompt, build_session_eval_prompt_no_memory
from trace.collect_personalized import StructuredOutputError, TurnPrediction, _structured_parse


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


def evaluate_urs_session(
    memory: UserMemory | None,
    session: SessionData,
    model: str,
    valid_reasons: set[str],
    default_reason: str,
    block_id: str = "",
) -> list[dict]:
    """对单个 URS session 出 1 条预测。返回形式与 evaluate_session 保持一致（单元素）。"""
    if memory is None:
        prompt = build_session_eval_prompt_no_memory(
            profile=session.profile,
            task_context=session.task_context,
            session_history=session.history,
        )
    else:
        prompt = build_session_eval_prompt(
            memory=memory,
            profile=session.profile,
            task_context=session.task_context,
            session_history=session.history,
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


# ──────────────────────────────────────────────────────────────────────────────
# Memory update（复用 personalized 的 update_memory，但 per_turn 不支持）
# ──────────────────────────────────────────────────────────────────────────────

