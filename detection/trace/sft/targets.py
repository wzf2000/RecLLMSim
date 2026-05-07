from __future__ import annotations

import json
from typing import Any

from .parsing import QWEN3_THINK_BEGIN, QWEN3_THINK_END


def format_qwen3_think_wrapped_assistant_target(reasoning: str, visible: str) -> str:
    """
    与 Qwen3 tokenizer chat_template 中带 reasoning 的 assistant 段一致：
    <think> + 思考 + </think> + 两个换行 + 可见段（如 JSON）。
    无思考时等价于模板里空的 think 块（仅换行）再接可见段。
    """
    r = (reasoning or "").strip("\n")
    v = visible.lstrip("\n")
    if r:
        return f"{QWEN3_THINK_BEGIN}\n{r}\n{QWEN3_THINK_END}\n\n{v}"
    return f"{QWEN3_THINK_BEGIN}\n\n{QWEN3_THINK_END}\n\n{v}"


def finalize_assistant_target_body(reasoning: str, visible: str, think_wrap: str) -> str:
    """在 teacher 的 reasoning / visible 之上套一层与推理对齐的格式。"""
    if think_wrap == "none":
        if (reasoning or "").strip():
            return f"{reasoning.strip()}\n\n{visible.lstrip()}"
        return visible.lstrip("\n")
    if think_wrap == "qwen3":
        return format_qwen3_think_wrapped_assistant_target(reasoning, visible)
    raise ValueError(f"Unknown think_wrap: {think_wrap}")


def build_assistant_target(
    row: dict[str, Any],
    include_reasoning_content: bool = False,
    think_wrap: str = "qwen3",
) -> str:
    reasoning_part = ""
    visible_part = ""

    ref = row.get("reflection")
    if isinstance(ref, dict):
        revised = str(ref.get("revised_reasoning") or "").strip()
        if revised:
            visible_part = json.dumps(
                {
                    "classification": int(row["gold_score"]),
                    "reason": str(row["gold_reason"]),
                    "analysis": revised,
                },
                ensure_ascii=False,
            )
            ref_rc = ref.get("reasoning_content")
            ref_reasoning = ref_rc.strip() if isinstance(ref_rc, str) and ref_rc.strip() else ""
            if include_reasoning_content and ref_reasoning:
                reasoning_part = ref_reasoning
            return finalize_assistant_target_body(reasoning_part, visible_part, think_wrap)

    reasoning_content = row.get("reasoning_content")
    raw_content = row.get("raw_content")
    raw_text = raw_content.strip() if isinstance(raw_content, str) and raw_content.strip() else ""
    reasoning_text = reasoning_content.strip() if isinstance(reasoning_content, str) and reasoning_content.strip() else ""
    if raw_text:
        if include_reasoning_content and reasoning_text:
            reasoning_part = reasoning_text
        visible_part = raw_text
        return finalize_assistant_target_body(reasoning_part, visible_part, think_wrap)

    fallback = {
        "classification": int(row["prediction"]),
        "reason": row["reason_prediction"],
        "analysis": row.get("analysis", ""),
    }
    visible_part = json.dumps(fallback, ensure_ascii=False)
    return finalize_assistant_target_body("", visible_part, think_wrap)
