from __future__ import annotations

from typing import Literal

from .satisfaction_constants import SATISFIED_REASON, get_reason_to_id
from .memory_schema import UserMemory
from .memory_formatting import (
    _MAX_REPLY_CHARS,
    _format_profile,
    _format_reason_json_rule,
    _format_reason_rule_block,
    _truncate,
)
from .memory_eval_common import _format_anchor_turns


def build_turn_eval_prompt_no_memory(
    profile: dict,
    task_context: str,
    history_window: list[str],
    assistant_reply: str,
) -> str:
    """无记忆 baseline prompt（保持不变）。"""
    reason_labels = list(get_reason_to_id().keys())
    reason_text = "、".join(reason_labels)
    reason_rule_block = _format_reason_rule_block()
    reason_json_rule = _format_reason_json_rule()
    history_text = "\n".join(history_window) if history_window else "（无历史对话）"

    prompt = (
        "你是一名会进行细粒度对话质量分析的评估员。\n"
        "请基于给定信息先进行推理，再同时预测：\n"
        "1) 当前用户对助手回复的满意度分数（1-5）\n"
        "2) 潜在原因（只有在分数 <=3 时才选择不满意原因；分数 >=4 时必须为【满意】）\n\n"
        f"【用户画像】{_format_profile(profile)}\n\n"
        f"【任务背景】{task_context}\n\n"
        f"【最近对话历史】\n{history_text}\n\n"
        f"【当前助手回复】{assistant_reply}\n\n"
        f"【可选原因标签】{reason_text}\n{reason_rule_block}\n"
        "请严格输出 JSON，不要输出其他内容：\n"
        "{\n"
        '  "classification": 1-5 中的整数,\n'
        f'  "reason": "{reason_json_rule}",\n'
        '  "analysis": "你的详细推理过程"\n'
        "}\n"
    )
    return prompt
