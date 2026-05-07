"""Shared formatting helpers for user-memory prompts."""

from __future__ import annotations

from collections import defaultdict

from .personalized_data import SessionData
from .satisfaction_constants import (
    SATISFIED_REASON,
    get_dissatisfied_reasons,
)

# ──────────────────────────────────────────────────────────────────────────────
# 内部工具
# ──────────────────────────────────────────────────────────────────────────────

_MAX_REPLY_CHARS = 200     # 单条 assistant 回复截断长度（压缩以降低 prompt token 数）
_MAX_SESSIONS_PROMPT = 8   # 放入 prompt 的最大 session 数（留足输出空间）
_MAX_EXAMPLES_PER_SCORE = 3  # 每个分数等级最多展示的 turn 例子数


def _format_profile(profile: dict) -> str:
    gender = "女" if profile.get("gender") == "Female" else "男"
    parts = [
        f"性别：{gender}",
        f"年龄：{profile.get('age', '未知')}",
        f"职业：{profile.get('occupation', '未知')}",
        f"背景：{profile.get('background', '未知')}",
        f"性格：{'，'.join(profile.get('personality', []))}",
        f"兴趣：{'，'.join(profile.get('daily_interests', []))}",
    ]
    return "  ".join(parts)


def _truncate(text: str, max_chars: int = _MAX_REPLY_CHARS) -> str:
    return text if len(text) <= max_chars else text[:max_chars] + "…"


def _format_reason_rule_block() -> str:
    dissatisfied_reason_text = "、".join(get_dissatisfied_reasons())
    return (
        "【原因标签合法性规则】\n"
        f"- 只有当 classification <= 3 时，reason 才能从以下不满意原因中选择："
        f"{dissatisfied_reason_text}\n"
        f"- 只要 classification >= 4，reason 必须输出 `{SATISFIED_REASON}`。\n"
        "- 如果 reason 与 classification 不一致，则该输出视为不合法。\n"
    )


def _format_reason_json_rule() -> str:
    return (
        f'若 classification >= 4 必须输出 "{SATISFIED_REASON}"；'
        '若 classification <= 3 只能从其余不满意原因标签中选择一个'
    )


def _collect_turns_by_score(
    sessions: list[SessionData],
) -> dict[int, list[dict]]:
    """
    从 sessions 中提取所有 (用户问题, 助手回复, 分数, 任务) 四元组，
    按分数分组返回。
    """
    by_score: dict[int, list[dict]] = defaultdict(list)
    for session in sessions:
        assistant_idx = 0
        last_user_msg = ""
        for utt in session.history:
            if utt["role"] == "user":
                last_user_msg = utt["content"]
            elif utt["role"] == "assistant":
                if assistant_idx < len(session.satisfaction_scores):
                    score = session.satisfaction_scores[assistant_idx]
                    reason = session.dissatisfaction_reasons[assistant_idx]
                    by_score[score].append({
                        "task": session.task,
                        "user_msg": last_user_msg,
                        "assistant_reply": utt["content"],
                        "score": score,
                        "reason": reason,
                    })
                    assistant_idx += 1
    return dict(by_score)


def _format_score_group(score: int, turns: list[dict], max_examples: int) -> str:
    """将同一分数的若干 turn 格式化为对比证据块。"""
    examples = turns[:max_examples]
    lines = [f"▸ {score} 分轮次（共 {len(turns)} 轮，展示 {len(examples)} 条）："]
    for i, t in enumerate(examples, 1):
        lines.append(f"  [{i}] 任务：{t['task']}")
        lines.append(f"      用户提问：{_truncate(t['user_msg'], 120)}")
        lines.append(f"      助手回复：{_truncate(t['assistant_reply'])}")
        if t["reason"] != "满意":
            lines.append(f"      不满意原因：{t['reason']}")
    return "\n".join(lines)


def _select_sessions(sessions: list[SessionData]) -> list[SessionData]:
    """按任务均匀采样，保留最多 _MAX_SESSIONS_PROMPT 个 session。"""
    if len(sessions) <= _MAX_SESSIONS_PROMPT:
        return sessions
    by_task: dict[str, list[SessionData]] = defaultdict(list)
    for s in sessions:
        by_task[s.task].append(s)
    selected: list[SessionData] = []
    per_task = max(1, _MAX_SESSIONS_PROMPT // len(by_task))
    for task_sessions in by_task.values():
        selected.extend(task_sessions[:per_task])
    return selected[:_MAX_SESSIONS_PROMPT]
