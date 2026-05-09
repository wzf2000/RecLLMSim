from __future__ import annotations

from .memory_formatting import _MAX_REPLY_CHARS, _truncate


def _format_anchor_turns(anchor_turns: list) -> str:
    """Format retrieved anchor turns as calibrated reference cases."""
    if not anchor_turns:
        return ""
    has_evidence_roles = any(getattr(a, "evidence_role", "") for a in anchor_turns)
    if has_evidence_roles:
        usage = (
            "用法：这些是从该用户过往 session 中检索到的边界成对证据。"
            "请比较当前回复更接近 DSAT-side（<=3）还是 SAT-side（>=4），"
            "并结合 summary memory 判断 3/4 满意边界；不要机械复制案例分数。"
        )
    else:
        usage = (
            "用法：这些是从该用户过往 session 中检索到的、与当前回复文本最相似的若干轮次。"
            "请把它们按分数排列当作【已校准的参考刻度】，将当前回复在整体质量维度上与其对齐——"
            "若当前回复和某个案例的整体质量处于同一档位，就直接给相同的分数。"
            "不要把这些案例当作【完美标杆】去挑当前回复的毛病。"
        )
    lines = [
        "═══ 该用户历史上的参考案例（真实标注分数） ═══",
        usage,
        "",
    ]
    for i, a in enumerate(anchor_turns, 1):
        tag = f"★{a.score}" + (f"（{a.reason}）" if a.score <= 3 else "")
        user_snip = _truncate(a.user_msg, 120)
        reply_snip = _truncate(a.assistant_reply, _MAX_REPLY_CHARS)
        role = getattr(a, "evidence_role", "")
        role_text = f"  证据侧：{role}" if role else ""
        lines.append(f"[案例 {i}] 任务：{a.task}  真实满意度：{tag}{role_text}")
        lines.append(f"  用户提问：{user_snip}")
        lines.append(f"  助手回复：{reply_snip}")
        lines.append("")
    return "\n".join(lines)
