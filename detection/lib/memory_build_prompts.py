"""Initial user-memory construction prompts."""

from __future__ import annotations

from .personalized_data import SessionData
from .satisfaction_constants import get_reason_to_id
from .memory_formatting import (
    _MAX_EXAMPLES_PER_SCORE,
    _collect_turns_by_score,
    _format_profile,
    _format_score_group,
    _select_sessions,
    _truncate,
)

# ──────────────────────────────────────────────────────────────────────────────
# Memory Building Prompt（v2：对比式）
# ──────────────────────────────────────────────────────────────────────────────

def build_memory_prompt(
    user_id: str,
    profile: dict,
    history_sessions: list[SessionData],
) -> str:
    """
    构造 memory building prompt（v2）。

    核心改进：在顺序展示 session 后，额外提供"按分数分组的对比证据"，
    迫使 LLM 直接对比 4 分和 5 分轮次的差异，避免生成泛化描述。
    """
    sessions_to_use = _select_sessions(history_sessions)
    reason_labels = list(get_reason_to_id().keys())

    # ── Part 1: 按任务顺序展示 session（保留对话上下文）─────────────────────
    session_lines: list[str] = []
    for idx, session in enumerate(sessions_to_use):
        lines = [
            f"【Session {idx + 1}】任务：{session.task}  "
            f"任务背景：{_truncate(session.task_context, 150)}",
        ]
        assistant_idx = 0
        for utt in session.history:
            role = "用户" if utt["role"] == "user" else "助手"
            content = _truncate(utt["content"])
            lines.append(f"  {role}：{content}")
            if utt["role"] == "assistant":
                score = session.satisfaction_scores[assistant_idx]
                reason = session.dissatisfaction_reasons[assistant_idx]
                tag = f"★{score}" + (f"（{reason}）" if score <= 3 else "")
                lines.append(f"  [满意度: {tag}]")
                assistant_idx += 1
        session_lines.append("\n".join(lines))

    session_block = "\n\n".join(session_lines)

    # ── Part 2: 按分数分组的对比证据（关键新增）─────────────────────────────
    turns_by_score = _collect_turns_by_score(sessions_to_use)
    contrast_lines: list[str] = []
    # 只展示有实际数据的分数级别，优先展示边界处（4 vs 5，3 vs 4）
    for score in [5, 4, 3, 2, 1]:
        turns = turns_by_score.get(score, [])
        if turns:
            contrast_lines.append(
                _format_score_group(score, turns, _MAX_EXAMPLES_PER_SCORE)
            )
    contrast_block = "\n\n".join(contrast_lines) if contrast_lines else "（无数据）"

    # ── 统计摘要 ──────────────────────────────────────────────────────────────
    all_scores = [
        s for session in sessions_to_use
        for s in session.satisfaction_scores
    ]
    avg = sum(all_scores) / len(all_scores) if all_scores else 0
    dist = {i: all_scores.count(i) for i in range(1, 6)}
    stat_line = (
        f"总轮数：{len(all_scores)}，平均分：{avg:.2f}，"
        f"分布：{' / '.join(f'{i}分×{dist[i]}' for i in range(1,6))}"
    )

    prompt = (
        "你是一名用户行为分析专家。请基于以下用户的历史对话记录，"
        "建立一份精准的个性化用户记忆，用于预测该用户对未来助手回复的满意度。\n\n"
        f"【用户画像】{_format_profile(profile)}\n"
        f"【满意度统计】{stat_line}\n\n"
        "═══ 历史对话（按任务顺序）═══\n"
        f"{session_block}\n\n"
        "═══ 按分数分组的对比证据（重点参考）═══\n"
        f"{contrast_block}\n\n"
        "═══ 分析任务 ═══\n"
        "请严格基于以上对比证据完成以下分析，不得使用对所有用户都成立的泛化描述：\n\n"
        "1. 【评分边界 4→5】：对比 5 分和 4 分轮次，"
        "指出哪些具体要素决定了能否从 4 分升至 5 分（必须引用上面的实际例子）\n"
        "2. 【评分边界 3→4】：对比 4 分和 3 分（及以下）轮次，"
        "指出导致从 4 分跌至 3 分的具体缺陷类型\n"
        "3. 【评分风格】：该用户是偏严格还是偏宽松？结合平均分给出校准说明\n"
        "4. 【用户特异性要求】：该用户有哪些一般用户没有的特定要求？"
        "（如果所有用户都会这样要求，则不算特异性）\n"
        "5. 【偏好格式】：该用户偏好什么回复结构或组织形式？\n"
        "6. 【任务观察】：各任务类型下有哪些特殊偏好？\n\n"
        f"可参考的不满意原因类别：{', '.join(reason_labels)}\n\n"
        "请严格按照 JSON Schema 输出，不要输出其他内容。"
    )
    return prompt


def build_memory_prompt_v3(
    user_id: str,
    profile: dict,
    history_sessions: list[SessionData],
) -> str:
    """
    构造 memory building prompt（v3）。

    相比 v2，v3 的重点不是再增加字段，而是：
      1. 显式声明哪些边界缺直接证据
      2. 要求 LLM 在证据不足时输出保守、非确定性的总结
      3. 把 calibration 与 rule extraction 分开
    """
    sessions_to_use = _select_sessions(history_sessions)
    reason_labels = list(get_reason_to_id().keys())

    session_lines: list[str] = []
    for idx, session in enumerate(sessions_to_use):
        lines = [
            f"【Session {idx + 1}】任务：{session.task}  "
            f"任务背景：{_truncate(session.task_context, 150)}",
        ]
        assistant_idx = 0
        for utt in session.history:
            role = "用户" if utt["role"] == "user" else "助手"
            content = _truncate(utt["content"])
            lines.append(f"  {role}：{content}")
            if utt["role"] == "assistant":
                score = session.satisfaction_scores[assistant_idx]
                reason = session.dissatisfaction_reasons[assistant_idx]
                tag = f"★{score}" + (f"（{reason}）" if score <= 3 else "")
                lines.append(f"  [满意度: {tag}]")
                assistant_idx += 1
        session_lines.append("\n".join(lines))

    session_block = "\n\n".join(session_lines)

    turns_by_score = _collect_turns_by_score(sessions_to_use)
    contrast_lines: list[str] = []
    for score in [5, 4, 3, 2, 1]:
        turns = turns_by_score.get(score, [])
        if turns:
            contrast_lines.append(
                _format_score_group(score, turns, _MAX_EXAMPLES_PER_SCORE)
            )
    contrast_block = "\n\n".join(contrast_lines) if contrast_lines else "（无数据）"

    all_scores = [
        s for session in sessions_to_use
        for s in session.satisfaction_scores
    ]
    avg = sum(all_scores) / len(all_scores) if all_scores else 0
    dist = {i: all_scores.count(i) for i in range(1, 6)}
    stat_line = (
        f"总轮数：{len(all_scores)}，平均分：{avg:.2f}，"
        f"分布：{' / '.join(f'{i}分×{dist[i]}' for i in range(1,6))}"
    )
    evidence_lines = [
        "═══ 证据充分性提醒（必须遵守）═══",
        f"- 3分轮次：{dist[3]}，4分轮次：{dist[4]}，5分轮次：{dist[5]}",
        f"- 1/2/3 总低分轮次：{dist[1] + dist[2] + dist[3]}",
    ]
    if dist[3] == 0 or dist[4] == 0:
        evidence_lines.append("- 3/4 缺直接相邻证据：three_vs_four_distinction 必须写成【弱推断 / 证据不足】，不能写成确定性硬规则。")
    else:
        evidence_lines.append("- 3/4 有直接相邻证据：可以总结较可靠的满意最低线。")
    if dist[4] == 0 or dist[5] == 0:
        evidence_lines.append("- 4/5 缺直接相邻证据：four_vs_five_distinction 必须写成【弱推断 / 证据不足】，不能写成确定性硬规则。")
    else:
        evidence_lines.append("- 4/5 有直接相邻证据：可以总结较可靠的 5 分门槛。")
    if dist[1] + dist[2] + dist[3] <= 3:
        evidence_lines.append("- 低分样本很少：不要过度总结 1/2/3 的严重度差别，只能给出保守结论。")
    evidence_block = "\n".join(evidence_lines)

    prompt = (
        "你是一名用户行为分析专家。请基于以下用户的历史对话记录，"
        "建立一份精准但保守的个性化用户记忆，用于预测该用户对未来助手回复的满意度。\n\n"
        f"【用户画像】{_format_profile(profile)}\n"
        f"【满意度统计】{stat_line}\n\n"
        f"{evidence_block}\n\n"
        "═══ 历史对话（按任务顺序）═══\n"
        f"{session_block}\n\n"
        "═══ 按分数分组的对比证据（重点参考）═══\n"
        f"{contrast_block}\n\n"
        "═══ 分析任务 ═══\n"
        "请严格基于以上证据完成分析，不得编造历史中未出现的边界规律。\n\n"
        "分析原则：\n"
        "1. 先总结【校准信息】：该用户整体偏严格还是偏宽松，平均打分处在哪个区间。\n"
        "2. 再总结【边界规则】：只有在相邻分数证据存在时，才允许写成较确定的边界规则。\n"
        "3. 若缺相邻证据，必须明确写成“证据不足，只能弱推断”，不能写成确定性判断。\n"
        "4. user_specific_requirements 只保留真正会改变评分的个性化要求，不要重复“结构清晰、详细具体”这类通用要求。\n\n"
        "需要输出的内容：\n"
        "1. 【评分边界 4→5】：对比 5 分和 4 分轮次，指出哪些具体要素决定了能否从 4 分升至 5 分；"
        "若缺证据，明确说明证据不足\n"
        "2. 【评分边界 3→4】：对比 4 分和 3 分（及以下）轮次，指出导致从 4 分跌至 3 分的具体缺陷类型；"
        "若缺证据，明确说明证据不足\n"
        "3. 【评分风格】：该用户整体打分刻度如何\n"
        "4. 【用户特异性要求】：只保留最能改变评分的 1-4 条要求\n"
        "5. 【偏好格式】：如果只是通用偏好，可简短概括，不必展开\n"
        "6. 【任务观察】：各任务类型下有哪些特殊偏好\n\n"
        f"可参考的不满意原因类别：{', '.join(reason_labels)}\n\n"
        "请严格按照 JSON Schema 输出，不要输出其他内容。"
    )
    return prompt
