"""
URS session-level 评分 prompt builders

与 lib/memory.py 的 turn-level prompt 对应：URS 的 label 是 session-level（整段对话一个
1-5 分），所以评分 prompt 把整段对话作为评估对象，输出 1 个分数。

本模块只引入 session-level prompt，memory schema / UserMemory / build_memory_prompt 等
完全复用 lib/memory.py 的现有实现（把 SessionData.satisfaction_scores 当成单元素列表
使用，语义一致）。
"""

from __future__ import annotations

from .memory import (
    UserMemory,
    UserMemoryContent,
    _format_score_group,
    _format_profile,
    _format_reason_json_rule,
    _format_reason_rule_block,
    _select_sessions,
    _truncate,
)
from .satisfaction_constants import get_reason_to_id

UrsPromptVersion = str


def _format_session_dialogue(history: list[dict], max_chars: int = 400) -> str:
    """把整段对话渲染成 "用户：... / 助手：..." 形式的多行文本。"""
    lines: list[str] = []
    for utt in history:
        role_label = "用户" if utt.get("role") == "user" else "助手"
        content = _truncate(utt.get("content", ""), max_chars)
        lines.append(f"{role_label}：{content}")
    return "\n".join(lines) if lines else "（空对话）"


def _collect_sessions_by_score(sessions: list) -> dict[int, list[dict]]:
    """URS session-level: 每个 session 只有一个整体标签。"""
    by_score: dict[int, list[dict]] = {1: [], 2: [], 3: [], 4: [], 5: []}
    for session in sessions:
        if not session.satisfaction_scores:
            continue
        score = int(session.satisfaction_scores[0])
        reason = (
            session.dissatisfaction_reasons[0]
            if session.dissatisfaction_reasons else "满意"
        )
        by_score[score].append({
            "task": session.task,
            "user_msg": _truncate(session.task_context, 120),
            "assistant_reply": _format_session_dialogue(session.history, max_chars=220),
            "score": score,
            "reason": reason,
        })
    return by_score


def _format_urs_calibration_block(prompt_version: UrsPromptVersion) -> str:
    if prompt_version not in {"urs_v2_calibrated", "urs_v2_memory_guarded"}:
        return ""
    return (
        "【URS 评分刻度校准】\n"
        "- 这是 URS session-level 评分：请评价用户对整段对话的总体满意度，不要只因局部小瑕疵下调到 3 分。\n"
        "- 5 分：整体非常满意；回答充分、准确、贴合需求，结构清晰，并明显超过基本要求。不要求绝对完美。\n"
        "- 4 分：整体满意；主要需求已被满足，即使存在轻微遗漏、表达不够优雅或少量可改进点，也应给 4 分而不是 3 分。\n"
        "- 3 分：一般/中性；只有部分满足需求，存在明显缺口、泛泛而谈、没有完全解决问题，或用户可能需要继续追问。\n"
        "- 2 分：不满意；核心需求大多没有满足，回答明显偏题、错误、缺少关键内容或实用性较差。\n"
        "- 1 分：很不满意；回答严重错误、无关、拒答不当，或基本无法使用。\n"
        "- 对 3/4 边界要特别谨慎：若整段对话已经解决主要任务且没有严重错误，优先判为 4；只有存在实质性缺陷时才判为 3 或更低。\n"
        "- 对 4/5 边界不要过度保守：若回答完整、有帮助且明显贴合用户意图，可以给 5；5 分不等于完美无缺。\n\n"
    )


def _memory_confidence(memory: UserMemory) -> tuple[str, str]:
    dist = memory.score_distribution
    counts = [
        int(dist.score_1),
        int(dist.score_2),
        int(dist.score_3),
        int(dist.score_4),
        int(dist.score_5),
    ]
    nonzero = sum(1 for c in counts if c > 0)
    n_sessions = int(memory.n_history_sessions)
    if n_sessions < 3:
        return (
            "low",
            f"history_sessions={n_sessions} < 3，历史证据很薄，memory 只能作为弱参考",
        )
    if nonzero <= 1:
        return (
            "low",
            f"score_buckets={nonzero}，历史分数几乎单一，不能据此断定用户总是宽松或严格",
        )
    if n_sessions < 5 or nonzero == 2:
        return (
            "medium",
            f"history_sessions={n_sessions}, score_buckets={nonzero}，memory 有一定参考价值但不稳定",
        )
    return (
        "high",
        f"history_sessions={n_sessions}, score_buckets={nonzero}，memory 证据相对充分",
    )


def _format_memory_guard_block(memory: UserMemory, prompt_version: UrsPromptVersion) -> str:
    if prompt_version != "urs_v2_memory_guarded":
        return ""
    confidence, reason = _memory_confidence(memory)
    return (
        "【Memory 使用约束】\n"
        f"- 当前 memory 可信度：{confidence}（{reason}）。\n"
        "- memory 只能帮助理解该用户可能的评分风格，不能替代对当前 session 内容质量的判断。\n"
        "- 若当前 session 有直接证据显示回答偏题、错误、拒答、未解决核心需求或明显有帮助，应优先相信当前 session 证据。\n"
        "- 不要把某个历史 intent 的具体要求泛化到无关 intent。例如旅游/家庭活动偏好不能直接用于经济学定义、天气查询或专业题目。\n"
        "- 若 memory 主要来自单一分数桶（全 5、全 4 或全 3），只能说明历史样本不足，不能据此把当前 session 自动推高或压低。\n"
        "- 若 memory 与当前 session 证据冲突，必须在 analysis 中说明冲突，并以当前 session 证据为主。\n\n"
    )


def build_urs_memory_prompt(
    user_id: str,
    profile: dict,
    history_sessions: list,
) -> str:
    """
    URS session-level memory building prompt。

    与 turn-level v2 的核心 schema 保持一致，但对比证据来自“整段对话级别”的满意度标签。
    """
    sessions_to_use = _select_sessions(history_sessions)
    reason_labels = list(get_reason_to_id().keys())

    session_lines: list[str] = []
    for idx, session in enumerate(sessions_to_use):
        score = session.satisfaction_scores[0]
        reason = session.dissatisfaction_reasons[0]
        tag = f"★{score}" + (f"（{reason}）" if score <= 3 else "")
        lines = [
            f"【Session {idx + 1}】任务：{session.task}",
            f"任务背景：{_truncate(session.task_context, 150)}",
            f"整段对话：\n{_format_session_dialogue(session.history, max_chars=220)}",
            f"[整体满意度: {tag}]",
        ]
        session_lines.append("\n".join(lines))

    session_block = "\n\n".join(session_lines)
    sessions_by_score = _collect_sessions_by_score(sessions_to_use)
    grouped_block = "\n\n".join(
        _format_score_group(score, sessions_by_score.get(score, []), max_examples=3)
        for score in [5, 4, 3, 2, 1]
        if sessions_by_score.get(score)
    )

    prompt = (
        "你是一名用户行为分析师。请基于该用户在 URS 数据集上的历史【整段对话级】满意度标签，"
        "总结出一份可直接用于后续 session-level 评分的个性化记忆。\n\n"
        f"【用户 ID】{user_id}\n"
        f"【用户画像】{_format_profile(profile)}\n\n"
        f"【历史 session（按原始顺序）】\n{session_block}\n\n"
        f"【按分数分组的对比证据】\n{grouped_block}\n\n"
        "你需要输出的内容：\n"
        "1. 【评分边界 4→5】：对比 5 分和 4 分 session，指出哪些具体要素决定了能否从 4 分升至 5 分；若缺证据，明确说明证据不足\n"
        "2. 【评分边界 3→4】：对比 4 分和 3 分（及以下）session，指出哪些缺陷会导致从 4 分跌至 3 分；若缺证据，明确说明证据不足\n"
        "3. 【评分风格】：该用户整体打分刻度如何\n"
        "4. 【用户特异性要求】：只保留最能改变评分的 1-4 条要求\n"
        "5. 【偏好格式】：如果只是通用偏好，可简短概括，不必展开\n"
        "6. 【任务观察】：各 intent 下有哪些特殊偏好\n\n"
        f"可参考的不满意原因类别：{', '.join(reason_labels)}\n\n"
        "请严格按照 JSON Schema 输出，不要输出其他内容。"
    )
    return prompt


def build_urs_memory_update_prompt(
    existing_memory: UserMemory,
    new_session,
    turn_predictions: list[dict],
    use_oracle_labels: bool = False,
) -> str:
    """URS session-level 的 memory update prompt。"""
    existing_json = existing_memory.model_dump_json(
        indent=2,
        exclude={"memory_version", "source_tasks", "n_history_sessions", "n_history_turns"},
    )
    pred = turn_predictions[0]
    score = pred.get("gold_score", "?") if use_oracle_labels else pred.get("pred_score", "?")
    reason = pred.get("gold_reason", "?") if use_oracle_labels else pred.get("pred_reason", "?")
    label_note = (
        "本次提供了真实 session-level 标签，可作为可靠证据更新记忆。"
        if use_oracle_labels
        else "本次仅有模型预测的 session-level 标签（可能有误），请谨慎参考，不要因单条弱证据大幅修改已有记忆。"
    )
    session_text = (
        f"任务：{new_session.task}\n"
        f"背景：{_truncate(new_session.task_context, 180)}\n"
        f"整段对话：\n{_format_session_dialogue(new_session.history, max_chars=220)}\n"
        f"[整体标签 ★{score}" + (f"（{reason}）]" if int(score) <= 3 else "]")
    )
    prompt = (
        "你正在维护一份 URS session-level 用户记忆。请根据新观察到的整段对话决定是否需要更新记忆。\n\n"
        f"【现有记忆】\n{existing_json}\n\n"
        f"【新 Session】\n{session_text}\n\n"
        f"【注意】{label_note}\n\n"
        "更新原则（保守优先）：\n"
        "- 若新 session 与已有模式一致，保持记忆不变或仅微调\n"
        "- 仅当新 session 提供了明确反例或补充信息时，才修改 four_vs_five_distinction / three_vs_four_distinction / user_specific_requirements\n"
        "- 更新 avg_satisfaction_score 和 score_distribution 的统计数字\n"
        "- 可新增 task_specific_observations 条目，但不删除已有条目\n\n"
        "请严格按照原 JSON Schema 输出更新后的记忆（不含元信息字段），不要输出其他内容。"
    )
    return prompt


def build_session_eval_prompt(
    memory: UserMemory,
    profile: dict,
    task_context: str,
    session_history: list[dict],
    prompt_version: UrsPromptVersion = "v2",
) -> str:
    """
    URS session-level 评分 prompt（v2 rubric，带 memory）。

    输入
      memory           — 该用户的 UserMemory（基于其他 intent 的历史 session 构建）
      profile          — 用户画像；URS 为空 dict，走 memory.py 里的 fallback
      task_context     — "[intent] title" 形式的主题描述
      session_history  — 整段对话，list[{"role": "user"|"assistant", "content": str}]
    """
    reason_labels = list(get_reason_to_id().keys())
    reason_text = "、".join(reason_labels)
    reason_rule_block = _format_reason_rule_block()
    reason_json_rule = _format_reason_json_rule()
    dialogue_text = _format_session_dialogue(session_history)
    calibration_block = _format_urs_calibration_block(prompt_version)
    memory_guard_block = _format_memory_guard_block(memory, prompt_version)

    user_reqs = "\n".join(
        f"  - {r}" for r in memory.user_specific_requirements
    ) if memory.user_specific_requirements else "  （无特异性要求记录）"

    rubric = (
        f"【该用户的个性化评分标准】\n"
        f"评分风格：{memory.scoring_style}\n"
        f"历史平均分：{memory.avg_satisfaction_score:.2f}  "
        f"（5分×{memory.score_distribution.score_5} / "
        f"4分×{memory.score_distribution.score_4} / "
        f"3分×{memory.score_distribution.score_3} / "
        f"2分×{memory.score_distribution.score_2} / "
        f"1分×{memory.score_distribution.score_1}）\n\n"
        f"▸ 3分以下 → 4分的门槛：{memory.three_vs_four_distinction}\n"
        f"▸ 4分 → 5分的门槛：{memory.four_vs_five_distinction}\n\n"
        f"该用户的特定要求（区别于一般用户）：\n{user_reqs}\n"
        f"偏好回复形式：{memory.preferred_response_format}\n"
        f"\n{memory_guard_block}"
    )

    prompt = (
        "你是一名个性化满意度评估员。\n"
        "请对下面【整段对话】给出一个 1-5 的整数分数，反映用户对本次对话的整体满意度。\n"
        "注意：是对整段对话打 1 个分数，不是逐轮打分。\n\n"
        f"{rubric}\n"
        f"【用户画像】{_format_profile(profile)}\n\n"
        f"【任务背景】{task_context}\n\n"
        f"【完整对话】\n{dialogue_text}\n\n"
        f"{calibration_block}"
        f"【可选原因标签】{reason_text}\n{reason_rule_block}\n"
        "【评分步骤】请严格按以下顺序推理：\n"
        "Step A: 通读整段对话，判断助手整体是否满足该用户的【3分→4分门槛】（即是否达到满意最低线）\n"
        "Step B: 若已达到 4 分，再判断是否进一步满足【4分→5分门槛】\n"
        "Step C: 若未达到 4 分，根据缺陷严重程度决定给 1/2/3 分\n"
        "Step D: 选取一个最贴切的原因标签（若 classification >= 4 必须为【满意】）\n\n"
        "请严格输出 JSON，不要输出其他内容：\n"
        "{\n"
        '  "classification": 1-5 中的整数（对整段对话的单一分数）,\n'
        f'  "reason": "{reason_json_rule}",\n'
        '  "analysis": "按 StepA/StepB/StepC 格式说明判断过程，须明确引用上方评分标准中的具体条件"\n'
        "}\n"
    )
    return prompt


def build_session_eval_prompt_no_memory(
    profile: dict,
    task_context: str,
    session_history: list[dict],
    prompt_version: UrsPromptVersion = "v2",
) -> str:
    """URS session-level 评分 prompt — 无记忆 baseline。"""
    reason_labels = list(get_reason_to_id().keys())
    reason_text = "、".join(reason_labels)
    reason_rule_block = _format_reason_rule_block()
    reason_json_rule = _format_reason_json_rule()
    dialogue_text = _format_session_dialogue(session_history)
    calibration_block = _format_urs_calibration_block(prompt_version)

    prompt = (
        "你是一名会进行细粒度对话质量分析的评估员。\n"
        "请对下面【整段对话】给出一个 1-5 的整数分数，反映用户对本次对话的整体满意度。\n"
        "同时选取一个最贴切的原因标签（分数 >=4 必须为【满意】；分数 <=3 选择一个不满意原因）。\n\n"
        f"【用户画像】{_format_profile(profile)}\n\n"
        f"【任务背景】{task_context}\n\n"
        f"【完整对话】\n{dialogue_text}\n\n"
        f"{calibration_block}"
        f"【可选原因标签】{reason_text}\n{reason_rule_block}\n"
        "请严格输出 JSON，不要输出其他内容：\n"
        "{\n"
        '  "classification": 1-5 中的整数（对整段对话的单一分数）,\n'
        f'  "reason": "{reason_json_rule}",\n'
        '  "analysis": "你的详细推理过程"\n'
        "}\n"
    )
    return prompt
