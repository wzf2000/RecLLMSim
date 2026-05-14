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


def _is_langaware(prompt_version: UrsPromptVersion) -> bool:
    return prompt_version == "urs_v2_calibrated_langaware"


def _is_task_guarded(prompt_version: UrsPromptVersion) -> bool:
    return prompt_version in {
        "urs_v2_calibrated_task_guarded",
        "urs_v2_calibrated_task_guarded_v2",
        "urs_v2_calibrated_task_guarded_memgate",
        "urs_v2_calibrated_task_guarded_evidence_first",
    }


def _is_task_guarded_v2(prompt_version: UrsPromptVersion) -> bool:
    return prompt_version == "urs_v2_calibrated_task_guarded_v2"


def _is_memgate(prompt_version: UrsPromptVersion) -> bool:
    return prompt_version == "urs_v2_calibrated_task_guarded_memgate"


def _is_evidence_first(prompt_version: UrsPromptVersion) -> bool:
    return prompt_version == "urs_v2_calibrated_task_guarded_evidence_first"


def _detect_session_language(task_context: str, session_history: list[dict]) -> str:
    text = task_context + "\n" + "\n".join(str(m.get("content", "")) for m in session_history[:2])
    ascii_chars = sum(1 for ch in text if ord(ch) < 128 and ch.isalpha())
    cjk_chars = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    if ascii_chars > cjk_chars * 2:
        return "en"
    return "zh"


def _detect_task_slug(task_context: str) -> str:
    text = task_context.strip()
    if text.startswith("[") and "]" in text:
        return text[1:text.index("]")].strip()
    return "unknown"


def _format_urs_calibration_block(
    prompt_version: UrsPromptVersion,
    language: str = "zh",
) -> str:
    if prompt_version not in {
        "urs_v2_calibrated",
        "urs_v2_memory_guarded",
        "urs_v2_calibrated_langaware",
        "urs_v2_calibrated_task_guarded",
        "urs_v2_calibrated_task_guarded_v2",
        "urs_v2_calibrated_task_guarded_memgate",
        "urs_v2_calibrated_task_guarded_evidence_first",
    }:
        return ""
    if prompt_version == "urs_v2_calibrated_langaware" and language == "en":
        return (
            "【URS Session-Level Rating Calibration】\n"
            "- This is URS session-level evaluation: rate the user's overall satisfaction with the whole dialogue, not just local imperfections.\n"
            "- 5: Very satisfied. The response is helpful, accurate, well aligned with the user's request, and clearly exceeds basic expectations. It does not need to be perfect.\n"
            "- 4: Satisfied. The main user need is met. Minor omissions, wording issues, or limited depth should still usually be rated 4 rather than 3.\n"
            "- 3: Neutral / average. The response only partially meets the request, leaves a clear gap, is generic, or would likely require follow-up.\n"
            "- 2: Dissatisfied. The response misses much of the core need, is off-topic, incorrect, or has low practical value.\n"
            "- 1: Very dissatisfied. The response is seriously wrong, irrelevant, improperly refuses, or is basically unusable.\n"
            "- Be careful at the 3/4 boundary: if the main task is solved and there is no serious error, prefer 4. Use 3 only when the core answer is missing, clearly wrong, off-topic, or too incomplete to be useful.\n"
            "- Do not be overly conservative at the 4/5 boundary: a complete, helpful, clearly aligned response can receive 5.\n\n"
            "【English URS robustness rules】\n"
            "- Many URS English answers are short or slightly truncated. Do not lower a score below 4 solely because the answer is brief, lacks extra structure, or appears lightly cut off after already giving the core answer.\n"
            "- For factual short-answer or definition questions, prioritize correctness and directness over extra examples, step-by-step guidance, or broad coverage.\n"
            "- Treat personalized memory as weak evidence. Apply memory-derived preferences only when they are directly relevant to the current intent and user request.\n"
            "- Do not import requirements from unrelated intents. For example, travel-planning preferences should not affect an economics definition, and creative-writing preferences should not affect a factual lookup.\n"
            "- Generic preferences such as 'detailed', 'structured', or 'actionable' are not enough to downgrade an otherwise correct and useful answer from 4 to 3.\n\n"
        )
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


def _format_task_guard_block(task_context: str, prompt_version: UrsPromptVersion) -> str:
    task = _detect_task_slug(task_context)
    task_rules = {
        "retrieval": (
            "- 检索/事实类：优先判断答案是否直接、准确地回答核心事实问题。\n"
            "- 简短但正确的定义、路线、概念解释通常可为 4；不要强制要求额外结构。\n"
            "- 若用户要求具体来源、产品、地点、价格、制造地、论文/文件/链接等硬约束，而回答没有给出或无法验证关键约束，不能给 4。\n"
        ),
        "professional": (
            "- 专业问题类：优先判断是否给出正确、可执行、足以解决问题的方案。\n"
            "- 若缺少关键步骤、代码/公式/法律或工程约束，或只给泛泛建议，通常不应给 4。\n"
            "- 若答案方向正确但需要用户继续追问才能执行，通常为 3。\n"
        ),
        "advice": (
            "- 建议类：优先判断建议是否贴合用户处境、具体且可操作。\n"
            "- 轻微不全面仍可为 4；但泛泛鸡汤、没有行动方案、忽视关键限制时不能给 4。\n"
        ),
        "creative": (
            "- 创意类：优先判断是否满足用户指定的主题、人物、格式、语气和关键情节。\n"
            "- 若核心创作内容已完成，轻微不够精致通常不低于 4。\n"
            "- 若漏掉明确指定的关键元素、格式完全不符或内容明显未完成，不能给 4。\n"
        ),
        "text": (
            "- 文本事务类：优先判断是否产出可直接使用的目标文本，并满足格式/语气/长度/对象等约束。\n"
            "- 若只是提供写作建议而没有完成用户要求的文本，通常不能给 4。\n"
        ),
        "leisure": (
            "- 娱乐休闲类：优先判断是否回应用户偏好并提供有用推荐或互动。\n"
            "- 推荐类回答若方向正确但缺少少量细节可为 4；若与偏好冲突或只泛泛应付，不能给 4。\n"
        ),
        "other": (
            "- 其他类：先识别用户的具体目标，再按核心目标是否被满足判断。\n"
        ),
    }
    if _is_task_guarded_v2(prompt_version):
        task_rules["professional"] = (
            "- 专业问题类：优先判断是否给出正确、可执行、足以解决问题的方案。\n"
            "- 若用户只问一个明确概念、步骤或简单专业问题，答案准确回答核心点即可给 4，不要求覆盖所有扩展风险或高级细节。\n"
            "- 只有缺少执行所必需的关键步骤、代码/公式/法律或工程约束，或明显泛泛而谈时，才应跌破 4。\n"
            "- 若答案方向正确但用户仍必须继续追问才能完成主要任务，通常为 3。\n"
        )
        task_rules["advice"] = (
            "- 建议类：优先判断建议是否贴合用户处境、具体且可操作。\n"
            "- 若用户请求本身较简单，答案给出清晰可执行的建议即可给 4，不要求穷尽所有方案。\n"
            "- 轻微不全面、缺少少量扩展建议仍可为 4；但泛泛鸡汤、没有行动方案、忽视关键限制时不能给 4。\n"
        )
        task_rules["text"] = (
            "- 文本事务类：优先判断是否回应了用户的文本处理目标，并满足关键格式/语气/长度/对象约束。\n"
            "- 若用户要求生成完整文本，则应产出可直接使用的目标文本；若用户只是要求改写、解释、建议或局部文本处理，完整满足该局部目标即可给 4。\n"
            "- 不要仅因文本较短、模板化或缺少额外润色就降到 3；只有没有完成关键文本目标、格式明显不符或需要重新追问时，才不能给 4。\n"
        )
    selected = task_rules.get(task, task_rules["other"])
    return (
        "【Task-aware guard（URS 数据集校准）】\n"
        f"当前 intent：{task}\n"
        f"{selected}"
        "\n【DSAT guard：以下情况通常不能给 4 或 5】\n"
        "- 核心请求没有被回答，或回答对象/任务明显错位。\n"
        "- 用户要求具体 artifact/source/product/route/price/constraint，但回答没有提供关键对象或未满足关键约束。\n"
        "- 回答只是泛泛背景、空泛建议、免责声明、找不到但没有有效替代方案。\n"
        "- 回答存在关键事实错误、明显幻觉、不当拒答或敏感话题处理不恰当。\n"
        "- 回答需要用户继续追问才能完成主要任务。\n\n"
        "【Memory 使用约束】\n"
        "- memory 只能作为弱参考，优先相信当前 session 的直接证据。\n"
        "- 只有当 memory 中的偏好与当前 intent 和当前请求直接相关时，才允许影响 3/4 或 4/5 边界。\n"
        "- 不要把其他 intent 的具体偏好泛化到当前任务；不要仅因通用偏好如“详细/结构化/可操作”就把正确有用的短答案降到 3。\n\n"
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
    confidence, reason = _memory_confidence(memory)
    if _is_memgate(prompt_version):
        return (
            "【Memory reliability gate】\n"
            f"- 当前 memory 可信度：{confidence}（{reason}）。\n"
            "- 当前 session 的直接证据永远优先于 memory。\n"
            "- 若当前回答明确满足核心请求且没有触发 DSAT guard，memory 不能把分数从 4/5 拉到 3 或以下。\n"
            "- 若当前回答明确未满足核心请求或触发 DSAT guard，memory 不能把分数从 1/2/3 拉到 4 或以上。\n"
            "- 当 memory 可信度为 low 时，memory 不允许改变 SAT/DSAT 边界，只能帮助区分 4/5 或 1/2/3 内部等级。\n"
            "- 当 memory 可信度为 medium 时，只有与当前 intent 和当前请求直接相关的偏好才可轻微影响 3/4 边界。\n"
            "- 当 memory 可信度为 high 时，memory 可用于调整边界，但仍必须服从当前 session 证据和 task-aware guard。\n"
            "- 若 memory 中的 task_specific_observations 来自其他 intent，或只是“详细/结构化/可操作”等通用偏好，不能单独改变 3/4 判断。\n\n"
        )
    if _is_evidence_first(prompt_version):
        return (
            "【Memory evidence policy】\n"
            f"- 当前 memory 可信度：{confidence}（{reason}）。\n"
            "- 先完全基于当前 session 和 task-aware guard 得到 provisional score。\n"
            "- memory 是弱参考，不是评分 rubric；不要把历史 3/4 或 4/5 边界当成硬规则套用到当前 session。\n"
            "- 默认情况下，memory 只能在同一满意侧内部微调：4↔5 或 1↔2↔3。\n"
            "- 只有当 memory 中存在与当前 intent、当前请求、当前缺陷直接匹配的具体证据时，才允许影响 3/4 边界。\n"
            "- 若 memory 只是通用偏好（详细、结构化、可操作）或来自其他 intent，不允许改变 provisional SAT/DSAT 判断。\n"
            "- 若当前 session 证据与 memory 冲突，必须以当前 session 证据为准。\n\n"
        )
    if prompt_version != "urs_v2_memory_guarded":
        return ""
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
    language = _detect_session_language(task_context, session_history)
    calibration_block = _format_urs_calibration_block(prompt_version, language=language)
    memory_guard_block = _format_memory_guard_block(memory, prompt_version)
    task_guard_block = (
        _format_task_guard_block(task_context, prompt_version)
        if _is_task_guarded(prompt_version) else ""
    )

    user_reqs = "\n".join(
        f"  - {r}" for r in memory.user_specific_requirements
    ) if memory.user_specific_requirements else "  （无特异性要求记录）"

    rubric = (
        f"{'【该用户的历史记忆（弱参考，不是评分标准）】' if _is_evidence_first(prompt_version) else '【该用户的个性化评分标准】'}\n"
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
    if _is_task_guarded(prompt_version):
        if _is_memgate(prompt_version):
            rating_steps = (
                "【评分步骤】请严格按以下顺序推理：\n"
                "Step A: 识别当前 session 的核心用户请求，以及是否存在具体硬约束（来源/产品/价格/路线/格式/代码/关键元素等）\n"
                "Step B: 只基于当前 session 证据和 task-aware guard，先给出一个 preliminary SAT/DSAT 判断；若未满足核心请求或触发 DSAT guard，preliminary 必须为 DSAT\n"
                "Step C: 判断 memory 可信度和相关性；说明 memory 是否允许改变 3/4 边界\n"
                "Step D: 只有当 memory reliability gate 允许时，才可用 memory 调整 preliminary SAT/DSAT；否则只能在同一侧内部微调分数\n"
                "Step E: 若最终达到 4 分，再判断是否明显超过基本要求而可给 5；5 分不要求完美\n"
                "Step F: 若最终未达到 4 分，根据缺陷严重程度决定 1/2/3 分\n"
                "Step G: 选取一个最贴切的原因标签（若 classification >= 4 必须为【满意】）\n\n"
            )
        elif _is_evidence_first(prompt_version):
            rating_steps = (
                "【评分步骤】请严格按以下顺序推理：\n"
                "Step A: 识别当前 session 的核心用户请求，以及是否存在具体硬约束（来源/产品/价格/路线/格式/代码/关键元素等）\n"
                "Step B: 只基于当前 session 证据和 task-aware guard 判断核心请求是否被正确且有用地满足，并给出 provisional score；此步不要使用 memory\n"
                "Step C: 检查是否触发 DSAT guard；若触发，provisional score 必须为 1/2/3，不能给 4/5\n"
                "Step D: 再查看 memory 是否有与当前 intent 和当前请求直接匹配的具体证据；若没有，保持 provisional score 或只做同侧微调\n"
                "Step E: 若 memory 证据直接相关，只允许做小幅调整；跨越 3/4 边界必须同时有当前 session 证据支持，不能只凭 memory\n"
                "Step F: 若最终达到 4 分，再判断是否明显超过基本要求而可给 5；5 分不要求完美\n"
                "Step G: 选取一个最贴切的原因标签（若 classification >= 4 必须为【满意】）\n\n"
            )
        else:
            rating_steps = (
                "【评分步骤】请严格按以下顺序推理：\n"
                "Step A: 识别当前 session 的核心用户请求，以及是否存在具体硬约束（来源/产品/价格/路线/格式/代码/关键元素等）\n"
                "Step B: 先基于当前 session 证据判断核心请求是否被正确且有用地满足；若未满足或触发 DSAT guard，不能给 4/5\n"
                "Step C: 再参考 task-aware guard 判断当前 intent 下什么缺陷足以跌破 4 分\n"
                "Step D: memory 只作为弱参考；若 memory 与当前 session 证据冲突，优先当前 session\n"
                "Step E: 若达到 4 分，再判断是否明显超过基本要求而可给 5；5 分不要求完美\n"
                "Step F: 若未达到 4 分，根据缺陷严重程度决定 1/2/3 分\n"
                "Step G: 选取一个最贴切的原因标签（若 classification >= 4 必须为【满意】）\n\n"
            )
        analysis_rule = (
            '  "analysis": "按 StepA-StepG 格式说明判断过程，须明确引用当前 session 证据、'
            'task-aware guard 和必要的 memory 证据；若使用 evidence-first 版本，须写出 provisional score 是否被 memory 调整"\n'
        )
    else:
        rating_steps = (
            "【评分步骤】请严格按以下顺序推理：\n"
            "Step A: 通读整段对话，判断助手整体是否满足该用户的【3分→4分门槛】（即是否达到满意最低线）\n"
            "Step B: 若已达到 4 分，再判断是否进一步满足【4分→5分门槛】\n"
            "Step C: 若未达到 4 分，根据缺陷严重程度决定给 1/2/3 分\n"
            "Step D: 选取一个最贴切的原因标签（若 classification >= 4 必须为【满意】）\n\n"
        )
        analysis_rule = (
            '  "analysis": "按 StepA/StepB/StepC 格式说明判断过程，须明确引用上方评分标准中的具体条件"\n'
        )

    if _is_langaware(prompt_version) and language == "en":
        prompt = (
            "You are a personalized satisfaction evaluator.\n"
            "Rate the user's overall satisfaction with the following complete dialogue using one integer score from 1 to 5.\n"
            "Important: assign one session-level score for the whole dialogue, not per-turn scores.\n\n"
            f"{rubric}\n"
            f"【User profile】{_format_profile(profile)}\n\n"
            f"【Task context】{task_context}\n\n"
            f"【Complete dialogue】\n{dialogue_text}\n\n"
            f"{calibration_block}"
            f"【Allowed reason labels】{reason_text}\n{reason_rule_block}\n"
            "【Rating steps】Follow this order:\n"
            "Step A: Identify the current session's main user request and judge whether the assistant directly addresses that core request.\n"
            "Step B: Decide the 3/4 boundary mainly from current-session evidence. If the core request is answered correctly and usefully, assign at least 4 unless there is a serious flaw.\n"
            "Step C: Use personalized memory only as a secondary adjustment for directly relevant preferences. Ignore unrelated or generic memory requirements.\n"
            "Step D: If the answer reaches 4, decide whether it further satisfies the 4-to-5 threshold. Do not require perfection for 5.\n"
            "Step E: If it does not reach 4, assign 1/2/3 based on flaw severity.\n"
            "Step F: Choose the reason label. If classification >= 4, reason must be 满意.\n\n"
            "Output strict JSON only:\n"
            "{\n"
            '  "classification": integer from 1 to 5,\n'
            f'  "reason": "{reason_json_rule}",\n'
            '  "analysis": "Explain using StepA-StepF and cite concrete current-session evidence. Mention memory only when directly relevant."\n'
            "}\n"
        )
        return prompt

    prompt = (
        "你是一名个性化满意度评估员。\n"
        "请对下面【整段对话】给出一个 1-5 的整数分数，反映用户对本次对话的整体满意度。\n"
        "注意：是对整段对话打 1 个分数，不是逐轮打分。\n\n"
        f"{rubric}\n"
        f"【用户画像】{_format_profile(profile)}\n\n"
        f"【任务背景】{task_context}\n\n"
        f"【完整对话】\n{dialogue_text}\n\n"
        f"{calibration_block}"
        f"{task_guard_block}"
        f"【可选原因标签】{reason_text}\n{reason_rule_block}\n"
        f"{rating_steps}"
        "请严格输出 JSON，不要输出其他内容：\n"
        "{\n"
        '  "classification": 1-5 中的整数（对整段对话的单一分数）,\n'
        f'  "reason": "{reason_json_rule}",\n'
        f"{analysis_rule}"
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
    language = _detect_session_language(task_context, session_history)
    calibration_block = _format_urs_calibration_block(prompt_version, language=language)
    task_guard_block = (
        _format_task_guard_block(task_context, prompt_version)
        if _is_task_guarded(prompt_version) else ""
    )
    if _is_task_guarded(prompt_version):
        rating_steps = (
            "【评分步骤】请严格按以下顺序推理：\n"
            "Step A: 识别当前 session 的核心用户请求，以及是否存在具体硬约束\n"
            "Step B: 若核心请求被正确且有用地满足，通常至少给 4；但若触发 DSAT guard，不能给 4/5\n"
            "Step C: 结合 task-aware guard 判断当前 intent 下的缺陷严重程度\n"
            "Step D: 若达到 4 分，再判断是否明显超过基本要求而可给 5；5 分不要求完美\n"
            "Step E: 若未达到 4 分，根据缺陷严重程度决定 1/2/3 分\n\n"
        )
        analysis_rule = (
            '  "analysis": "按 StepA-StepE 格式说明判断过程，须明确引用当前 session 证据和 task-aware guard"\n'
        )
    else:
        rating_steps = ""
        analysis_rule = '  "analysis": "你的详细推理过程"\n'

    if _is_langaware(prompt_version) and language == "en":
        prompt = (
            "You are a fine-grained dialogue quality evaluator.\n"
            "Rate the user's overall satisfaction with the following complete dialogue using one integer score from 1 to 5.\n"
            "Also choose the most appropriate reason label. If the score is >=4, the reason must be 满意; if <=3, choose a dissatisfied reason.\n\n"
            f"【User profile】{_format_profile(profile)}\n\n"
            f"【Task context】{task_context}\n\n"
            f"【Complete dialogue】\n{dialogue_text}\n\n"
            f"{calibration_block}"
            f"【Allowed reason labels】{reason_text}\n{reason_rule_block}\n"
            "【Rating steps】Follow this order:\n"
            "Step A: Identify the current session's main user request.\n"
            "Step B: If the assistant answers the core request correctly and usefully, assign at least 4 unless there is a serious flaw.\n"
            "Step C: For short factual answers, do not require extra structure or examples unless the user explicitly asked for them.\n"
            "Step D: If the answer reaches 4, decide whether it deserves 5. Do not require perfection for 5.\n"
            "Step E: If it does not reach 4, assign 1/2/3 based on flaw severity.\n\n"
            "Output strict JSON only:\n"
            "{\n"
            '  "classification": integer from 1 to 5,\n'
            f'  "reason": "{reason_json_rule}",\n'
            '  "analysis": "Explain using StepA-StepE and cite concrete current-session evidence."\n'
            "}\n"
        )
        return prompt

    prompt = (
        "你是一名会进行细粒度对话质量分析的评估员。\n"
        "请对下面【整段对话】给出一个 1-5 的整数分数，反映用户对本次对话的整体满意度。\n"
        "同时选取一个最贴切的原因标签（分数 >=4 必须为【满意】；分数 <=3 选择一个不满意原因）。\n\n"
        f"【用户画像】{_format_profile(profile)}\n\n"
        f"【任务背景】{task_context}\n\n"
        f"【完整对话】\n{dialogue_text}\n\n"
        f"{calibration_block}"
        f"{task_guard_block}"
        f"【可选原因标签】{reason_text}\n{reason_rule_block}\n"
        f"{rating_steps}"
        "请严格输出 JSON，不要输出其他内容：\n"
        "{\n"
        '  "classification": 1-5 中的整数（对整段对话的单一分数）,\n'
        f'  "reason": "{reason_json_rule}",\n'
        f"{analysis_rule}"
        "}\n"
    )
    return prompt
