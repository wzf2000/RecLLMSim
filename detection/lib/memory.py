"""
用户记忆模块（User Memory） v2

改进要点（相较 v1）：
  - Schema 用对比式评分边界替代泛化满意/不满意模式列表：
      four_vs_five_distinction  — 该用户4分和5分的具体区别
      three_vs_four_distinction — 该用户3分及以下和4分的具体区别
      scoring_style             — 严格/宽松/中等，附校准说明
      user_specific_requirements— 区别于一般用户的特定要求（不允许泛化描述）
  - Memory building prompt 新增按分数分组的对比证据区，迫使 LLM 分析相邻分数差异
  - Eval prompt 改为逐步判断的个性化评分 rubric，而非泛化的"参考以下模式"

主要组件：
  ScoreDistribution      — 满意度分布（固定 5 字段，兼容 OpenAI structured output）
  TaskObservation        — 单个任务类型的观察
  UserMemoryContent      — LLM 生成部分（严格兼容 OpenAI structured output）
  UserMemory             — 完整记忆 = UserMemoryContent + 程序侧元信息
  build_memory_prompt    — 对比式 memory building prompt
  build_memory_update_prompt — memory 更新 prompt
  build_turn_eval_prompt     — rubric 式 turn 评估 prompt
  build_turn_eval_prompt_no_memory — 无记忆 baseline prompt
"""

from __future__ import annotations

from collections import defaultdict
from typing import Literal

from pydantic import BaseModel, Field

from .personalized_data import SessionData
from .satisfaction_constants import (
    SATISFIED_REASON,
    get_dissatisfied_reasons,
    get_reason_to_id,
)

# ──────────────────────────────────────────────────────────────────────────────
# 辅助子模型（OpenAI structured output 兼容：无 dict，所有字段必填）
# ──────────────────────────────────────────────────────────────────────────────

class ScoreDistribution(BaseModel):
    """满意度 1-5 分的出现次数（固定字段）。"""
    score_1: int = Field(description="满意度为 1 分的 assistant 轮数")
    score_2: int = Field(description="满意度为 2 分的 assistant 轮数")
    score_3: int = Field(description="满意度为 3 分的 assistant 轮数")
    score_4: int = Field(description="满意度为 4 分的 assistant 轮数")
    score_5: int = Field(description="满意度为 5 分的 assistant 轮数")

    def to_dict(self) -> dict[str, int]:
        return {
            "1": self.score_1, "2": self.score_2, "3": self.score_3,
            "4": self.score_4, "5": self.score_5,
        }


class TaskObservation(BaseModel):
    """针对单个任务类型的关键观察。"""
    task_name: str = Field(description="任务类型名称，如旅行规划")
    observation: str = Field(description="该任务下用户特有的偏好或敏感点")


# ──────────────────────────────────────────────────────────────────────────────
# LLM 生成模型（v2：对比式评分边界）
# ──────────────────────────────────────────────────────────────────────────────

class UserMemoryContent(BaseModel):
    """
    LLM 生成的用户记忆（v2）。
    核心改进：用对比式评分边界替代泛化模式列表，使记忆可直接作为评分 rubric。
    """

    # ── 统计基准 ──────────────────────────────────────────────────────────────
    avg_satisfaction_score: float = Field(
        ge=1.0, le=5.0,
        description="历史 assistant 轮的平均满意度分数（用于校准绝对分值）",
    )
    score_distribution: ScoreDistribution = Field(
        description="满意度 1-5 分各自的出现次数",
    )
    scoring_style: str = Field(
        description=(
            "该用户的评分风格及校准说明，1-2 句。"
            "须说明其严格/宽松程度及含义，例如："
            "【偏严格：平均分 3.8，给出 5 分的门槛很高，需要回复完全命中需求且格式完美】"
            "或【偏宽松：平均分 4.5，只要回复无明显缺陷即可得 5 分，3 分表示有实质性问题】"
        ),
    )

    # ── 对比式评分边界（核心字段）────────────────────────────────────────────
    four_vs_five_distinction: str = Field(
        description=(
            "该用户 4 分和 5 分的具体区别，1-3 句。"
            "须基于历史数据中实际出现的 4 分和 5 分轮次的差异，"
            "指出哪些具体要素的有无决定了能否从 4 分升至 5 分。"
            "示例：【5 分要求提供可直接执行的具体步骤和真实资源链接；"
            "4 分时回复正确但缺乏上述细节，或某一环节不够完整】"
        ),
    )
    three_vs_four_distinction: str = Field(
        description=(
            "该用户 3 分及以下和 4 分的具体区别，1-3 句。"
            "须指出哪些缺陷会导致从 4 分跌至 3 分或更低。"
            "示例：【达到 4 分要求回复直接回答用户问题且无明显错误；"
            "3 分及以下出现在回复内容笼统无实质帮助、或忽略了用户的明确约束条件】"
        ),
    )

    # ── 用户特异性要求（禁止泛化描述）───────────────────────────────────────
    user_specific_requirements: list[str] = Field(
        description=(
            "该用户区别于一般用户的特定要求，1-5 条。"
            "每条必须是该用户独有的、可操作的要求，"
            "禁止使用【回复要详细】【要具体】等对任何用户都适用的泛化描述。"
            "好的示例：【要求提供可购买的具体品牌和价格区间，而非泛泛推荐品类】"
            "【要求按周次拆分学习计划，不接受按月粒度的规划】"
        ),
    )

    # ── 沟通偏好 ──────────────────────────────────────────────────────────────
    preferred_response_format: str = Field(
        description="用户偏好的回复组织形式（格式、结构），尽量具体",
    )

    # ── 任务特定观察 ──────────────────────────────────────────────────────────
    task_specific_observations: list[TaskObservation] = Field(
        description=(
            "针对各历史任务类型的关键观察，每个有记录的任务一条（0-4 条）。"
            "observation 须说明该任务场景下用户的特殊偏好或敏感点"
        ),
    )


# ──────────────────────────────────────────────────────────────────────────────
# 完整 UserMemory（UserMemoryContent + 程序侧元信息）
# ──────────────────────────────────────────────────────────────────────────────

class UserMemory(UserMemoryContent):
    """完整用户记忆 = LLM 生成内容 + 程序侧元信息。"""
    memory_version: str = Field(default="v2")
    source_tasks: list[str] = Field(default_factory=list)
    n_history_sessions: int = Field(default=0)
    n_history_turns: int = Field(default=0)

    @classmethod
    def from_content(
        cls,
        content: UserMemoryContent,
        source_tasks: list[str] | None = None,
        n_history_sessions: int = 0,
        n_history_turns: int = 0,
    ) -> "UserMemory":
        data = content.model_dump()
        data["memory_version"] = "v2"
        data["source_tasks"] = list(source_tasks or [])
        data["n_history_sessions"] = n_history_sessions
        data["n_history_turns"] = n_history_turns
        return cls(**data)


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


# ──────────────────────────────────────────────────────────────────────────────
# Memory Update Prompt（v2：保守更新）
# ──────────────────────────────────────────────────────────────────────────────

def build_memory_update_prompt(
    existing_memory: UserMemory,
    new_session: SessionData,
    turn_predictions: list[dict],
    use_oracle_labels: bool = False,
) -> str:
    """
    构造 memory update prompt（v2）。

    策略：保守更新——仅在新 session 提供了与已有记忆明显矛盾或补充的证据时才修改，
    避免预测误差噪声污染已有记忆。
    """
    existing_json = existing_memory.model_dump_json(
        indent=2,
        exclude={"memory_version", "source_tasks", "n_history_sessions", "n_history_turns"},
    )

    session_lines = [
        f"任务：{new_session.task}  背景：{_truncate(new_session.task_context, 150)}",
        "逐轮信息：",
    ]
    assistant_idx = 0
    last_user = ""
    for utt in new_session.history:
        if utt["role"] == "user":
            last_user = _truncate(utt["content"], 120)
        elif utt["role"] == "assistant" and assistant_idx < len(turn_predictions):
            pred = turn_predictions[assistant_idx]
            reply = _truncate(utt["content"])
            pred_s = pred.get("pred_score", "?")
            line = f"  用户：{last_user}\n  助手：{reply}"
            if use_oracle_labels:
                gold_s = pred.get("gold_score", "?")
                gold_r = pred.get("gold_reason", "?")
                line += f"\n  [真实 ★{gold_s}（{gold_r}）]"
            else:
                line += f"\n  [预测 ★{pred_s}]"
            session_lines.append(line)
            assistant_idx += 1

    session_text = "\n".join(session_lines)
    label_note = (
        "本次提供了真实标签，可作为可靠证据更新记忆。"
        if use_oracle_labels
        else "本次仅有模型预测分数（可能有误），请谨慎参考，不要因预测误差大幅修改已有记忆。"
    )

    prompt = (
        "你正在维护一份用户记忆。请根据新观察到的 session 决定是否需要更新记忆。\n\n"
        f"【现有记忆】\n{existing_json}\n\n"
        f"【新 Session】\n{session_text}\n\n"
        f"【注意】{label_note}\n\n"
        "更新原则（保守优先）：\n"
        "- 若新 session 与已有模式一致，保持记忆不变或仅微调\n"
        "- 仅当新 session 提供了明确的反例或补充信息时，才修改 four_vs_five_distinction / "
        "three_vs_four_distinction / user_specific_requirements\n"
        "- 更新 avg_satisfaction_score 和 score_distribution 的统计数字\n"
        "- 可新增 task_specific_observations 条目，但不删除已有条目\n\n"
        "请严格按照原 JSON Schema 输出更新后的记忆（不含元信息字段），不要输出其他内容。"
    )
    return prompt


# ──────────────────────────────────────────────────────────────────────────────
# Turn Evaluation Prompt（v2：rubric 式逐步判断）
# ──────────────────────────────────────────────────────────────────────────────

def _format_anchor_turns(anchor_turns: list) -> str:
    """
    将检索到的 anchor turns 格式化为 prompt 中的"参考案例"块（rank-match 模式）。

    注意：anchor 的使用方式是 rank-matching / nearest-neighbor —— 让模型找到
    当前回复在过往案例中"整体质量最接近"的一条，直接对齐其分数。切忌让模型
    把 anchor 当"高分标杆"然后挑现在回复的毛病（会导致系统性压低预测）。
    """
    if not anchor_turns:
        return ""
    lines = [
        "═══ 该用户历史上的参考案例（真实标注分数） ═══",
        "用法：这些是从该用户过往 session 中检索到的、与当前回复文本最相似的若干轮次。"
        "请把它们按分数排列当作【已校准的参考刻度】，将当前回复在整体质量维度上与其对齐——"
        "若当前回复和某个案例的整体质量处于同一档位，就直接给相同的分数。"
        "不要把这些案例当作【完美标杆】去挑当前回复的毛病。",
        "",
    ]
    for i, a in enumerate(anchor_turns, 1):
        tag = f"★{a.score}" + (f"（{a.reason}）" if a.score <= 3 else "")
        user_snip = _truncate(a.user_msg, 120)
        reply_snip = _truncate(a.assistant_reply, _MAX_REPLY_CHARS)
        lines.append(f"[案例 {i}] 任务：{a.task}  真实满意度：{tag}")
        lines.append(f"  用户提问：{user_snip}")
        lines.append(f"  助手回复：{reply_snip}")
        lines.append("")
    return "\n".join(lines)


def build_turn_eval_prompt(
    memory: UserMemory,
    profile: dict,
    task_context: str,
    history_window: list[str],
    assistant_reply: str,
    anchor_turns: list | None = None,
    prompt_version: Literal[
        "v2", "qwen_short", "boundary_34", "boundary_34_refute", "boundary_34_refute_v2",
        "boundary_34_selective_refute", "boundary_34_selective_refute_v2",
        "boundary_34_selective_refute_v3", "boundary_34_selective_refute_v4",
    ] = "v2",
) -> str:
    """
    构造单轮满意度预测 prompt（v2）。

    核心改进：将 memory 转化为逐步判断的个性化评分 rubric，
    而非泛化的"参考以下模式"。评分逻辑显式分三步：
      Step 1: 是否达到 4 分门槛（three_vs_four_distinction）
      Step 2: 若达到，是否进一步达到 5 分（four_vs_five_distinction）
      Step 3: 若未达到 4 分，根据缺陷程度判断 1/2/3 分

    若提供 anchor_turns（list[AnchorTurn]），会在 rubric 之后插入"参考案例"块，
    作为 few-shot in-context 锚点。

    prompt_version:
      - "v2": 保持原有 rubric prompt，不改历史实验行为
      - "qwen_short": 面向 Qwen3-8B 的更短、更硬的 checklist prompt
      - "boundary_34": 仅围绕 3/4 满意边界判断，输出限制为 3 或 4
      - "boundary_34_refute": 在 3/4 边界上先做反证检查，抑制默认判 4
      - "boundary_34_refute_v2": 更温和的 refute 版本，仅在存在明确致命缺陷时判 3
      - "boundary_34_selective_refute": 第一遍温和判 3/4，并显式标记是否需要二次反证复核
      - "boundary_34_selective_refute_v2": selective 的收紧版本，只在高不确定边界样本上触发二判
      - "boundary_34_selective_refute_v3": 仅优化 first-pass 的边界措辞，gate 和二判保持 v2
      - "boundary_34_selective_refute_v4": 平衡 first-pass，强制同时考虑最强的 3/4 证据
    """
    reason_labels = list(get_reason_to_id().keys())
    reason_text = "、".join(reason_labels)
    reason_rule_block = _format_reason_rule_block()
    reason_json_rule = _format_reason_json_rule()
    history_text = "\n".join(history_window) if history_window else "（无历史）"

    # 组装 task 特定观察（若有当前任务的记录则优先展示）
    task_obs_lines = ""
    if memory.task_specific_observations:
        relevant = [o for o in memory.task_specific_observations
                    if task_context and o.task_name in task_context[:50]]
        others   = [o for o in memory.task_specific_observations
                    if o not in relevant]
        ordered  = relevant + others
        task_obs_lines = "\n".join(
            f"  {o.task_name}：{o.observation}" for o in ordered
        )

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
    )
    if task_obs_lines:
        rubric += f"任务特定观察：\n{task_obs_lines}\n"

    anchor_block = _format_anchor_turns(anchor_turns or [])
    anchor_section = (anchor_block + "\n") if anchor_block else ""

    if prompt_version == "boundary_34_selective_refute_v4":
        anchor_instruction = (
            "【参考案例使用规则】\n"
            "1. 只把案例当作 3/4 边界参考：真实分数 <=3 是【未达满意线案例】，>=4 是【达到满意线案例】。\n"
            "2. 不要只看一边的案例。若当前回复更像未达满意线案例，要敢于判 `3`；若更像达到满意线案例，也不要因不够优秀就压成 `3`。\n"
            "3. 案例用于校准边界，不用于追求 5 分标准。\n\n"
            if anchor_turns else ""
        )
        prompt = (
            "你是一名个性化满意度边界评估员。\n"
            "这是 selective-refute v4 的第一遍初判：目标是尽可能平衡地判断当前助手回复是否达到该用户的【满意最低线】。\n"
            "输出只能是：\n"
            "- `4` = 满意（达到最低满意线）\n"
            "- `3` = 不满意（未达到最低满意线）\n\n"
            "本题不要默认保护 `4`，也不要默认压成 `3`。你必须同时考虑：\n"
            "- 最强的“为什么它应该是 `3`”的证据\n"
            "- 最强的“为什么它至少已经到 `4`”的证据\n"
            "再决定哪一边更强。\n\n"
            f"【用户评分摘要】\n"
            f"评分风格：{memory.scoring_style}\n"
            f"历史平均分：{memory.avg_satisfaction_score:.2f}\n"
            f"满意最低线（3→4 边界）：{memory.three_vs_four_distinction}\n"
            f"更高要求（4→5，仅供背景参考）：{memory.four_vs_five_distinction}\n"
            f"用户特定要求：\n{user_reqs}\n"
            f"偏好回复形式：{memory.preferred_response_format}\n"
            + (f"任务特定观察：\n{task_obs_lines}\n" if task_obs_lines else "")
            + "\n"
            + f"{anchor_section}"
            + anchor_instruction
            + f"【用户画像】{_format_profile(profile)}\n\n"
            + f"【任务背景】{task_context}\n\n"
            + f"【最近对话历史】\n{history_text}\n\n"
            + f"【待评估的助手回复】\n{assistant_reply}\n\n"
            + f"【可选原因标签】{reason_text}\n{reason_rule_block}\n"
            + "【第一遍平衡边界判断规则】\n"
            + "Step 1. 先写出最强的 `3` 证据：\n"
            + "  - 核心问题是否未被回答？\n"
            + "  - 关键约束 / 关键任务目标 / 用户特别在意的要求是否被漏掉？\n"
            + "  - 缺口是否已经明显影响可用性，导致用户仍会觉得没被满足？\n"
            + "Step 2. 再写出最强的 `4` 证据：\n"
            + "  - 核心问题是否已经被回答？\n"
            + "  - 关键要求是否已经基本满足？\n"
            + "  - 剩余问题是否只是普通缺口，而不阻止用户把它当作基本满意的回复？\n"
            + "Step 3. 明确比较这两边哪一边更强：\n"
            + "  - 若最强的 `3` 证据更强，判 `3`\n"
            + "  - 若最强的 `4` 证据更强，判 `4`\n"
            + "Step 4. 只有当两边最强证据真的势均力敌时，才允许 `needs_refute_review=true`。\n"
            + "  - 明显偏向任一边时必须输出 `needs_refute_review=false`\n\n"
            + "注意：\n"
            + "- `不够细致` 既可能只是普通缺口，也可能已经影响可用性；不要默认把它归到任何一边。\n"
            + "- 友好语气、表面帮助性不能替代“核心问题是否真正回答”。\n"
            + "- 不要因为不是 5 分水平就压成 `3`，也不要因为看起来有帮助就放成 `4`。\n"
            + "- `classification` 只能输出 `3` 或 `4`。\n"
            + "- `analysis` 只需 1-2 句，必须同时提到：最强的 `3` 证据、最强的 `4` 证据，以及最终哪一边更强。\n\n"
            + "请严格输出 JSON，不要输出其他内容：\n"
            + "{\n"
            + '  "classification": 只能是 3 或 4,\n'
            + f'  "reason": "{reason_json_rule}",\n'
            + '  "analysis": "用 1-2 句写明：最强的 3 证据是什么，最强的 4 证据是什么，最终哪一边更强，以及是否需要复核",\n'
            + '  "needs_refute_review": true 或 false\n'
            + "}\n"
        )
        return prompt

    if prompt_version == "boundary_34_selective_refute_v3":
        anchor_instruction = (
            "【参考案例使用规则】\n"
            "1. 只把案例当作 3/4 边界参考：真实分数 <=3 是【未达满意线案例】，>=4 是【达到满意线案例】。\n"
            "2. 优先比较当前回复是否已经达到“这个用户愿意认为它基本有用、基本满意”的最低线，而不是和优秀案例比完整度。\n"
            "3. 若当前回复已经明显站在某一边，不要因为它不够优秀就把它拖回边界附近。\n\n"
            if anchor_turns else ""
        )
        prompt = (
            "你是一名个性化满意度边界评估员。\n"
            "这是 selective-refute v3 的第一遍初判：先尽可能准确地判断当前助手回复是否达到该用户的【满意最低线】。\n"
            "输出只能是：\n"
            "- `4` = 满意（达到最低满意线）\n"
            "- `3` = 不满意（未达到最低满意线）\n\n"
            "这一步最重要的不是区分“优秀”和“一般”，而是区分：\n"
            "- 只是普通缺口、还不够细，但已经达到最低满意线\n"
            "- 真正没过满意线，用户仍会觉得不满意\n\n"
            f"【用户评分摘要】\n"
            f"评分风格：{memory.scoring_style}\n"
            f"历史平均分：{memory.avg_satisfaction_score:.2f}\n"
            f"满意最低线（3→4 边界）：{memory.three_vs_four_distinction}\n"
            f"更高要求（4→5，仅供背景参考）：{memory.four_vs_five_distinction}\n"
            f"用户特定要求：\n{user_reqs}\n"
            f"偏好回复形式：{memory.preferred_response_format}\n"
            + (f"任务特定观察：\n{task_obs_lines}\n" if task_obs_lines else "")
            + "\n"
            + f"{anchor_section}"
            + anchor_instruction
            + f"【用户画像】{_format_profile(profile)}\n\n"
            + f"【任务背景】{task_context}\n\n"
            + f"【最近对话历史】\n{history_text}\n\n"
            + f"【待评估的助手回复】\n{assistant_reply}\n\n"
            + f"【可选原因标签】{reason_text}\n{reason_rule_block}\n"
            + "【第一遍边界判断规则】\n"
            + "Step 1. 先判断：回复是否真正回答了用户此刻最核心的问题。\n"
            + "  - 若核心问题没有被回答，优先判 `3`。\n"
            + "Step 2. 再判断：关键约束、关键任务目标、该用户特别在意的要求，是否至少被基本满足。\n"
            + "  - 若关键要求被漏掉，且这会明显影响可用性，优先判 `3`。\n"
            + "Step 3. 只有在核心问题已回答、关键要求也基本满足时，才去看剩余缺口属于哪类：\n"
            + "  - 【普通缺口】= 细节不足、还可更完整、还可更个性化，但不妨碍用户把它当作基本满意的答复，此时应判 `4`\n"
            + "  - 【关键缺口】= 缺失会让用户仍觉得没被满足、没法直接用、或明显偏离要求，此时应判 `3`\n"
            + "Step 4. 只有当你真的无法判断某个唯一可疑点到底是普通缺口还是关键缺口时，才允许 `needs_refute_review=true`。\n"
            + "  - 明显满意或明显不满意都必须输出 `needs_refute_review=false`\n"
            + "  - `needs_refute_review=true` 必须是少数情况\n\n"
            + "注意：\n"
            + "- `不够细致` 默认更接近【普通缺口】，除非它已经严重到让回复不可用或明显没满足核心要求。\n"
            + "- 不要因为它不是 5 分水平，就把一个本来已经过线的回复判成 3。\n"
            + "- 也不要因为回复语气友好、表面在帮忙，就把一个没回答核心问题的回复判成 4。\n"
            + "- `classification` 只能输出 `3` 或 `4`。\n"
            + "- `analysis` 只需 1-2 句，明确写出：核心问题是否被回答；关键要求是否被满足；当前可疑点为何属于普通缺口或关键缺口。\n\n"
            + "请严格输出 JSON，不要输出其他内容：\n"
            + "{\n"
            + '  "classification": 只能是 3 或 4,\n'
            + f'  "reason": "{reason_json_rule}",\n'
            + '  "analysis": "用 1-2 句写明：核心问题是否被回答，关键要求是否被满足，当前可疑点为何属于普通缺口或关键缺口，以及是否需要复核",\n'
            + '  "needs_refute_review": true 或 false\n'
            + "}\n"
        )
        return prompt

    if prompt_version == "boundary_34_selective_refute_v2":
        anchor_instruction = (
            "【参考案例使用规则】\n"
            "1. 只把案例理解为边界参考：真实分数 <=3 是【未达满意线案例】，>=4 是【达到满意线案例】。\n"
            "2. 只有当当前回复与两类案例都存在明显相似点、边界仍拿不准时，才考虑触发复核。\n"
            "3. 若当前回复整体明显站在某一边，就不要触发复核。\n\n"
            if anchor_turns else ""
        )
        prompt = (
            "你是一名个性化满意度边界评估员。\n"
            "这是 selective-refute v2 的第一遍初判：先温和判断当前助手回复是否达到该用户的【满意最低线】。\n"
            "输出只能是：\n"
            "- `4` = 满意（达到最低满意线）\n"
            "- `3` = 不满意（未达到最低满意线）\n\n"
            "除分数外，你还需要判断：这个样本是否【高度接近 3/4 边界】，需要进入第二遍复核。\n"
            "注意，`needs_refute_review=true` 必须是少数情况；只有在你确实拿不准时才允许触发。\n\n"
            f"【用户评分摘要】\n"
            f"评分风格：{memory.scoring_style}\n"
            f"历史平均分：{memory.avg_satisfaction_score:.2f}\n"
            f"满意最低线（3→4 边界）：{memory.three_vs_four_distinction}\n"
            f"更高要求（4→5，仅供背景参考）：{memory.four_vs_five_distinction}\n"
            f"用户特定要求：\n{user_reqs}\n"
            f"偏好回复形式：{memory.preferred_response_format}\n"
            + (f"任务特定观察：\n{task_obs_lines}\n" if task_obs_lines else "")
            + "\n"
            + f"{anchor_section}"
            + anchor_instruction
            + f"【用户画像】{_format_profile(profile)}\n\n"
            + f"【任务背景】{task_context}\n\n"
            + f"【最近对话历史】\n{history_text}\n\n"
            + f"【待评估的助手回复】\n{assistant_reply}\n\n"
            + f"【可选原因标签】{reason_text}\n{reason_rule_block}\n"
            + "【第一遍只做严格筛选后的边界判断】\n"
            + "Step 1. 判断回复是否回答了核心问题，并基本满足关键约束。\n"
            + "Step 2. 判断它是否达到该用户的满意最低线：达到给 `4`，未达到给 `3`。\n"
            + "Step 3. 再判断是否真的需要复核。只有下面两类高不确定情形才允许 `needs_refute_review=true`：\n"
            + "  - 当前判成 `3`，但你怀疑问题主要只是“边缘性的细节不足”，未必真的低于满意线\n"
            + "  - 当前判成 `4`，但你怀疑它可能漏掉了一个关键要求，是否仍算满意拿不准\n"
            + "Step 4. 若主要证据已经明显站在一边，必须输出 `needs_refute_review=false`。\n\n"
            + "注意：\n"
            + "- 不要因为“还可以更好”就触发复核。\n"
            + "- 不要因为理由是 `不够细致` 就自动触发复核。\n"
            + "- 只有当一个具体可疑点是否属于关键失败拿不准时，才触发复核。\n"
            + "- `classification` 只能输出 `3` 或 `4`。\n"
            + "- `analysis` 只需 1-2 句，明确写出：当前边界判断是什么；可疑点是什么；是否真的需要复核。\n\n"
            + "请严格输出 JSON，不要输出其他内容：\n"
            + "{\n"
            + '  "classification": 只能是 3 或 4,\n'
            + f'  "reason": "{reason_json_rule}",\n'
            + '  "analysis": "用 1-2 句写明当前为何判为 3 或 4、唯一的可疑点是什么，以及是否真的需要复核",\n'
            + '  "needs_refute_review": true 或 false\n'
            + "}\n"
        )
        return prompt

    if prompt_version == "boundary_34_selective_refute":
        anchor_instruction = (
            "【参考案例使用规则】\n"
            "1. 只把案例看成边界参考：真实分数 <=3 视为【未达满意线案例】，>=4 视为【达到满意线案例】。\n"
            "2. 案例只用于帮助你判断当前回复是否接近 3/4 边界，不要机械复用案例分数。\n"
            "3. 若当前回复明显优于未达满意线案例，或明显达到满意线，就不要触发二次复核。\n\n"
            if anchor_turns else ""
        )
        prompt = (
            "你是一名个性化满意度边界评估员。\n"
            "这是 selective-refute 的第一遍初判：先温和判断当前助手回复是否达到该用户的【满意最低线】。\n"
            "输出只能是：\n"
            "- `4` = 满意（达到最低满意线）\n"
            "- `3` = 不满意（未达到最低满意线）\n\n"
            "除分数外，你还需要判断这个样本是否【真的接近 3/4 边界】，从而需要进入二次反证复核。\n"
            "只有在证据混合、边界不稳时，才把 `needs_refute_review` 设为 `true`；明显满意或明显不满意都应设为 `false`。\n\n"
            f"【用户评分摘要】\n"
            f"评分风格：{memory.scoring_style}\n"
            f"历史平均分：{memory.avg_satisfaction_score:.2f}\n"
            f"满意最低线（3→4 边界）：{memory.three_vs_four_distinction}\n"
            f"更高要求（4→5，仅供背景参考）：{memory.four_vs_five_distinction}\n"
            f"用户特定要求：\n{user_reqs}\n"
            f"偏好回复形式：{memory.preferred_response_format}\n"
            + (f"任务特定观察：\n{task_obs_lines}\n" if task_obs_lines else "")
            + "\n"
            + f"{anchor_section}"
            + anchor_instruction
            + f"【用户画像】{_format_profile(profile)}\n\n"
            + f"【任务背景】{task_context}\n\n"
            + f"【最近对话历史】\n{history_text}\n\n"
            + f"【待评估的助手回复】\n{assistant_reply}\n\n"
            + f"【可选原因标签】{reason_text}\n{reason_rule_block}\n"
            + "【第一遍只做温和边界判断】\n"
            + "Step 1. 判断回复是否回答了核心问题，并基本满足关键约束。\n"
            + "Step 2. 判断它是否达到该用户的满意最低线：达到给 `4`，未达到给 `3`。\n"
            + "Step 3. 再判断这个案例是否【真的接近边界】。\n"
            + "  只有下面情况才把 `needs_refute_review=true`：\n"
            + "  - 回复大体有帮助，但有一个可能是关键缺陷的点，是否足以掉到 3 不确定\n"
            + "  - 当前判成 3，但主要问题可能只是“不够细致”，未必真的低于满意线\n"
            + "  - 当前判成 4，但可能漏掉了一个关键要求，是否仍算满意不确定\n"
            + "Step 4. 若结论已经很明显，就输出 `needs_refute_review=false`。\n\n"
            + "注意：\n"
            + "- `needs_refute_review=true` 应该是少数情况，不要把它当默认值。\n"
            + "- 不要因为回复不够优秀就自动触发复核；只有接近 3/4 边界时才触发。\n"
            + "- `classification` 只能输出 `3` 或 `4`。\n"
            + "- `analysis` 只需 1-2 句，写明当前判断依据，以及为什么需要或不需要二次复核。\n\n"
            + "请严格输出 JSON，不要输出其他内容：\n"
            + "{\n"
            + '  "classification": 只能是 3 或 4,\n'
            + f'  "reason": "{reason_json_rule}",\n'
            + '  "analysis": "用 1-2 句写明当前为何判为 3 或 4，以及是否接近 3/4 边界",\n'
            + '  "needs_refute_review": true 或 false\n'
            + "}\n"
        )
        return prompt

    if prompt_version == "boundary_34_refute_v2":
        anchor_instruction = (
            "【参考案例使用规则】\n"
            "1. 先把案例按边界用途理解：真实分数 <=3 是【未达满意线案例】，>=4 是【达到满意线案例】。\n"
            "2. 优先观察未达满意线案例中的【致命缺陷】是什么，再看达到满意线案例是否只是存在可改进的小缺口。\n"
            "3. 不要因为当前回复不如优秀案例完整，就直接判成 3。\n\n"
            if anchor_turns else ""
        )
        prompt = (
            "你是一名个性化满意度边界评估员。\n"
            "本题只判断当前助手回复是否达到该用户的【满意最低线】。\n"
            "输出只能是：\n"
            "- `4` = 满意（达到最低满意线）\n"
            "- `3` = 不满意（未达到最低满意线）\n\n"
            "你需要保留【反证检查】，但采用更温和的判定原则：\n"
            "只有当存在【明确且关键的失败】时，才允许判 `3`。\n"
            "如果回复已经回答了核心问题，关键约束也基本满足，而剩余问题只是“还不够细”“还可以更好”，应优先判 `4`。\n\n"
            f"【用户评分摘要】\n"
            f"评分风格：{memory.scoring_style}\n"
            f"历史平均分：{memory.avg_satisfaction_score:.2f}\n"
            f"满意最低线（3→4 边界）：{memory.three_vs_four_distinction}\n"
            f"更高要求（4→5，仅供背景参考）：{memory.four_vs_five_distinction}\n"
            f"用户特定要求：\n{user_reqs}\n"
            f"偏好回复形式：{memory.preferred_response_format}\n"
            + (f"任务特定观察：\n{task_obs_lines}\n" if task_obs_lines else "")
            + "\n"
            + f"{anchor_section}"
            + anchor_instruction
            + f"【用户画像】{_format_profile(profile)}\n\n"
            + f"【任务背景】{task_context}\n\n"
            + f"【最近对话历史】\n{history_text}\n\n"
            + f"【待评估的助手回复】\n{assistant_reply}\n\n"
            + f"【可选原因标签】{reason_text}\n{reason_rule_block}\n"
            + "【先做温和反证，再决定是否给 3】\n"
            + "Step 1. 先判断回复是否已经基本回答了用户的核心问题，并满足关键约束。\n"
            + "Step 2. 再检查是否存在【明确且关键的失败】。只有下面这些情况才足以判 `3`：\n"
            + "  - 没有直接回答主要问题\n"
            + "  - 明显忽略关键约束、条件或任务目标\n"
            + "  - 内容过于空泛，用户几乎无法据此采取行动\n"
            + "  - 漏掉了该用户最在意、且会显著影响满意度的要求\n"
            + "  - 存在会明显伤害可用性的缺口，而不是普通的“还不够细致”\n"
            + "Step 3. 明确区分两类问题：\n"
            + "  - 【致命缺陷】= 会让回复掉到 3\n"
            + "  - 【普通缺口】= 已达到最低满意线，但还不够优秀，仍应给 4\n"
            + "Step 4. 做简短反证：\n"
            + "  - 如果你能指出一个明确的【致命缺陷】，输出 `classification=3`\n"
            + "  - 如果只看到普通缺口，而没有致命缺陷，输出 `classification=4`\n"
            + "Step 5. 选择一个最贴切的原因标签。\n\n"
            + "注意：\n"
            + "- `不够细致` 本身不等于 `3`；只有它严重到导致回复不满足最低满意线时，才可以判 `3`。\n"
            + "- 不要因为它不如 5 分案例完整，就直接判 `3`。\n"
            + "- 如果回复已回答核心问题，且关键要求基本满足，应优先保护 `4`。\n"
            + "- `classification` 只能输出 `3` 或 `4`。\n"
            + "- `analysis` 只需简短说明：最强的降分证据是什么；它是否属于致命缺陷；最终为何判 3 或 4。\n\n"
            + "请严格输出 JSON，不要输出其他内容：\n"
            + "{\n"
            + '  "classification": 只能是 3 或 4,\n'
            + f'  "reason": "{reason_json_rule}",\n'
            + '  "analysis": "用 2-3 句写明：最强的降分证据是什么；它是否属于致命缺陷；最终为何判为 3 或 4。若只是普通缺口，应明确说明仍达到最低满意线" \n'
            + "}\n"
        )
        return prompt

    if prompt_version == "boundary_34_refute":
        anchor_instruction = (
            "【参考案例使用规则】\n"
            "1. 先把案例按边界用途理解：真实分数 <=3 是【未达满意线案例】，>=4 是【达到满意线案例】。\n"
            "2. 优先观察未达满意线案例缺了什么，再看达到满意线案例满足了什么。\n"
            "3. 不要机械复用案例分数；案例只用于帮助你发现“哪些缺口足以把回复判成 3”。\n\n"
            if anchor_turns else ""
        )
        prompt = (
            "你是一名个性化满意度边界评估员。\n"
            "本题只判断当前助手回复是否达到该用户的【满意最低线】。\n"
            "输出只能是：\n"
            "- `4` = 满意（达到最低满意线）\n"
            "- `3` = 不满意（未达到最低满意线）\n\n"
            "与旧版不同，本题必须先做【反证检查】。\n"
            "也就是说：先主动寻找足以把回复判成 `3` 的关键缺陷；"
            "只有当这些缺陷都不成立时，才允许给 `4`。\n\n"
            f"【用户评分摘要】\n"
            f"评分风格：{memory.scoring_style}\n"
            f"历史平均分：{memory.avg_satisfaction_score:.2f}\n"
            f"满意最低线（3→4 边界）：{memory.three_vs_four_distinction}\n"
            f"更高要求（4→5，仅供背景参考）：{memory.four_vs_five_distinction}\n"
            f"用户特定要求：\n{user_reqs}\n"
            f"偏好回复形式：{memory.preferred_response_format}\n"
            + (f"任务特定观察：\n{task_obs_lines}\n" if task_obs_lines else "")
            + "\n"
            + f"{anchor_section}"
            + anchor_instruction
            + f"【用户画像】{_format_profile(profile)}\n\n"
            + f"【任务背景】{task_context}\n\n"
            + f"【最近对话历史】\n{history_text}\n\n"
            + f"【待评估的助手回复】\n{assistant_reply}\n\n"
            + f"【可选原因标签】{reason_text}\n{reason_rule_block}\n"
            + "【先做失败检查，再决定是否给 4】\n"
            + "Step 1. 先检查是否存在任何一个【足以降到 3 分】的关键失败。\n"
            + "  重点检查：\n"
            + "  - 没有直接回答用户主要问题\n"
            + "  - 明显忽略关键约束、条件或任务目标\n"
            + "  - 内容太泛、太空，用户难以据此采取行动\n"
            + "  - 漏掉了该用户特别在意的要求或偏好格式\n"
            + "  - 存在会明显伤害满意度的缺口，而不只是“还可以更好”\n"
            + "Step 2. 做【反证】。\n"
            + "  问自己：如果我要把它判成 3，最强证据是什么？\n"
            + "  - 如果能找到明确且实质的证据，输出 `classification=3`\n"
            + "  - 只有当这些证据都站不住脚，才继续考虑 `classification=4`\n"
            + "Step 3. 只有同时满足下面两点，才能给 `4`：\n"
            + "  - 回复已经基本回答了用户问题，并满足关键要求\n"
            + "  - 没有发现任何一个足以把它拉回 3 的关键缺陷\n"
            + "Step 4. 选择一个最贴切的原因标签。\n\n"
            + "注意：\n"
            + "- 不要因为“语气像在帮忙”就给 4，关键是是否真正过了满意最低线。\n"
            + "- 也不要因为“还不够优秀”就给 3；只有出现了足以降到 3 的关键缺陷，才判 3。\n"
            + "- `classification` 只能输出 `3` 或 `4`。\n"
            + "- 在 `analysis` 里要明确写出：你检查过哪些降分证据，以及这些证据是否成立。\n\n"
            + "请严格输出 JSON，不要输出其他内容：\n"
            + "{\n"
            + '  "classification": 只能是 3 或 4,\n'
            + f'  "reason": "{reason_json_rule}",\n'
            + '  "analysis": "用 3-5 句写明：最可能把该回复判成 3 的关键缺陷是什么；这个缺陷是否成立；最终为什么判成 3 或 4。若使用参考案例，注明更接近未达满意线案例还是达到满意线案例" \n'
            + "}\n"
        )
        return prompt

    if prompt_version == "boundary_34":
        anchor_instruction = (
            "【参考案例使用规则】\n"
            "1. 只关心这些案例在满意边界上的含义：真实分数 <=3 视为【不满意案例】，>=4 视为【满意案例】。\n"
            "2. 不要尝试复用案例的精确分数，只判断当前回复更接近【不满意】还是【满意】。\n"
            "3. 你的任务不是判断这条回复有多优秀，而是判断：它有没有达到该用户的【最低满意线】。\n\n"
            if anchor_turns else ""
        )
        prompt = (
            "你是一名个性化满意度边界评估员。\n"
            "本题只判断当前助手回复是否达到该用户的【满意最低线】。\n"
            "请不要做 1/2/5 分细分，只输出：\n"
            "- `4` = 满意（达到最低满意线）\n"
            "- `3` = 不满意（未达到最低满意线）\n\n"
            f"【用户评分摘要】\n"
            f"评分风格：{memory.scoring_style}\n"
            f"历史平均分：{memory.avg_satisfaction_score:.2f}\n"
            f"满意最低线（3→4 边界）：{memory.three_vs_four_distinction}\n"
            f"补充说明（4→5 的更高要求，仅供参考，不作为本题判定目标）：{memory.four_vs_five_distinction}\n"
            f"用户特定要求：\n{user_reqs}\n"
            f"偏好回复形式：{memory.preferred_response_format}\n"
            + (f"任务特定观察：\n{task_obs_lines}\n" if task_obs_lines else "")
            + "\n"
            + f"{anchor_section}"
            + anchor_instruction
            + f"【用户画像】{_format_profile(profile)}\n\n"
            + f"【任务背景】{task_context}\n\n"
            + f"【最近对话历史】\n{history_text}\n\n"
            + f"【待评估的助手回复】\n{assistant_reply}\n\n"
            + f"【可选原因标签】{reason_text}\n{reason_rule_block}\n"
            + "【只按这 3 步判断】\n"
            + "Step 1. 先判断回复是否直接回答了用户问题，并满足关键约束。\n"
            + "Step 2. 再判断它是否达到该用户的【满意最低线】。\n"
            + "  - 若达到最低满意线，输出 `classification=4`\n"
            + "  - 若未达到最低满意线，输出 `classification=3`\n"
            + "Step 3. 选择一个最贴切的原因标签。\n\n"
            + "注意：\n"
            + "- 本题的目标是判定【满意 / 不满意】，不是区分 4 和 5。\n"
            + "- 除非回复明显没达到最低要求，否则不要因为“还不够优秀”就判成 3。\n"
            + "- `classification` 只能输出 `3` 或 `4`。\n\n"
            + "请严格输出 JSON，不要输出其他内容：\n"
            + "{\n"
            + '  "classification": 只能是 3 或 4,\n'
            + f'  "reason": "{reason_json_rule}",\n'
            + '  "analysis": "用 2-4 句写明：是否直接回答问题；是否达到最低满意线；最终为何判为 3 或 4。若使用参考案例，注明更接近满意案例还是不满意案例" \n'
            + "}\n"
        )
        return prompt

    if prompt_version == "qwen_short":
        anchor_instruction = (
            "【参考案例使用规则】\n"
            "1. 若提供了参考案例，先找与当前回复整体质量最接近的一条。\n"
            "2. 参考案例只用于帮助校准分数，不要因为它更完整就机械压低当前回复。\n"
            "3. 最终分数仍以【4分基线】和【5分门槛】为准。\n\n"
            if anchor_turns else ""
        )
        prompt = (
            "你是一名个性化满意度评分员。任务是给当前助手回复打 1-5 分。\n"
            "请严格按下面 checklist 判断，不要写长篇分析。\n\n"
            f"【用户评分摘要】\n"
            f"评分风格：{memory.scoring_style}\n"
            f"历史平均分：{memory.avg_satisfaction_score:.2f}\n"
            f"4分基线：{memory.three_vs_four_distinction}\n"
            f"5分门槛：{memory.four_vs_five_distinction}\n"
            f"用户特定要求：\n{user_reqs}\n"
            f"偏好回复形式：{memory.preferred_response_format}\n"
            + (f"任务特定观察：\n{task_obs_lines}\n" if task_obs_lines else "")
            + "\n"
            + f"{anchor_section}"
            + anchor_instruction
            + f"【用户画像】{_format_profile(profile)}\n\n"
            + f"【任务背景】{task_context}\n\n"
            + f"【最近对话历史】\n{history_text}\n\n"
            + f"【待评估的助手回复】\n{assistant_reply}\n\n"
            + f"【可选原因标签】{reason_text}\n{reason_rule_block}\n"
            + "【只按这 3 步判断】\n"
            + "Step 1. 先判断是否达到 4 分基线。\n"
            + "  - 若没有直接回答问题、明显忽略约束、帮助性不足，给 1/2/3。\n"
            + "Step 2. 若已达到 4 分，再判断是否满足 5 分门槛。\n"
            + "  - 只有明显满足关键细节、格式和用户特定要求时才给 5。\n"
            + "  - 只要整体合格但还缺少关键一项，就给 4。\n"
            + "Step 3. 选择一个最贴切的原因标签。\n\n"
            + "请严格输出 JSON，不要输出其他内容：\n"
            + "{\n"
            + '  "classification": 1-5 中的整数,\n'
            + f'  "reason": "{reason_json_rule}",\n'
            + '  "analysis": "用 2-4 句写明：是否过 4 分基线；若过基线，是否满足 5 分门槛；最终分数依据。若使用参考案例，注明案例编号" \n'
            + "}\n"
        )
        return prompt

    # 把 anchor 做成 rank-match 的先验：先定位最接近的案例并复用其分数，
    # rubric 仅用于验证一致性。这种框架下 rubric 不会把分数往下拽。
    extra_step = (
        "Step 0 (Rank-Match)：阅读上方【参考案例】。在 1-2 句内找出与当前回复"
        "【整体质量最接近】的一条案例（注意是比较整体水平，不是挑差异），"
        "把该案例的真实分数作为当前回复的初始估计。\n"
        "Step 1 (Sanity-Check)：用下面的 rubric 校验该估计与评分风格是否一致，"
        "仅当 rubric 明确提示了重大的差异（如缺失用户特定要求）才调整分数；"
        "若 rubric 与估计一致，保持 rank-match 得到的分数。\n"
        if anchor_turns else ""
    )

    prompt = (
        "你是一名个性化对话质量评估员。"
        "请严格按照以下该用户的个性化评分标准，对助手回复进行评分。\n\n"
        f"{rubric}\n"
        f"{anchor_section}"
        f"【用户画像】{_format_profile(profile)}\n\n"
        f"【任务背景】{task_context}\n\n"
        f"【最近对话历史】\n{history_text}\n\n"
        f"【待评估的助手回复】\n{assistant_reply}\n\n"
        f"【可选原因标签】{reason_text}\n{reason_rule_block}\n"
        "【评分步骤】请严格按以下顺序推理：\n"
        f"{extra_step}"
        + (
            "（若 Step 1 未提示需调整，直接输出 Step 0 的分数，跳过下面的 rubric-only 三步）\n"
            if anchor_turns else ""
        )
        + "Step A: 对照【3分以下→4分的门槛】判断此回复是否达到 4 分基线\n"
        "Step B: 若达到 4 分，再对照【4分→5分的门槛】判断是否满足 5 分条件\n"
        "Step C: 若未达到 4 分，根据缺陷的严重程度（参考用户特定要求）决定给 1/2/3 分\n\n"
        "请严格输出 JSON，不要输出其他内容：\n"
        "{\n"
        '  "classification": 1-5 中的整数,\n'
        f'  "reason": "{reason_json_rule}",\n'
        '  "analysis": "'
        + ('按 Step0/Step1/StepA-C 格式说明判断过程，'
           '先给出 rank-match 得到的分数和依据案例编号，再简述 Step 1 的一致性校验'
           if anchor_turns else
           '按 StepA/StepB/StepC 格式说明判断过程，须明确引用上方评分标准中的具体条件')
        + '"\n'
        "}\n"
    )
    return prompt


def build_turn_eval_refute_followup_prompt(
    memory: UserMemory,
    profile: dict,
    task_context: str,
    history_window: list[str],
    assistant_reply: str,
    initial_classification: int,
    initial_reason: str,
    initial_analysis: str,
    prompt_version: Literal["boundary_34_selective_refute", "boundary_34_selective_refute_v2"] = "boundary_34_selective_refute",
) -> str:
    """Selective-refute 第二遍复核 prompt。"""
    reason_labels = list(get_reason_to_id().keys())
    reason_text = "、".join(reason_labels)
    reason_rule_block = _format_reason_rule_block()
    reason_json_rule = _format_reason_json_rule()
    history_text = "\n".join(history_window) if history_window else "（无历史）"
    user_reqs = "\n".join(
        f"  - {r}" for r in memory.user_specific_requirements
    ) if memory.user_specific_requirements else "  （无特异性要求记录）"

    task_obs_lines = ""
    if memory.task_specific_observations:
        relevant = [o for o in memory.task_specific_observations
                    if task_context and o.task_name in task_context[:50]]
        others = [o for o in memory.task_specific_observations
                  if o not in relevant]
        ordered = relevant + others
        task_obs_lines = "\n".join(
            f"  {o.task_name}：{o.observation}" for o in ordered
        )

    if prompt_version == "boundary_34_selective_refute_v2":
        prompt = (
            "你是一名个性化满意度边界复核员。\n"
            "这是 selective-refute v2 的第二遍复核，只在第一遍认为样本高度接近 3/4 边界时触发。\n"
            "你的任务不是重新完整评分，而是核实：第一遍指出的唯一可疑点，是否真的足以推翻第一遍初判。\n\n"
            "输出只能是：\n"
            "- `4` = 满意（达到最低满意线）\n"
            "- `3` = 不满意（未达到最低满意线）\n\n"
            f"【用户评分摘要】\n"
            f"评分风格：{memory.scoring_style}\n"
            f"满意最低线（3→4 边界）：{memory.three_vs_four_distinction}\n"
            f"更高要求（4→5，仅供背景参考）：{memory.four_vs_five_distinction}\n"
            f"用户特定要求：\n{user_reqs}\n"
            f"偏好回复形式：{memory.preferred_response_format}\n"
            + (f"任务特定观察：\n{task_obs_lines}\n" if task_obs_lines else "")
            + "\n"
            + f"【用户画像】{_format_profile(profile)}\n\n"
            + f"【任务背景】{task_context}\n\n"
            + f"【最近对话历史】\n{history_text}\n\n"
            + f"【待评估的助手回复】\n{assistant_reply}\n\n"
            + f"【第一遍初判】classification={initial_classification}, reason={initial_reason}\n"
            + f"【第一遍依据】{initial_analysis}\n\n"
            + f"【可选原因标签】{reason_text}\n{reason_rule_block}\n"
            + "【复核规则】\n"
            + "Step 1. 先把第一遍的可疑点复述成一个明确问题：它到底是不是关键失败？\n"
            + "Step 2. 默认保持第一遍初判，只有在发现【明确反证】时才允许改判。\n"
            + "Step 3. 如果第一遍判 `3`：只有当你能明确指出核心问题已被回答、关键约束也已满足时，才可改为 `4`。\n"
            + "Step 4. 如果第一遍判 `4`：只有当你能明确指出关键要求被漏掉、核心问题未被回答，或回复明显低于最低满意线时，才可改为 `3`。\n"
            + "Step 5. 不要因为模糊的“也许够了”或“还可以更好”就改判；改判必须有明确证据。\n\n"
            + "注意：\n"
            + "- 这是核实，不是重新打分。\n"
            + "- 第二遍不应默认保护 `4`，也不应默认推翻第一遍；默认动作是维持初判。\n"
            + "- `classification` 只能输出 `3` 或 `4`。\n"
            + "- `analysis` 只写 1-2 句：是否发现足以推翻初判的明确反证；最终为何维持或改判。\n\n"
            + "请严格输出 JSON，不要输出其他内容：\n"
            + "{\n"
            + '  "classification": 只能是 3 或 4,\n'
            + f'  "reason": "{reason_json_rule}",\n'
            + '  "analysis": "用 1-2 句写明：是否发现足以推翻第一遍初判的明确反证；最终为何维持或改判" \n'
            + "}\n"
        )
        return prompt

    prompt = (
        "你是一名个性化满意度边界复核员。\n"
        "这是 selective-refute 的第二遍复核，只在第一遍认为样本接近 3/4 边界时触发。\n"
        "你的任务不是重新长篇分析，而是检查：第一遍提到的可疑问题，是否真的足以跨过满意/不满意边界。\n\n"
        "输出只能是：\n"
        "- `4` = 满意（达到最低满意线）\n"
        "- `3` = 不满意（未达到最低满意线）\n\n"
        f"【用户评分摘要】\n"
        f"评分风格：{memory.scoring_style}\n"
        f"满意最低线（3→4 边界）：{memory.three_vs_four_distinction}\n"
        f"更高要求（4→5，仅供背景参考）：{memory.four_vs_five_distinction}\n"
        f"用户特定要求：\n{user_reqs}\n"
        f"偏好回复形式：{memory.preferred_response_format}\n"
        + (f"任务特定观察：\n{task_obs_lines}\n" if task_obs_lines else "")
        + "\n"
        + f"【用户画像】{_format_profile(profile)}\n\n"
        + f"【任务背景】{task_context}\n\n"
        + f"【最近对话历史】\n{history_text}\n\n"
        + f"【待评估的助手回复】\n{assistant_reply}\n\n"
        + f"【第一遍初判】classification={initial_classification}, reason={initial_reason}\n"
        + f"【第一遍依据】{initial_analysis}\n\n"
        + f"【可选原因标签】{reason_text}\n{reason_rule_block}\n"
        + "【复核规则】\n"
        + "Step 1. 只盯住第一遍提到的可疑点，判断它是否真的是【关键失败】。\n"
        + "Step 2. 若该问题只是普通缺口、轻度不够细致、仍不影响核心可用性，应保护 `4`。\n"
        + "Step 3. 只有当该问题确实导致核心问题未被回答、关键要求被忽略，或回复明显低于最低满意线时，才判 `3`。\n"
        + "Step 4. 给出最终 3/4，并选一个最贴切的原因标签。\n\n"
        + "注意：\n"
        + "- 这是复核，不要重新展开完整评分流程。\n"
        + "- 若第一遍的可疑点并不足以跨过边界，应维持或改判为 `4`。\n"
        + "- `classification` 只能输出 `3` 或 `4`。\n"
        + "- `analysis` 只写 1-2 句：该可疑点是否构成关键失败；最终为何判成 3 或 4。\n\n"
        + "请严格输出 JSON，不要输出其他内容：\n"
        + "{\n"
        + '  "classification": 只能是 3 或 4,\n'
        + f'  "reason": "{reason_json_rule}",\n'
        + '  "analysis": "用 1-2 句写明：第一遍提到的可疑点是否真的足以跨过满意边界，以及最终为何判 3 或 4" \n'
        + "}\n"
    )
    return prompt


def build_turn_eval_fullscale_sat_refinement_prompt(
    memory: UserMemory,
    profile: dict,
    task_context: str,
    history_window: list[str],
    assistant_reply: str,
    router_reason: str,
    router_analysis: str,
    anchor_turns: list | None = None,
) -> str:
    """在 boundary router 判为 SAT 后，细化到 4/5。"""
    reason_labels = list(get_reason_to_id().keys())
    reason_text = "、".join(reason_labels)
    reason_rule_block = _format_reason_rule_block()
    reason_json_rule = _format_reason_json_rule()
    history_text = "\n".join(history_window) if history_window else "（无历史）"
    user_reqs = "\n".join(
        f"  - {r}" for r in memory.user_specific_requirements
    ) if memory.user_specific_requirements else "  （无特异性要求记录）"

    task_obs_lines = ""
    if memory.task_specific_observations:
        relevant = [o for o in memory.task_specific_observations
                    if task_context and o.task_name in task_context[:50]]
        others = [o for o in memory.task_specific_observations
                  if o not in relevant]
        ordered = relevant + others
        task_obs_lines = "\n".join(
            f"  {o.task_name}：{o.observation}" for o in ordered
        )

    anchor_block = _format_anchor_turns(anchor_turns or [])
    anchor_section = (anchor_block + "\n") if anchor_block else ""
    anchor_instruction = (
        "【参考案例使用规则】\n"
        "1. 在这一步只关心 `4` 和 `5` 的区别。\n"
        "2. 若使用参考案例，优先比较真实分数为 `4/5` 的案例；<=3 的案例只说明回复至少已经过线，不用于决定 5 分。\n"
        "3. 不要因为回复已经过了满意线，就自动给 5；只有明确达到 5 分门槛时才升到 5。\n\n"
        if anchor_turns else ""
    )

    return (
        "你是一名个性化满意度细化评估员。\n"
        "第一层 boundary router 已确认：当前回复至少达到满意线。\n"
        "你的任务不是重新判断满意/不满意，而是只在 `4` 和 `5` 之间做细化。\n\n"
        "输出只能是：\n"
        "- `5` = 明确达到该用户的高满意门槛\n"
        "- `4` = 已满意，但还没到 5 分门槛\n\n"
        f"【用户评分摘要】\n"
        f"评分风格：{memory.scoring_style}\n"
        f"历史平均分：{memory.avg_satisfaction_score:.2f}\n"
        f"满意最低线（3→4，仅供背景参考）：{memory.three_vs_four_distinction}\n"
        f"高满意门槛（4→5 关键）：{memory.four_vs_five_distinction}\n"
        f"用户特定要求：\n{user_reqs}\n"
        f"偏好回复形式：{memory.preferred_response_format}\n"
        + (f"任务特定观察：\n{task_obs_lines}\n" if task_obs_lines else "")
        + "\n"
        + f"{anchor_section}"
        + anchor_instruction
        + f"【用户画像】{_format_profile(profile)}\n\n"
        + f"【任务背景】{task_context}\n\n"
        + f"【最近对话历史】\n{history_text}\n\n"
        + f"【待评估的助手回复】\n{assistant_reply}\n\n"
        + f"【第一层 router 输出】reason={router_reason}\n"
        + f"【第一层 router 分析】{router_analysis}\n\n"
        + f"【可选原因标签】{reason_text}\n{reason_rule_block}\n"
        + "【4/5 细化规则】\n"
        + "Step 1. 把 `4` 当成默认值：既然已经过了满意线，除非有明确证据达到高满意门槛，否则保持 `4`。\n"
        + "Step 2. 只检查这些是否足以升到 `5`：\n"
        + "  - 是否完整命中用户真正关心的点，而不只是基本回答\n"
        + "  - 是否满足了该用户对细节、格式、可执行性、资源具体度的更高要求\n"
        + "  - 是否几乎没有明显短板，整体完成度接近该用户的高满意案例\n"
        + "Step 3. 如果只是“合格但还有一两处明显缺口”，输出 `4`；只有明确达到 `four_vs_five_distinction` 描述的高门槛，才输出 `5`。\n\n"
        + "注意：\n"
        + "- 本步不能回退到 `3`。\n"
        + f"- 由于最终分数 >=4，`reason` 必须输出 `{SATISFIED_REASON}`。\n"
        + "- `analysis` 只需 1-2 句，明确写出：为什么仍是 4，或为什么已经到 5。\n\n"
        + "请严格输出 JSON，不要输出其他内容：\n"
        + "{\n"
        + '  "classification": 只能是 4 或 5,\n'
        + f'  "reason": "{reason_json_rule}",\n'
        + '  "analysis": "用 1-2 句写明：为什么保持 4，或为什么已经达到 5 分门槛" \n'
        + "}\n"
    )


def build_turn_eval_fullscale_dsat_refinement_prompt(
    memory: UserMemory,
    profile: dict,
    task_context: str,
    history_window: list[str],
    assistant_reply: str,
    router_reason: str,
    router_analysis: str,
    anchor_turns: list | None = None,
) -> str:
    """在 boundary router 判为 DSAT 后，细化到 1/2/3。"""
    reason_labels = list(get_reason_to_id().keys())
    reason_text = "、".join(reason_labels)
    reason_rule_block = _format_reason_rule_block()
    reason_json_rule = _format_reason_json_rule()
    history_text = "\n".join(history_window) if history_window else "（无历史）"
    user_reqs = "\n".join(
        f"  - {r}" for r in memory.user_specific_requirements
    ) if memory.user_specific_requirements else "  （无特异性要求记录）"

    task_obs_lines = ""
    if memory.task_specific_observations:
        relevant = [o for o in memory.task_specific_observations
                    if task_context and o.task_name in task_context[:50]]
        others = [o for o in memory.task_specific_observations
                  if o not in relevant]
        ordered = relevant + others
        task_obs_lines = "\n".join(
            f"  {o.task_name}：{o.observation}" for o in ordered
        )

    anchor_block = _format_anchor_turns(anchor_turns or [])
    anchor_section = (anchor_block + "\n") if anchor_block else ""
    anchor_instruction = (
        "【参考案例使用规则】\n"
        "1. 在这一步只关心 `1/2/3` 的严重度差异。\n"
        "2. 若使用参考案例，优先比较真实分数 <=3 的案例；>=4 的案例只说明当前回复已经确定没过线。\n"
        "3. 不要把所有不满意都压成 3；`2` 和 `1` 只留给明显更严重的失败。\n\n"
        if anchor_turns else ""
    )

    return (
        "你是一名个性化满意度细化评估员。\n"
        "第一层 boundary router 已确认：当前回复没有达到满意线。\n"
        "你的任务不是重新判断是否满意，而是只在 `1/2/3` 之间细化严重程度。\n\n"
        "输出只能是：\n"
        "- `3` = 不满意，但仍有部分帮助，或只是明显低于满意线\n"
        "- `2` = 很不满意，核心问题大多没解决，帮助性较弱\n"
        "- `1` = 极不满意，几乎不可用、明显错误或严重偏离需求\n\n"
        f"【用户评分摘要】\n"
        f"评分风格：{memory.scoring_style}\n"
        f"历史平均分：{memory.avg_satisfaction_score:.2f}\n"
        f"满意最低线（3→4 关键）：{memory.three_vs_four_distinction}\n"
        f"高满意门槛（4→5，仅供背景参考）：{memory.four_vs_five_distinction}\n"
        f"用户特定要求：\n{user_reqs}\n"
        f"偏好回复形式：{memory.preferred_response_format}\n"
        + (f"任务特定观察：\n{task_obs_lines}\n" if task_obs_lines else "")
        + "\n"
        + f"{anchor_section}"
        + anchor_instruction
        + f"【用户画像】{_format_profile(profile)}\n\n"
        + f"【任务背景】{task_context}\n\n"
        + f"【最近对话历史】\n{history_text}\n\n"
        + f"【待评估的助手回复】\n{assistant_reply}\n\n"
        + f"【第一层 router 输出】reason={router_reason}\n"
        + f"【第一层 router 分析】{router_analysis}\n\n"
        + f"【可选原因标签】{reason_text}\n{reason_rule_block}\n"
        + "【1/2/3 细化规则】\n"
        + "Step 1. 先接受第一层结论：当前回复已经没过满意线，因此本步只能在 `1/2/3` 之间选。\n"
        + "Step 2. 判断严重度：\n"
        + "  - `3`：仍有一定帮助，但关键缺口让它没过满意线\n"
        + "  - `2`：帮助性有限，核心问题大多没解决，内容较空泛或关键要求大面积缺失\n"
        + "  - `1`：几乎不可用、严重答非所问、明显错误，或几乎没有可执行信息\n"
        + "Step 3. 只有在失败非常严重时才给 `1`；一般的“不满意但有点用”应优先给 `3`，而不是过度压到 `1/2`。\n"
        + "Step 4. 选择一个最贴切的不满意原因标签。\n\n"
        + "注意：\n"
        + "- 本步不能回升到 `4/5`。\n"
        + f"- 由于最终分数 <=3，`reason` 只能选择不满意原因，不能输出 `{SATISFIED_REASON}`。\n"
        + "- `analysis` 只需 1-2 句，明确写出：为什么是 3，或为什么严重到 2/1。\n\n"
        + "请严格输出 JSON，不要输出其他内容：\n"
        + "{\n"
        + '  "classification": 只能是 1、2 或 3,\n'
        + f'  "reason": "{reason_json_rule}",\n'
        + '  "analysis": "用 1-2 句写明：当前失败严重到什么程度，以及为什么是 3 / 2 / 1" \n'
        + "}\n"
    )


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
