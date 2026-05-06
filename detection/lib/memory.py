"""
用户记忆模块（User Memory） v2 / v3

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

import json
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


class UserMemoryContentV3(UserMemoryContent):
    """
    LLM 生成的用户记忆（v3）。

    与 v2 保持核心字段兼容，但在 prompt 侧强调：
      1. 证据不足时必须显式保守
      2. calibration 信息与边界规则分离
      3. requirements 只保留真正会改变评分的个性化要求
    """


def _score_distribution_to_counts(
    dist: ScoreDistribution,
) -> dict[int, int]:
    return {
        1: dist.score_1,
        2: dist.score_2,
        3: dist.score_3,
        4: dist.score_4,
        5: dist.score_5,
    }


def _build_v3_calibration_summary(
    avg_score: float,
    dist: ScoreDistribution,
) -> str:
    counts = _score_distribution_to_counts(dist)
    total = sum(counts.values()) or 1
    sat_ratio = (counts[4] + counts[5]) / total
    if avg_score >= 4.4:
        style = "偏宽松"
    elif avg_score >= 4.0:
        style = "中等"
    elif avg_score >= 3.6:
        style = "偏严格"
    else:
        style = "严格"
    return (
        f"{style}：历史均分 {avg_score:.2f}，"
        f"SAT 占比 {sat_ratio:.1%}（4分×{counts[4]} / 5分×{counts[5]}），"
        f"DSAT 证据 {counts[1] + counts[2] + counts[3]} 轮。"
    )


def _build_v3_evidence_notes(
    dist: ScoreDistribution,
) -> list[str]:
    counts = _score_distribution_to_counts(dist)
    notes: list[str] = []
    if counts[3] == 0 or counts[4] == 0:
        notes.append("3/4 边界缺直接相邻证据，只能弱推断，不能把 three_vs_four_distinction 当硬规则。")
    if counts[4] == 0 or counts[5] == 0:
        notes.append("4/5 边界缺直接相邻证据，只能弱推断，不能把 four_vs_five_distinction 当硬规则。")
    if counts[1] + counts[2] + counts[3] == 0:
        notes.append("没有任何 <=3 的历史样本，低分严重度细分基本无证据支撑。")
    elif counts[1] + counts[2] + counts[3] <= 3:
        notes.append("<=3 的历史样本很少，1/2/3 细分应保守，默认不要轻易给 1 或 2。")
    return notes


def _low_score_evidence_level(dist: ScoreDistribution) -> Literal["none", "sparse", "moderate", "rich"]:
    low = dist.score_1 + dist.score_2 + dist.score_3
    if low == 0:
        return "none"
    if low <= 3:
        return "sparse"
    if low <= 8:
        return "moderate"
    return "rich"


class UserMemoryV3(UserMemory):
    """完整用户记忆（v3）= v2 核心字段 + 程序侧证据充分性元信息。"""

    memory_version: str = Field(default="v3")
    calibration_summary: str = Field(
        description="程序侧生成的校准摘要，概括该用户的整体打分刻度",
    )
    can_compare_3_vs_4: bool = Field(
        description="是否同时存在 3 分与 4 分历史，可支持较可靠的 3/4 边界比较",
    )
    can_compare_4_vs_5: bool = Field(
        description="是否同时存在 4 分与 5 分历史，可支持较可靠的 4/5 边界比较",
    )
    low_score_evidence_level: Literal["none", "sparse", "moderate", "rich"] = Field(
        description="<=3 历史样本的证据丰富度，用于控制 1/2/3 细分时的保守程度",
    )
    evidence_notes: list[str] = Field(
        default_factory=list,
        description="程序侧生成的证据充分性提醒",
    )

    @classmethod
    def from_content(
        cls,
        content: UserMemoryContentV3,
        source_tasks: list[str] | None = None,
        n_history_sessions: int = 0,
        n_history_turns: int = 0,
    ) -> "UserMemoryV3":
        data = content.model_dump()
        dist = content.score_distribution
        data["memory_version"] = "v3"
        data["source_tasks"] = list(source_tasks or [])
        data["n_history_sessions"] = n_history_sessions
        data["n_history_turns"] = n_history_turns
        data["calibration_summary"] = _build_v3_calibration_summary(
            content.avg_satisfaction_score,
            dist,
        )
        data["can_compare_3_vs_4"] = dist.score_3 > 0 and dist.score_4 > 0
        data["can_compare_4_vs_5"] = dist.score_4 > 0 and dist.score_5 > 0
        data["low_score_evidence_level"] = _low_score_evidence_level(dist)
        data["evidence_notes"] = _build_v3_evidence_notes(dist)
        return cls(**data)


class MemoryUpdatePatchV2_1(BaseModel):
    """
    v2.1 的 memory update patch。

    设计目标：
      - 统计量由代码端确定性更新
      - verbal 字段只做字段级 patch，而不是整份 memory 重写
    """

    update_scoring_style: bool = Field(
        description="是否根据新证据改写 scoring_style"
    )
    scoring_style: str = Field(
        description="若 update_scoring_style=true，则给出更新后的 scoring_style；否则原样复述现有值"
    )
    update_three_vs_four: bool = Field(
        description="是否根据新 session 的明确 3/4 证据改写 three_vs_four_distinction"
    )
    three_vs_four_distinction: str = Field(
        description="若 update_three_vs_four=true，则给出新的 3/4 边界总结；否则原样复述现有值"
    )
    update_four_vs_five: bool = Field(
        description="是否根据新 session 的明确 4/5 证据改写 four_vs_five_distinction"
    )
    four_vs_five_distinction: str = Field(
        description="若 update_four_vs_five=true，则给出新的 4/5 边界总结；否则原样复述现有值"
    )
    add_user_specific_requirements: list[str] = Field(
        description=(
            "需要新增到 user_specific_requirements 的条目，0-3 条。"
            "只允许新增真正有辨识度、会改变评分的个性化要求；若无新增则返回空列表。"
        )
    )
    update_preferred_response_format: bool = Field(
        description="是否根据新证据改写 preferred_response_format"
    )
    preferred_response_format: str = Field(
        description="若 update_preferred_response_format=true，则给出新的 preferred_response_format；否则原样复述现有值"
    )
    add_or_update_task_specific_observations: list[TaskObservation] = Field(
        description=(
            "需要新增或覆盖的 task_specific_observations。"
            "仅在新 session 对某任务提供了明确新信息时输出；否则返回空列表。"
        )
    )
    rationale: str = Field(
        description="1-3 句简要说明：这次 update 主要依据哪些新证据，哪些字段保持不变"
    )


class RequirementPatchV2_2(BaseModel):
    requirement: str = Field(description="候选新增 requirement 文本")
    confidence: Literal["low", "medium", "high"] = Field(
        description="该 requirement 的证据强度"
    )
    support_count: int = Field(
        ge=0,
        le=10,
        description="本 session 中支持该 requirement 的证据条数",
    )


class TaskObservationPatchV2_2(BaseModel):
    task_name: str = Field(description="任务类型名称，如旅行规划")
    observation: str = Field(description="候选 observation 文本")
    confidence: Literal["low", "medium", "high"] = Field(
        description="该 task observation 的证据强度"
    )
    support_count: int = Field(
        ge=0,
        le=10,
        description="本 session 中支持该 observation 的证据条数",
    )


class MemoryUpdatePatchV2_2(BaseModel):
    """
    v2.2 的 memory update patch。

    核心改动：
      - update 输入从原始 session 改为结构化 evidence bundle
      - patch 显式输出每个字段的证据强度和支持数
      - 程序侧再做更硬的 gate，避免 noisy non-oracle update 污染边界规则
    """

    update_scoring_style: bool = Field(description="是否建议更新 scoring_style")
    scoring_style: str = Field(
        description="若不更新则原样复述现有值；若更新则给出新的 scoring_style"
    )
    scoring_style_confidence: Literal["low", "medium", "high"] = Field(
        description="更新 scoring_style 的证据强度"
    )

    update_three_vs_four: bool = Field(
        description="是否建议更新 three_vs_four_distinction"
    )
    three_vs_four_distinction: str = Field(
        description="若不更新则原样复述现有值；若更新则给出新的 3/4 边界总结"
    )
    three_vs_four_confidence: Literal["low", "medium", "high"] = Field(
        description="更新 3/4 边界的证据强度"
    )
    three_vs_four_evidence_count: int = Field(
        ge=0,
        le=10,
        description="本 session 中支持更新 3/4 边界的证据条数",
    )

    update_four_vs_five: bool = Field(
        description="是否建议更新 four_vs_five_distinction"
    )
    four_vs_five_distinction: str = Field(
        description="若不更新则原样复述现有值；若更新则给出新的 4/5 边界总结"
    )
    four_vs_five_confidence: Literal["low", "medium", "high"] = Field(
        description="更新 4/5 边界的证据强度"
    )
    four_vs_five_evidence_count: int = Field(
        ge=0,
        le=10,
        description="本 session 中支持更新 4/5 边界的证据条数",
    )

    add_user_specific_requirements: list[RequirementPatchV2_2] = Field(
        description="候选新增 requirement 列表；只保留真正有辨识度的要求"
    )

    update_preferred_response_format: bool = Field(
        description="是否建议更新 preferred_response_format"
    )
    preferred_response_format: str = Field(
        description="若不更新则原样复述现有值；若更新则给出新的格式偏好"
    )
    preferred_response_format_confidence: Literal["low", "medium", "high"] = Field(
        description="更新格式偏好的证据强度"
    )
    preferred_response_format_support_count: int = Field(
        ge=0,
        le=10,
        description="本 session 中支持该格式偏好的证据条数",
    )

    add_or_update_task_specific_observations: list[TaskObservationPatchV2_2] = Field(
        description="候选任务观察 patch；只在当前 session 提供了新信息时输出"
    )

    rationale: str = Field(
        description="1-4 句说明：哪些 evidence 可信、哪些字段不该更新以及原因"
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


def _collect_update_turns(
    new_session: SessionData,
    turn_predictions: list[dict],
    use_oracle_labels: bool,
) -> list[dict]:
    """抽取 update 所需的 turn 级证据，分数字段优先使用 gold（oracle）否则使用 pred。"""
    turns: list[dict] = []
    assistant_idx = 0
    last_user = ""
    for utt in new_session.history:
        if utt["role"] == "user":
            last_user = utt["content"]
        elif utt["role"] == "assistant" and assistant_idx < len(turn_predictions):
            pred = turn_predictions[assistant_idx]
            score = int(pred["gold_score"] if use_oracle_labels else pred["pred_score"])
            if use_oracle_labels:
                reason = pred.get("gold_reason", SATISFIED_REASON if score >= 4 else "其它")
            else:
                reason = pred.get("pred_reason", SATISFIED_REASON if score >= 4 else "其它")
            turns.append({
                "turn_idx": assistant_idx,
                "task": new_session.task,
                "user_msg": last_user,
                "assistant_reply": utt["content"],
                "score": score,
                "reason": reason,
                "pred_score": int(pred.get("pred_score", score)),
                "pred_reason": pred.get("pred_reason", reason),
                "analysis": pred.get("analysis", ""),
            })
            assistant_idx += 1
    return turns


def _format_update_examples(
    title: str,
    turns: list[dict],
    max_examples: int = 2,
) -> str:
    if not turns:
        return f"{title}\n  （无）"
    lines = [title]
    for i, t in enumerate(turns[:max_examples], 1):
        reason_suffix = f"（{t['reason']}）" if t["score"] <= 3 else ""
        lines.append(f"  [{i}] ★{t['score']}{reason_suffix}")
        lines.append(f"      用户：{_truncate(t['user_msg'], 120)}")
        lines.append(f"      助手：{_truncate(t['assistant_reply'])}")
    return "\n".join(lines)


def _dedupe_preserve_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        normalized = item.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        out.append(normalized)
    return out


_GENERIC_REQUIREMENT_FRAGMENTS = (
    "详细", "具体", "清晰", "结构化", "实用", "全面", "完整", "分点", "有条理",
)


def _is_generic_requirement(text: str) -> bool:
    normalized = text.strip()
    if not normalized:
        return True
    if len(normalized) <= 4:
        return True
    return any(fragment in normalized for fragment in _GENERIC_REQUIREMENT_FRAGMENTS)


def _count_analysis_borderline_mentions(turn_predictions: list[dict]) -> int:
    keywords = ("基线", "边界", "接近", "未达到4", "未达到 4", "达到4", "达到 4", "未达到5", "未达到 5")
    count = 0
    for pred in turn_predictions:
        analysis = str(pred.get("analysis", ""))
        if any(k in analysis for k in keywords):
            count += 1
    return count


def _build_update_evidence_bundle(
    existing_memory: UserMemory,
    new_session: SessionData,
    turn_predictions: list[dict],
    use_oracle_labels: bool,
) -> dict:
    turns = _collect_update_turns(new_session, turn_predictions, use_oracle_labels)
    by_score: dict[int, list[dict]] = defaultdict(list)
    for t in turns:
        by_score[t["score"]].append(t)

    def _format_examples(examples: list[dict], limit: int = 2) -> list[dict]:
        out: list[dict] = []
        for t in examples[:limit]:
            out.append({
                "turn_idx": t["turn_idx"],
                "score": t["score"],
                "reason": t["reason"],
                "user_request": _truncate(t["user_msg"], 120),
                "assistant_reply_excerpt": _truncate(t["assistant_reply"], 160),
                "analysis_signal": _truncate(t.get("analysis", ""), 120),
            })
        return out

    score_sequence = [t["score"] for t in turns]
    score_flip_34 = sum(
        1 for a, b in zip(score_sequence, score_sequence[1:])
        if {a, b} == {3, 4}
    )
    score_flip_45 = sum(
        1 for a, b in zip(score_sequence, score_sequence[1:])
        if {a, b} == {4, 5}
    )
    pred_distribution = {
        f"score_{s}": sum(1 for t in turns if t["score"] == s)
        for s in range(1, 6)
    }
    dominant_score = max(pred_distribution.items(), key=lambda kv: kv[1])[0] if turns else "score_4"

    return {
        "session_summary": {
            "task": new_session.task,
            "n_turns": len(turns),
            "score_sequence": score_sequence,
            "mean_score": round(sum(score_sequence) / len(score_sequence), 4) if score_sequence else 0.0,
            "min_score": min(score_sequence) if score_sequence else None,
            "max_score": max(score_sequence) if score_sequence else None,
            "task_context_excerpt": _truncate(new_session.task_context, 160),
            "uses_oracle_labels": use_oracle_labels,
        },
        "current_memory_summary": {
            "avg_satisfaction_score": existing_memory.avg_satisfaction_score,
            "score_distribution": existing_memory.score_distribution.model_dump(),
            "scoring_style": existing_memory.scoring_style,
        },
        "boundary_evidence": {
            "n_leq3": len(by_score[1]) + len(by_score[2]) + len(by_score[3]),
            "n_4": len(by_score[4]),
            "n_5": len(by_score[5]),
            "has_3_and_4": bool(by_score[3] and by_score[4]),
            "has_4_and_5": bool(by_score[4] and by_score[5]),
            "low_examples": _format_examples(by_score[1] + by_score[2] + by_score[3]),
            "mid_examples": _format_examples(by_score[4]),
            "high_examples": _format_examples(by_score[5]),
        },
        "uncertainty_signals": {
            "predicted_score_only": not use_oracle_labels,
            "analysis_borderline_mentions": _count_analysis_borderline_mentions(turn_predictions),
            "score_flip_3_4": score_flip_34,
            "score_flip_4_5": score_flip_45,
            "dominant_score": dominant_score,
            "dominant_score_ratio": (
                round(max(pred_distribution.values()) / len(turns), 4) if turns else 0.0
            ),
        },
    }


def build_memory_update_prompt_v2_1(
    existing_memory: UserMemory,
    new_session: SessionData,
    turn_predictions: list[dict],
    use_oracle_labels: bool = False,
) -> str:
    """
    构造 memory update prompt（v2.1）。

    核心改动：
      1. 不再要求整份 memory 重写
      2. 统计量由代码侧更新
      3. verbal 字段只允许 patch 式修改
    """
    existing_json = existing_memory.model_dump_json(
        indent=2,
        exclude={"memory_version", "source_tasks", "n_history_sessions", "n_history_turns"},
    )
    turns = _collect_update_turns(new_session, turn_predictions, use_oracle_labels)
    by_score: dict[int, list[dict]] = defaultdict(list)
    for t in turns:
        by_score[t["score"]].append(t)

    low_turns = by_score[1] + by_score[2] + by_score[3]
    mid_turns = by_score[4]
    high_turns = by_score[5]

    session_stats = (
        f"本 session 分布："
        f"5分×{len(by_score[5])} / 4分×{len(by_score[4])} / 3分×{len(by_score[3])} / "
        f"2分×{len(by_score[2])} / 1分×{len(by_score[1])}"
    )
    label_note = (
        "本次提供了真实标签，可视为可靠证据。"
        if use_oracle_labels
        else "本次仅有模型预测标签，属于弱证据；除非形成清晰模式，否则不要改 verbal 边界字段。"
    )
    prompt = (
        "你正在维护一份 v2.1 用户记忆。请根据新 session 给出【字段级 patch】，而不是重写整份 memory。\n\n"
        f"【现有记忆】\n{existing_json}\n\n"
        f"【新 Session 概览】\n"
        f"任务：{new_session.task}\n"
        f"任务背景：{_truncate(new_session.task_context, 180)}\n"
        f"{session_stats}\n"
        f"【说明】{label_note}\n\n"
        f"{_format_update_examples('【<=3 证据（可能影响 3/4 边界或低分要求）】', low_turns)}\n\n"
        f"{_format_update_examples('【4 分证据（与 <=3 或 5 比较时使用）】', mid_turns)}\n\n"
        f"{_format_update_examples('【5 分证据（可能影响 4/5 边界）】', high_turns)}\n\n"
        "更新原则：\n"
        "1. avg_satisfaction_score 和 score_distribution 由程序自动更新，你不需要负责统计数字。\n"
        "2. 只有在新 session 提供了明确相邻分数证据时，才改写边界字段：\n"
        "   - 改写 three_vs_four_distinction 需要有清晰的 <=3 与 4 分对比证据\n"
        "   - 改写 four_vs_five_distinction 需要有清晰的 4 与 5 分对比证据\n"
        "3. 如果只是重复了现有模式，必须保持 verbal 字段不变。\n"
        "4. user_specific_requirements 只能新增真正会改变评分的个性化要求；禁止加入泛化要求。\n"
        "5. preferred_response_format 只有在出现新的稳定格式偏好时才改。\n"
        "6. task_specific_observations 只新增/覆盖当前 session 提供了明确新信息的任务观察。\n\n"
        "请严格按下面的 JSON Schema 输出 patch，不要输出其他内容：\n"
        "{\n"
        '  "update_scoring_style": true 或 false,\n'
        '  "scoring_style": "若不更新则原样复述现有值",\n'
        '  "update_three_vs_four": true 或 false,\n'
        '  "three_vs_four_distinction": "若不更新则原样复述现有值",\n'
        '  "update_four_vs_five": true 或 false,\n'
        '  "four_vs_five_distinction": "若不更新则原样复述现有值",\n'
        '  "add_user_specific_requirements": ["仅新增条目；若无则空列表"],\n'
        '  "update_preferred_response_format": true 或 false,\n'
        '  "preferred_response_format": "若不更新则原样复述现有值",\n'
        '  "add_or_update_task_specific_observations": [{"task_name": "...", "observation": "..."}],\n'
        '  "rationale": "1-3句说明主要依据与保持不变的原因"\n'
        "}\n"
    )
    return prompt


def build_memory_update_prompt_v2_4(
    existing_memory: UserMemory,
    new_session: SessionData,
    turn_predictions: list[dict],
    use_oracle_labels: bool = False,
) -> str:
    """
    构造 memory update prompt（v2.4）。

    设计目标：
      - 以 v2.1 的原始 turn 级证据和 patch schema 为基础
      - 不引入 v2.2 的 evidence bundle
      - 不引入 v2.3 的额外边界摘要，避免过度拉向 DSAT
      - 仅强调无相邻证据时不改 verbal boundary
    """
    existing_json = existing_memory.model_dump_json(
        indent=2,
        exclude={"memory_version", "source_tasks", "n_history_sessions", "n_history_turns"},
    )
    turns = _collect_update_turns(new_session, turn_predictions, use_oracle_labels)
    by_score: dict[int, list[dict]] = defaultdict(list)
    for t in turns:
        by_score[t["score"]].append(t)

    low_turns = by_score[1] + by_score[2] + by_score[3]
    mid_turns = by_score[4]
    high_turns = by_score[5]

    session_stats = (
        f"本 session 分布："
        f"5分×{len(by_score[5])} / 4分×{len(by_score[4])} / 3分×{len(by_score[3])} / "
        f"2分×{len(by_score[2])} / 1分×{len(by_score[1])}"
    )
    label_note = (
        "本次提供了真实标签，可视为可靠证据。"
        if use_oracle_labels
        else "本次仅有模型预测标签，属于弱证据；请保持 v2.1 的校准收益，但不要把单边样本升级成新的硬边界规则。"
    )
    prompt = (
        "你正在维护一份 v2.4 用户记忆。请根据新 session 给出【字段级 patch】，而不是重写整份 memory。\n\n"
        f"【现有记忆】\n{existing_json}\n\n"
        f"【新 Session 概览】\n"
        f"任务：{new_session.task}\n"
        f"任务背景：{_truncate(new_session.task_context, 180)}\n"
        f"{session_stats}\n"
        f"【说明】{label_note}\n\n"
        f"{_format_update_examples('【<=3 证据（可能影响 3/4 边界或低分要求）】', low_turns)}\n\n"
        f"{_format_update_examples('【4 分证据（与 <=3 或 5 比较时使用）】', mid_turns)}\n\n"
        f"{_format_update_examples('【5 分证据（可能影响 4/5 边界）】', high_turns)}\n\n"
        "更新原则：\n"
        "1. avg_satisfaction_score 和 score_distribution 由程序自动更新，你不需要负责统计数字。\n"
        "2. 尽量保持 v2.1 的轻量 patch 风格：只在有新信息时更新，不做大幅重写。\n"
        "3. three_vs_four_distinction 只有在本 session 同时存在 3 分和 4 分证据时才建议改写；"
        "若只有 <=3 或只有 4 分，请保持原样。\n"
        "4. four_vs_five_distinction 只有在本 session 同时存在 4 分和 5 分证据时才建议改写；"
        "若只有单边证据，请保持原样。\n"
        "5. 不要为了提高不满意识别而系统性压低分数；边界文字只能描述证据中真实出现的差异。\n"
        "6. user_specific_requirements 只能新增真正会改变评分的个性化要求；"
        "禁止加入“更详细、更具体、更清晰、更结构化、更实用”等泛化要求。\n"
        "7. preferred_response_format 只有在出现新的稳定格式偏好时才改。\n"
        "8. task_specific_observations 只新增/覆盖当前 session 提供了明确新信息的任务观察。\n\n"
        "请严格按下面的 JSON Schema 输出 patch，不要输出其他内容：\n"
        "{\n"
        '  "update_scoring_style": true 或 false,\n'
        '  "scoring_style": "若不更新则原样复述现有值",\n'
        '  "update_three_vs_four": true 或 false,\n'
        '  "three_vs_four_distinction": "若不更新则原样复述现有值",\n'
        '  "update_four_vs_five": true 或 false,\n'
        '  "four_vs_five_distinction": "若不更新则原样复述现有值",\n'
        '  "add_user_specific_requirements": ["仅新增条目；若无则空列表"],\n'
        '  "update_preferred_response_format": true 或 false,\n'
        '  "preferred_response_format": "若不更新则原样复述现有值",\n'
        '  "add_or_update_task_specific_observations": [{"task_name": "...", "observation": "..."}],\n'
        '  "rationale": "1-3句说明主要依据与保持不变的原因"\n'
        "}\n"
    )
    return prompt


def build_memory_update_prompt_v2_5(
    existing_memory: UserMemory,
    new_session: SessionData,
    turn_predictions: list[dict],
    use_oracle_labels: bool = False,
) -> str:
    """
    构造 memory update prompt（v2.5）。

    设计目标：
      - 继续使用 v2.1 的原始 turn 级证据与 patch schema
      - 不把 SAT drift 问题交给 verbal boundary 文本解决
      - 明确要求非 oracle update 不改 scoring_style
      - 统计均值的上移由 merge 侧轻量阻尼
    """
    existing_json = existing_memory.model_dump_json(
        indent=2,
        exclude={"memory_version", "source_tasks", "n_history_sessions", "n_history_turns"},
    )
    turns = _collect_update_turns(new_session, turn_predictions, use_oracle_labels)
    by_score: dict[int, list[dict]] = defaultdict(list)
    for t in turns:
        by_score[t["score"]].append(t)

    low_turns = by_score[1] + by_score[2] + by_score[3]
    mid_turns = by_score[4]
    high_turns = by_score[5]

    session_stats = (
        f"本 session 分布："
        f"5分×{len(by_score[5])} / 4分×{len(by_score[4])} / 3分×{len(by_score[3])} / "
        f"2分×{len(by_score[2])} / 1分×{len(by_score[1])}"
    )
    label_note = (
        "本次提供了真实标签，可视为可靠证据。"
        if use_oracle_labels
        else "本次仅有模型预测标签，属于弱证据；请不要根据预测标签改写 scoring_style 或提高用户整体宽松程度。"
    )
    prompt = (
        "你正在维护一份 v2.5 用户记忆。请根据新 session 给出【字段级 patch】，而不是重写整份 memory。\n\n"
        f"【现有记忆】\n{existing_json}\n\n"
        f"【新 Session 概览】\n"
        f"任务：{new_session.task}\n"
        f"任务背景：{_truncate(new_session.task_context, 180)}\n"
        f"{session_stats}\n"
        f"【说明】{label_note}\n\n"
        f"{_format_update_examples('【<=3 证据（可能影响 3/4 边界或低分要求）】', low_turns)}\n\n"
        f"{_format_update_examples('【4 分证据（与 <=3 或 5 比较时使用）】', mid_turns)}\n\n"
        f"{_format_update_examples('【5 分证据（可能影响 4/5 边界）】', high_turns)}\n\n"
        "更新原则：\n"
        "1. avg_satisfaction_score 和 score_distribution 由程序自动更新，你不需要负责统计数字。\n"
        "2. 非 oracle 场景下，scoring_style 默认保持原样；除非说明中明确写着真实标签可靠，否则 update_scoring_style 必须为 false。\n"
        "3. 不要因为本 session 预测分数偏高，就把用户描述成更宽松、更容易满意。\n"
        "4. three_vs_four_distinction 只有在出现清晰的 <=3 与 4 分对比证据时才改写。\n"
        "5. four_vs_five_distinction 只有在出现清晰的 4 与 5 分对比证据时才改写。\n"
        "6. user_specific_requirements 只能新增真正会改变评分的个性化要求；禁止加入泛化要求。\n"
        "7. preferred_response_format 只有在出现新的稳定格式偏好时才改。\n"
        "8. task_specific_observations 只新增/覆盖当前 session 提供了明确新信息的任务观察。\n\n"
        "请严格按下面的 JSON Schema 输出 patch，不要输出其他内容：\n"
        "{\n"
        '  "update_scoring_style": true 或 false,\n'
        '  "scoring_style": "若不更新则原样复述现有值",\n'
        '  "update_three_vs_four": true 或 false,\n'
        '  "three_vs_four_distinction": "若不更新则原样复述现有值",\n'
        '  "update_four_vs_five": true 或 false,\n'
        '  "four_vs_five_distinction": "若不更新则原样复述现有值",\n'
        '  "add_user_specific_requirements": ["仅新增条目；若无则空列表"],\n'
        '  "update_preferred_response_format": true 或 false,\n'
        '  "preferred_response_format": "若不更新则原样复述现有值",\n'
        '  "add_or_update_task_specific_observations": [{"task_name": "...", "observation": "..."}],\n'
        '  "rationale": "1-3句说明主要依据与保持不变的原因"\n'
        "}\n"
    )
    return prompt


def build_memory_update_prompt_v2_2(
    existing_memory: UserMemory,
    new_session: SessionData,
    turn_predictions: list[dict],
    use_oracle_labels: bool = False,
) -> str:
    """
    构造 memory update prompt（v2.2）。

    核心改动：
      1. 输入改成结构化 evidence bundle
      2. 要求模型输出 patch 时同时给出 confidence / support_count
      3. 让 non-oracle update 看到不确定性信号，避免把 noisy session 直接升级成硬规则
    """
    existing_json = existing_memory.model_dump_json(
        indent=2,
        exclude={"memory_version", "source_tasks", "n_history_sessions", "n_history_turns"},
    )
    evidence_bundle = _build_update_evidence_bundle(
        existing_memory=existing_memory,
        new_session=new_session,
        turn_predictions=turn_predictions,
        use_oracle_labels=use_oracle_labels,
    )
    evidence_json = json.dumps(evidence_bundle, ensure_ascii=False, indent=2)
    label_note = (
        "本次提供了真实标签，可将 evidence 视作强证据。"
        if use_oracle_labels
        else "本次 only 有模型预测标签。除非 evidence bundle 显示出稳定、清晰的边界模式，否则不要更新 verbal 边界字段。"
    )
    prompt = (
        "你正在维护一份 v2.2 用户记忆。请根据【结构化 evidence bundle】输出字段级 patch。\n\n"
        f"【现有记忆】\n{existing_json}\n\n"
        f"【Evidence Bundle】\n{evidence_json}\n\n"
        f"【说明】{label_note}\n\n"
        "更新原则：\n"
        "1. avg_satisfaction_score 和 score_distribution 由程序自动更新，你不需要负责统计数字。\n"
        "2. three_vs_four_distinction 只有在 evidence bundle 同时出现清晰的 <=3 与 4 分证据时才允许更新。\n"
        "3. four_vs_five_distinction 只有在 evidence bundle 同时出现清晰的 4 与 5 分证据时才允许更新。\n"
        "4. 若 uncertainty_signals 表明本 session 边界不稳定（例如大量 borderline mentions、频繁 3/4 或 4/5 flip），应降低 confidence，而不是勉强改写规则。\n"
        "5. user_specific_requirements 只新增真正会改变评分的个性化要求；像“更详细、更具体、更清晰”这类泛化词不要加入。\n"
        "6. preferred_response_format 只有在 evidence 明确显示稳定格式偏好时才更新；单个 session 默认不改。\n"
        "7. task_specific_observations 只对当前 session 明确提供新信息的任务给出 patch。\n\n"
        "请严格输出 v2.2 patch JSON，不要输出其他内容。字段要求：\n"
        "{\n"
        '  "update_scoring_style": true/false,\n'
        '  "scoring_style": "...",\n'
        '  "scoring_style_confidence": "low|medium|high",\n'
        '  "update_three_vs_four": true/false,\n'
        '  "three_vs_four_distinction": "...",\n'
        '  "three_vs_four_confidence": "low|medium|high",\n'
        '  "three_vs_four_evidence_count": 0-10,\n'
        '  "update_four_vs_five": true/false,\n'
        '  "four_vs_five_distinction": "...",\n'
        '  "four_vs_five_confidence": "low|medium|high",\n'
        '  "four_vs_five_evidence_count": 0-10,\n'
        '  "add_user_specific_requirements": [{"requirement":"...","confidence":"low|medium|high","support_count":0-10}],\n'
        '  "update_preferred_response_format": true/false,\n'
        '  "preferred_response_format": "...",\n'
        '  "preferred_response_format_confidence": "low|medium|high",\n'
        '  "preferred_response_format_support_count": 0-10,\n'
        '  "add_or_update_task_specific_observations": [{"task_name":"...","observation":"...","confidence":"low|medium|high","support_count":0-10}],\n'
        '  "rationale": "1-4 句说明哪些字段有足够 evidence，哪些字段应保持不变"\n'
        "}\n"
    )
    return prompt


def build_memory_update_prompt_v2_3(
    existing_memory: UserMemory,
    new_session: SessionData,
    turn_predictions: list[dict],
    use_oracle_labels: bool = False,
) -> str:
    """
    构造 memory update prompt（v2.3）。

    设计目标：
      - 回到 v2.1 的原始 turn 级证据输入
      - 只额外补充轻量边界摘要
      - 不再像 v2.2 那样把证据重压缩成抽象 bundle
    """
    existing_json = existing_memory.model_dump_json(
        indent=2,
        exclude={"memory_version", "source_tasks", "n_history_sessions", "n_history_turns"},
    )
    turns = _collect_update_turns(new_session, turn_predictions, use_oracle_labels)
    by_score: dict[int, list[dict]] = defaultdict(list)
    for t in turns:
        by_score[t["score"]].append(t)

    low_turns = by_score[1] + by_score[2] + by_score[3]
    mid_turns = by_score[4]
    high_turns = by_score[5]
    has_3_and_4 = bool(by_score[3] and by_score[4])
    has_4_and_5 = bool(by_score[4] and by_score[5])

    session_stats = (
        f"本 session 分布："
        f"5分×{len(by_score[5])} / 4分×{len(by_score[4])} / 3分×{len(by_score[3])} / "
        f"2分×{len(by_score[2])} / 1分×{len(by_score[1])}"
    )
    label_note = (
        "本次提供了真实标签，可视为可靠证据。"
        if use_oracle_labels
        else "本次仅有模型预测标签。请优先相信那些形成清晰 3/4 或 4/5 对比的 turn；"
             "若只是单边高分或单边低分，不要轻易改写长期边界规则。"
    )
    boundary_summary = (
        "【边界样本摘要】\n"
        f"- 是否同时出现 3 分和 4 分：{'是' if has_3_and_4 else '否'}\n"
        f"- 是否同时出现 4 分和 5 分：{'是' if has_4_and_5 else '否'}\n"
        f"- <=3 样本数：{len(low_turns)}\n"
        f"- 4 分样本数：{len(mid_turns)}\n"
        f"- 5 分样本数：{len(high_turns)}\n"
        "- 若没有相邻分数证据，只能微调 calibration 或补充 requirement，不能把边界文字改成新硬规则。"
    )

    prompt = (
        "你正在维护一份 v2.3 用户记忆。请根据新 session 给出【字段级 patch】，不要重写整份 memory。\n\n"
        f"【现有记忆】\n{existing_json}\n\n"
        f"【新 Session 概览】\n"
        f"任务：{new_session.task}\n"
        f"任务背景：{_truncate(new_session.task_context, 180)}\n"
        f"{session_stats}\n"
        f"【说明】{label_note}\n\n"
        f"{boundary_summary}\n\n"
        f"{_format_update_examples('【<=3 证据（可能影响 3/4 边界或低分要求）】', low_turns)}\n\n"
        f"{_format_update_examples('【4 分证据（与 <=3 或 5 比较时使用）】', mid_turns)}\n\n"
        f"{_format_update_examples('【5 分证据（可能影响 4/5 边界）】', high_turns)}\n\n"
        "更新原则：\n"
        "1. avg_satisfaction_score 和 score_distribution 由程序自动更新，你不需要负责统计数字。\n"
        "2. three_vs_four_distinction 只有在新 session 同时出现清晰的 3 分与 4 分证据时才建议改写；否则保持原样。\n"
        "3. four_vs_five_distinction 只有在新 session 同时出现清晰的 4 分与 5 分证据时才建议改写；否则保持原样。\n"
        "4. 若只是重复现有模式，必须保持 verbal 字段不变。\n"
        "5. user_specific_requirements 只新增真正会改变评分的个性化要求；不要加入“更详细、更具体、更清晰”这类泛化要求。\n"
        "6. preferred_response_format 只有在出现新的稳定格式偏好时才改。\n"
        "7. task_specific_observations 只新增/覆盖当前 session 提供了明确新信息的任务观察。\n\n"
        "请严格按下面的 JSON Schema 输出 patch，不要输出其他内容：\n"
        "{\n"
        '  "update_scoring_style": true 或 false,\n'
        '  "scoring_style": "若不更新则原样复述现有值",\n'
        '  "update_three_vs_four": true 或 false,\n'
        '  "three_vs_four_distinction": "若不更新则原样复述现有值",\n'
        '  "update_four_vs_five": true 或 false,\n'
        '  "four_vs_five_distinction": "若不更新则原样复述现有值",\n'
        '  "add_user_specific_requirements": ["仅新增条目；若无则空列表"],\n'
        '  "update_preferred_response_format": true 或 false,\n'
        '  "preferred_response_format": "若不更新则原样复述现有值",\n'
        '  "add_or_update_task_specific_observations": [{"task_name": "...", "observation": "..."}],\n'
        '  "rationale": "1-3句说明主要依据与保持不变的原因"\n'
        "}\n"
    )
    return prompt


def merge_memory_v2_1_patch(
    existing_memory: UserMemory,
    patch: MemoryUpdatePatchV2_1,
    turn_predictions: list[dict],
    use_oracle_labels: bool = False,
) -> UserMemory:
    """将 v2.1 patch 合并回 v2 memory，并由代码端更新统计量。"""
    scores = [
        int(pred["gold_score"] if use_oracle_labels else pred["pred_score"])
        for pred in turn_predictions
    ]
    old_counts = existing_memory.score_distribution.model_copy()
    counts_map = {
        1: old_counts.score_1,
        2: old_counts.score_2,
        3: old_counts.score_3,
        4: old_counts.score_4,
        5: old_counts.score_5,
    }
    for s in scores:
        counts_map[s] += 1
    new_dist = ScoreDistribution(
        score_1=counts_map[1],
        score_2=counts_map[2],
        score_3=counts_map[3],
        score_4=counts_map[4],
        score_5=counts_map[5],
    )
    old_total = sum(_score_distribution_to_counts(existing_memory.score_distribution).values())
    new_total = old_total + len(scores)
    weighted_sum = existing_memory.avg_satisfaction_score * old_total + sum(scores)
    new_avg = weighted_sum / new_total if new_total else existing_memory.avg_satisfaction_score

    reqs = _dedupe_preserve_order(
        list(existing_memory.user_specific_requirements)
        + list(patch.add_user_specific_requirements)
    )[:5]

    task_obs_map = {
        obs.task_name: obs.model_copy()
        for obs in existing_memory.task_specific_observations
    }
    for obs in patch.add_or_update_task_specific_observations:
        task_obs_map[obs.task_name] = obs

    return UserMemory(
        avg_satisfaction_score=round(new_avg, 4),
        score_distribution=new_dist,
        scoring_style=patch.scoring_style if patch.update_scoring_style else existing_memory.scoring_style,
        four_vs_five_distinction=(
            patch.four_vs_five_distinction
            if patch.update_four_vs_five else existing_memory.four_vs_five_distinction
        ),
        three_vs_four_distinction=(
            patch.three_vs_four_distinction
            if patch.update_three_vs_four else existing_memory.three_vs_four_distinction
        ),
        user_specific_requirements=reqs,
        preferred_response_format=(
            patch.preferred_response_format
            if patch.update_preferred_response_format else existing_memory.preferred_response_format
        ),
        task_specific_observations=list(task_obs_map.values()),
        memory_version="v2",
        source_tasks=list(existing_memory.source_tasks),
        n_history_sessions=existing_memory.n_history_sessions + 1,
        n_history_turns=existing_memory.n_history_turns + len(turn_predictions),
    )


def merge_memory_v2_2_patch(
    existing_memory: UserMemory,
    patch: MemoryUpdatePatchV2_2,
    turn_predictions: list[dict],
    use_oracle_labels: bool = False,
) -> UserMemory:
    """将 v2.2 patch 合并回 v2 memory，并用更硬的程序 gate 控制 verbal 字段更新。"""
    scores = [
        int(pred["gold_score"] if use_oracle_labels else pred["pred_score"])
        for pred in turn_predictions
    ]
    old_counts = existing_memory.score_distribution.model_copy()
    counts_map = {
        1: old_counts.score_1,
        2: old_counts.score_2,
        3: old_counts.score_3,
        4: old_counts.score_4,
        5: old_counts.score_5,
    }
    for s in scores:
        counts_map[s] += 1
    new_dist = ScoreDistribution(
        score_1=counts_map[1],
        score_2=counts_map[2],
        score_3=counts_map[3],
        score_4=counts_map[4],
        score_5=counts_map[5],
    )
    old_total = sum(_score_distribution_to_counts(existing_memory.score_distribution).values())
    new_total = old_total + len(scores)
    weighted_sum = existing_memory.avg_satisfaction_score * old_total + sum(scores)
    new_avg = weighted_sum / new_total if new_total else existing_memory.avg_satisfaction_score

    new_score_counts = {s: scores.count(s) for s in range(1, 6)}
    has_3_and_4 = new_score_counts[3] > 0 and new_score_counts[4] > 0
    has_4_and_5 = new_score_counts[4] > 0 and new_score_counts[5] > 0

    allow_scoring_style = (
        patch.update_scoring_style
        and patch.scoring_style_confidence in {"medium", "high"}
        and len(scores) >= 3
    )
    allow_three_vs_four = (
        patch.update_three_vs_four
        and has_3_and_4
        and patch.three_vs_four_confidence in {"medium", "high"}
        and patch.three_vs_four_evidence_count >= 2
    )
    allow_four_vs_five = (
        patch.update_four_vs_five
        and has_4_and_5
        and patch.four_vs_five_confidence in {"medium", "high"}
        and patch.four_vs_five_evidence_count >= 2
    )
    allow_preferred_format = (
        patch.update_preferred_response_format
        and patch.preferred_response_format_confidence == "high"
        and patch.preferred_response_format_support_count >= 2
    )

    reqs = list(existing_memory.user_specific_requirements)
    for item in patch.add_user_specific_requirements:
        if item.confidence == "low" or item.support_count < 2:
            continue
        if _is_generic_requirement(item.requirement):
            continue
        reqs.append(item.requirement)
    reqs = _dedupe_preserve_order(reqs)[:5]

    task_obs_map = {
        obs.task_name: obs.model_copy()
        for obs in existing_memory.task_specific_observations
    }
    for obs in patch.add_or_update_task_specific_observations:
        if obs.confidence == "low" or obs.support_count < 1:
            continue
        task_obs_map[obs.task_name] = TaskObservation(
            task_name=obs.task_name,
            observation=obs.observation,
        )

    return UserMemory(
        avg_satisfaction_score=round(new_avg, 4),
        score_distribution=new_dist,
        scoring_style=patch.scoring_style if allow_scoring_style else existing_memory.scoring_style,
        four_vs_five_distinction=(
            patch.four_vs_five_distinction
            if allow_four_vs_five else existing_memory.four_vs_five_distinction
        ),
        three_vs_four_distinction=(
            patch.three_vs_four_distinction
            if allow_three_vs_four else existing_memory.three_vs_four_distinction
        ),
        user_specific_requirements=reqs,
        preferred_response_format=(
            patch.preferred_response_format
            if allow_preferred_format else existing_memory.preferred_response_format
        ),
        task_specific_observations=list(task_obs_map.values()),
        memory_version="v2",
        source_tasks=list(existing_memory.source_tasks),
        n_history_sessions=existing_memory.n_history_sessions + 1,
        n_history_turns=existing_memory.n_history_turns + len(turn_predictions),
    )


def merge_memory_v2_3_patch(
    existing_memory: UserMemory,
    patch: MemoryUpdatePatchV2_1,
    turn_predictions: list[dict],
    use_oracle_labels: bool = False,
) -> UserMemory:
    """
    将 v2.3 patch 合并回 v2 memory。

    相比 v2.1：
      - 保留 field-level patch 框架
      - 只增加轻量 boundary gate
      - 继续过滤泛化 requirement
    """
    scores = [
        int(pred["gold_score"] if use_oracle_labels else pred["pred_score"])
        for pred in turn_predictions
    ]
    old_counts = existing_memory.score_distribution.model_copy()
    counts_map = {
        1: old_counts.score_1,
        2: old_counts.score_2,
        3: old_counts.score_3,
        4: old_counts.score_4,
        5: old_counts.score_5,
    }
    for s in scores:
        counts_map[s] += 1
    new_dist = ScoreDistribution(
        score_1=counts_map[1],
        score_2=counts_map[2],
        score_3=counts_map[3],
        score_4=counts_map[4],
        score_5=counts_map[5],
    )
    old_total = sum(_score_distribution_to_counts(existing_memory.score_distribution).values())
    new_total = old_total + len(scores)
    weighted_sum = existing_memory.avg_satisfaction_score * old_total + sum(scores)
    new_avg = weighted_sum / new_total if new_total else existing_memory.avg_satisfaction_score

    new_score_counts = {s: scores.count(s) for s in range(1, 6)}
    has_3_and_4 = new_score_counts[3] > 0 and new_score_counts[4] > 0
    has_4_and_5 = new_score_counts[4] > 0 and new_score_counts[5] > 0

    reqs = list(existing_memory.user_specific_requirements)
    for item in patch.add_user_specific_requirements:
        if _is_generic_requirement(item):
            continue
        reqs.append(item)
    reqs = _dedupe_preserve_order(reqs)[:5]

    task_obs_map = {
        obs.task_name: obs.model_copy()
        for obs in existing_memory.task_specific_observations
    }
    for obs in patch.add_or_update_task_specific_observations:
        task_obs_map[obs.task_name] = obs

    return UserMemory(
        avg_satisfaction_score=round(new_avg, 4),
        score_distribution=new_dist,
        scoring_style=patch.scoring_style if patch.update_scoring_style else existing_memory.scoring_style,
        four_vs_five_distinction=(
            patch.four_vs_five_distinction
            if patch.update_four_vs_five and has_4_and_5
            else existing_memory.four_vs_five_distinction
        ),
        three_vs_four_distinction=(
            patch.three_vs_four_distinction
            if patch.update_three_vs_four and has_3_and_4
            else existing_memory.three_vs_four_distinction
        ),
        user_specific_requirements=reqs,
        preferred_response_format=(
            patch.preferred_response_format
            if patch.update_preferred_response_format else existing_memory.preferred_response_format
        ),
        task_specific_observations=list(task_obs_map.values()),
        memory_version="v2",
        source_tasks=list(existing_memory.source_tasks),
        n_history_sessions=existing_memory.n_history_sessions + 1,
        n_history_turns=existing_memory.n_history_turns + len(turn_predictions),
    )


def merge_memory_v2_4_patch(
    existing_memory: UserMemory,
    patch: MemoryUpdatePatchV2_1,
    turn_predictions: list[dict],
    use_oracle_labels: bool = False,
) -> UserMemory:
    """
    将 v2.4 patch 合并回 v2 memory。

    相比 v2.1 只做轻量保护：
      - 统计量更新完全保持 v2.1
      - scoring_style / preferred_response_format / task observations 保持 v2.1
      - boundary 字段要求相邻分数证据
      - requirement 过滤泛化文本
    """
    scores = [
        int(pred["gold_score"] if use_oracle_labels else pred["pred_score"])
        for pred in turn_predictions
    ]
    old_counts = existing_memory.score_distribution.model_copy()
    counts_map = {
        1: old_counts.score_1,
        2: old_counts.score_2,
        3: old_counts.score_3,
        4: old_counts.score_4,
        5: old_counts.score_5,
    }
    for s in scores:
        counts_map[s] += 1
    new_dist = ScoreDistribution(
        score_1=counts_map[1],
        score_2=counts_map[2],
        score_3=counts_map[3],
        score_4=counts_map[4],
        score_5=counts_map[5],
    )
    old_total = sum(_score_distribution_to_counts(existing_memory.score_distribution).values())
    new_total = old_total + len(scores)
    weighted_sum = existing_memory.avg_satisfaction_score * old_total + sum(scores)
    new_avg = weighted_sum / new_total if new_total else existing_memory.avg_satisfaction_score

    new_score_counts = {s: scores.count(s) for s in range(1, 6)}
    has_3_and_4 = new_score_counts[3] > 0 and new_score_counts[4] > 0
    has_4_and_5 = new_score_counts[4] > 0 and new_score_counts[5] > 0

    reqs = list(existing_memory.user_specific_requirements)
    for item in patch.add_user_specific_requirements:
        if _is_generic_requirement(item):
            continue
        reqs.append(item)
    reqs = _dedupe_preserve_order(reqs)[:5]

    task_obs_map = {
        obs.task_name: obs.model_copy()
        for obs in existing_memory.task_specific_observations
    }
    for obs in patch.add_or_update_task_specific_observations:
        task_obs_map[obs.task_name] = obs

    return UserMemory(
        avg_satisfaction_score=round(new_avg, 4),
        score_distribution=new_dist,
        scoring_style=patch.scoring_style if patch.update_scoring_style else existing_memory.scoring_style,
        four_vs_five_distinction=(
            patch.four_vs_five_distinction
            if patch.update_four_vs_five and has_4_and_5
            else existing_memory.four_vs_five_distinction
        ),
        three_vs_four_distinction=(
            patch.three_vs_four_distinction
            if patch.update_three_vs_four and has_3_and_4
            else existing_memory.three_vs_four_distinction
        ),
        user_specific_requirements=reqs,
        preferred_response_format=(
            patch.preferred_response_format
            if patch.update_preferred_response_format else existing_memory.preferred_response_format
        ),
        task_specific_observations=list(task_obs_map.values()),
        memory_version="v2",
        source_tasks=list(existing_memory.source_tasks),
        n_history_sessions=existing_memory.n_history_sessions + 1,
        n_history_turns=existing_memory.n_history_turns + len(turn_predictions),
    )


def merge_memory_v2_5_patch(
    existing_memory: UserMemory,
    patch: MemoryUpdatePatchV2_1,
    turn_predictions: list[dict],
    use_oracle_labels: bool = False,
) -> UserMemory:
    """
    将 v2.5 patch 合并回 v2 memory。

    相比 v2.1：
      - 非 oracle 时冻结 scoring_style
      - 非 oracle 时对 avg_satisfaction_score 的上移做轻量阻尼
      - requirement 过滤泛化文本
      - 其他字段保持 v2.1 的轻量 patch 行为
    """
    scores = [
        int(pred["gold_score"] if use_oracle_labels else pred["pred_score"])
        for pred in turn_predictions
    ]
    old_counts = existing_memory.score_distribution.model_copy()
    counts_map = {
        1: old_counts.score_1,
        2: old_counts.score_2,
        3: old_counts.score_3,
        4: old_counts.score_4,
        5: old_counts.score_5,
    }
    for s in scores:
        counts_map[s] += 1
    new_dist = ScoreDistribution(
        score_1=counts_map[1],
        score_2=counts_map[2],
        score_3=counts_map[3],
        score_4=counts_map[4],
        score_5=counts_map[5],
    )
    old_total = sum(_score_distribution_to_counts(existing_memory.score_distribution).values())
    new_total = old_total + len(scores)
    weighted_sum = existing_memory.avg_satisfaction_score * old_total + sum(scores)
    raw_new_avg = weighted_sum / new_total if new_total else existing_memory.avg_satisfaction_score
    if use_oracle_labels or raw_new_avg <= existing_memory.avg_satisfaction_score:
        new_avg = raw_new_avg
    else:
        # Predicted-label updates often drift SAT-heavy; allow only a damped upward prior shift.
        new_avg = existing_memory.avg_satisfaction_score + 0.25 * (
            raw_new_avg - existing_memory.avg_satisfaction_score
        )

    reqs = list(existing_memory.user_specific_requirements)
    for item in patch.add_user_specific_requirements:
        if _is_generic_requirement(item):
            continue
        reqs.append(item)
    reqs = _dedupe_preserve_order(reqs)[:5]

    task_obs_map = {
        obs.task_name: obs.model_copy()
        for obs in existing_memory.task_specific_observations
    }
    for obs in patch.add_or_update_task_specific_observations:
        task_obs_map[obs.task_name] = obs

    return UserMemory(
        avg_satisfaction_score=round(new_avg, 4),
        score_distribution=new_dist,
        scoring_style=(
            patch.scoring_style
            if patch.update_scoring_style and use_oracle_labels
            else existing_memory.scoring_style
        ),
        four_vs_five_distinction=(
            patch.four_vs_five_distinction
            if patch.update_four_vs_five else existing_memory.four_vs_five_distinction
        ),
        three_vs_four_distinction=(
            patch.three_vs_four_distinction
            if patch.update_three_vs_four else existing_memory.three_vs_four_distinction
        ),
        user_specific_requirements=reqs,
        preferred_response_format=(
            patch.preferred_response_format
            if patch.update_preferred_response_format else existing_memory.preferred_response_format
        ),
        task_specific_observations=list(task_obs_map.values()),
        memory_version="v2",
        source_tasks=list(existing_memory.source_tasks),
        n_history_sessions=existing_memory.n_history_sessions + 1,
        n_history_turns=existing_memory.n_history_turns + len(turn_predictions),
    )


def build_memory_update_prompt_v3(
    existing_memory: UserMemoryV3,
    new_session: SessionData,
    turn_predictions: list[dict],
    use_oracle_labels: bool = False,
) -> str:
    """
    构造 memory update prompt（v3）。

    相比 v2，v3 更强调：
      1. 不要在证据不足时把弱推断改写成硬规则
      2. 优先更新 calibration 与具体反例
      3. 只有新证据明确时才改边界文本
    """
    existing_json = existing_memory.model_dump_json(
        indent=2,
        exclude={"source_tasks", "n_history_sessions", "n_history_turns"},
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
        else "本次仅有模型预测分数（可能有误），请谨慎参考，不要把弱证据升级成硬规则。"
    )

    prompt = (
        "你正在维护一份 v3 用户记忆。请根据新观察到的 session 决定是否需要更新记忆。\n\n"
        f"【现有记忆】\n{existing_json}\n\n"
        f"【新 Session】\n{session_text}\n\n"
        f"【注意】{label_note}\n\n"
        "更新原则：\n"
        "- 优先更新 avg_satisfaction_score / score_distribution 这类 calibration 信息\n"
        "- three_vs_four_distinction / four_vs_five_distinction 只有在新 session 提供了明确相邻分数反例时才修改\n"
        "- 若原本属于“证据不足、只能弱推断”的边界，不要因为单条可疑预测就写成确定性规则\n"
        "- user_specific_requirements 只保留真正改变评分的个性化要求，不要累积通用偏好文本\n"
        "- preferred_response_format 若没有新增信息，可保持极简\n\n"
        "请严格按照原 JSON Schema 输出更新后的记忆（不含程序侧元信息字段），不要输出其他内容。"
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
        "v2", "v3", "v3_1", "history_prior_delta", "qwen_short", "boundary_34", "boundary_34_refute", "boundary_34_refute_v2",
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
      - "v3": 分离 calibration 与 boundary 规则；证据不足时弱化边界总结
      - "v3_1": 在 v3 基础上重新加硬 3/4 最低满意线；证据不足只影响 1/2/3 细分，不放松 SAT gate
      - "history_prior_delta": 显式以历史均分为 prior，先判 residual delta 与 3/4 boundary，再重建最终分
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

    if prompt_version == "history_prior_delta":
        anchor_instruction = (
            "【参考案例使用规则】\n"
            "1. 参考案例只用于判断当前回复相对历史先验是更差、相近还是更好。\n"
            "2. 优先找整体质量和当前回复最接近的案例，比较它与该用户历史平均水平的相对位置。\n"
            "3. 不要直接复制案例分数；本题先输出 residual delta，再由程序转回最终分。\n\n"
            if anchor_turns else ""
        )
        prompt = (
            "你是一名个性化满意度 residual judge。请不要直接自由打 1-5 分。\n"
            "本题必须先以该用户的 history prior 为起点，判断当前回复相对先验的 delta，"
            "再判断是否通过 3/4 满意边界。程序会根据你输出的 prior + delta 和 boundary 重建最终分。\n\n"
            f"【History Prior（必须作为起点）】\n"
            f"history_prior_score = {memory.avg_satisfaction_score:.2f}\n"
            f"历史分布：5分×{memory.score_distribution.score_5} / "
            f"4分×{memory.score_distribution.score_4} / "
            f"3分×{memory.score_distribution.score_3} / "
            f"2分×{memory.score_distribution.score_2} / "
            f"1分×{memory.score_distribution.score_1}\n"
            f"评分风格：{memory.scoring_style}\n\n"
            f"【个性化边界】\n"
            f"3→4 满意最低线：{memory.three_vs_four_distinction}\n"
            f"4→5 更高要求：{memory.four_vs_five_distinction}\n\n"
            f"【真正影响评分的个性化要求】\n{user_reqs}\n"
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
            + "【判断步骤】\n"
            + "Step 1. 固定 history_prior_score：直接使用上面给出的历史平均分，不要自行改写。\n"
            + "Step 2. 判断 residual delta：当前回复相对该用户通常得到的回复，是 below / around / above？\n"
            + "  - delta_score=-2：明显低于该用户历史常态，存在严重关键失败。\n"
            + "  - delta_score=-1：低于常态，有一个清楚的关键缺口或可用性损失。\n"
            + "  - delta_score=0：大致符合该用户历史常态。\n"
            + "  - delta_score=1：高于常态，明显更贴合需求或更完整。\n"
            + "  - delta_score=2：显著高于常态，接近该用户历史中的强满意样本。\n"
            + "Step 3. 单独判断 3/4 boundary：当前回复是否达到该用户的最低满意线？\n"
            + "  - 若核心问题没被回答、关键约束被忽略、内容空泛到影响使用，passes_satisfaction_boundary=false，boundary_score=3。\n"
            + "  - 若核心问题已回答且关键要求基本满足，passes_satisfaction_boundary=true，boundary_score=4。\n"
            + "Step 4. 最终分由程序重建：round(history_prior_score + delta_score) 后裁剪到 1-5；"
            + "若 boundary_score=3 则最终不超过3，若 boundary_score=4 则最终不低于4。\n"
            + "你仍需在 classification 中填入你按此规则得到的最终分，方便诊断。\n"
            + "Step 5. 选择 reason：若最终分 >=4，reason 必须是 `满意`；若最终分 <=3，reason 必须是不满意原因。\n\n"
            + "注意：\n"
            + "- history prior 解释用户整体偏高分或偏低分，delta 才解释当前回复比常态好/差。\n"
            + "- boundary 是硬约束：没过最低满意线时，即使 prior 很高也不能给 4/5；已过最低满意线时不能给 1/2/3。\n"
            + "- 不要把 5 分门槛误当成 4 分门槛；不够完美不等于未满意。\n\n"
            + "请严格输出 JSON，不要输出其他内容：\n"
            + "{\n"
            + '  "classification": 1-5 中的整数,\n'
            + f'  "reason": "{reason_json_rule}",\n'
            + '  "analysis": "按 Step1-5 简述：prior 是多少；delta 证据；是否过 3/4 boundary；最终分如何由 prior+delta+boundary 得到",\n'
            + f'  "history_prior_score": {memory.avg_satisfaction_score:.2f},\n'
            + '  "delta_label": "below",\n'
            + '  "delta_score": -2 到 2 的整数,\n'
            + '  "passes_satisfaction_boundary": true 或 false,\n'
            + '  "boundary_score": 只能是 3 或 4\n'
            + "}\n"
        )
        return prompt

    if prompt_version == "v3":
        calibration_summary = getattr(memory, "calibration_summary", "")
        evidence_notes = list(getattr(memory, "evidence_notes", []))
        can_compare_3_vs_4 = bool(getattr(memory, "can_compare_3_vs_4", True))
        can_compare_4_vs_5 = bool(getattr(memory, "can_compare_4_vs_5", True))
        low_score_evidence_level = getattr(memory, "low_score_evidence_level", "moderate")
        evidence_block = (
            "\n".join(f"  - {note}" for note in evidence_notes)
            if evidence_notes else
            "  - 边界证据正常，可按规则使用"
        )
        rule_34 = (
            memory.three_vs_four_distinction
            if can_compare_3_vs_4 else
            f"【弱推断，不能当硬规则】{memory.three_vs_four_distinction}"
        )
        rule_45 = (
            memory.four_vs_five_distinction
            if can_compare_4_vs_5 else
            f"【弱推断，不能当硬规则】{memory.four_vs_five_distinction}"
        )
        low_score_note = (
            "当前 <=3 历史证据很少；若回复低于 4 分，默认先给 3，只有出现明显不可用/严重错误/严重答非所问时才降到 2 或 1。"
            if low_score_evidence_level in {"none", "sparse"} else
            "当前 <=3 历史证据足以支持 1/2/3 的相对严重度细分。"
        )
        anchor_instruction = (
            "【参考案例使用规则】\n"
            "1. 先找与当前回复整体质量最接近的案例，作为初始刻度，不要只盯着高分案例挑毛病。\n"
            "2. 若 memory 的某条边界规则被标记为【弱推断】，优先参考统计校准和真实案例，而不要机械服从该规则。\n"
            "3. 若参考案例与弱边界规则冲突，优先相信更直接的证据：实际案例和整体分布。\n\n"
            if anchor_turns else ""
        )
        prompt = (
            "你是一名个性化对话质量评估员。请给当前助手回复打 1-5 分。\n"
            "这是 memory v3 路线：你必须把【校准信息】与【边界规则】分开使用。\n\n"
            f"【校准信息（优先作为整体分数刻度）】\n"
            f"程序校准摘要：{calibration_summary or memory.scoring_style}\n"
            f"历史平均分：{memory.avg_satisfaction_score:.2f}\n"
            f"分布：5分×{memory.score_distribution.score_5} / "
            f"4分×{memory.score_distribution.score_4} / "
            f"3分×{memory.score_distribution.score_3} / "
            f"2分×{memory.score_distribution.score_2} / "
            f"1分×{memory.score_distribution.score_1}\n\n"
            f"【边界规则（按证据充分性使用）】\n"
            f"3→4 边界：{rule_34}\n"
            f"4→5 边界：{rule_45}\n"
            f"低分细分提示：{low_score_note}\n"
            f"证据提醒：\n{evidence_block}\n\n"
            f"【真正影响评分的个性化要求】\n{user_reqs}\n"
            + (f"任务特定观察：\n{task_obs_lines}\n" if task_obs_lines else "")
            + f"偏好回复形式（仅供次要参考）：{memory.preferred_response_format}\n\n"
            + f"{anchor_section}"
            + anchor_instruction
            + f"【用户画像】{_format_profile(profile)}\n\n"
            + f"【任务背景】{task_context}\n\n"
            + f"【最近对话历史】\n{history_text}\n\n"
            + f"【待评估的助手回复】\n{assistant_reply}\n\n"
            + f"【可选原因标签】{reason_text}\n{reason_rule_block}\n"
            + "【评分步骤】\n"
            + "Step A. 先用【校准信息】估计：这个用户整体是更容易给高分，还是更容易压分。不要忽略这个先验。\n"
            + "Step B. 再用【3→4 边界】判断是否过 4 分基线。\n"
            + "  - 若该边界被标记为弱推断，不要把它当硬规则；要更多参考统计刻度、真实案例和用户特定要求。\n"
            + "Step C. 若达到 4 分，再用【4→5 边界】判断是否升到 5。\n"
            + "  - 若 4→5 边界是弱推断，默认保守给 4；只有回复明显超过一般 4 分完成度时才给 5。\n"
            + "Step D. 若未达到 4 分，再细分 1/2/3。\n"
            + "  - 若低分证据 sparse/none，默认优先给 3；只有严重不可用、明显错误或严重答非所问时才给 2/1。\n"
            + "Step E. 选择最贴切的 reason。若最终分数 >=4，reason 必须是 `满意`。\n\n"
            + "注意：\n"
            + "- 校准信息决定“整体刻度”，边界规则决定“临界点”；二者都要用，但不要让弱证据边界压倒更强的校准/案例证据。\n"
            + "- 不要因为 memory 里出现了一条像规则的话，就忽略它可能只是弱推断。\n\n"
            + "请严格输出 JSON，不要输出其他内容：\n"
            + "{\n"
            + '  "classification": 1-5 中的整数,\n'
            + f'  "reason": "{reason_json_rule}",\n'
            + '  "analysis": "按 StepA-E 简述：校准先验是什么；3/4 或 4/5 边界是否可靠；最终分数如何决定" \n'
            + "}\n"
        )
        return prompt

    if prompt_version == "v3_1":
        calibration_summary = getattr(memory, "calibration_summary", "")
        evidence_notes = list(getattr(memory, "evidence_notes", []))
        can_compare_3_vs_4 = bool(getattr(memory, "can_compare_3_vs_4", True))
        can_compare_4_vs_5 = bool(getattr(memory, "can_compare_4_vs_5", True))
        low_score_evidence_level = getattr(memory, "low_score_evidence_level", "moderate")
        evidence_block = (
            "\n".join(f"  - {note}" for note in evidence_notes)
            if evidence_notes else
            "  - 边界证据正常，可按规则使用"
        )
        rule_34 = (
            memory.three_vs_four_distinction
            if can_compare_3_vs_4 else
            f"【弱推断，不能当硬规则】{memory.three_vs_four_distinction}"
        )
        rule_45 = (
            memory.four_vs_five_distinction
            if can_compare_4_vs_5 else
            f"【弱推断，不能当硬规则】{memory.four_vs_five_distinction}"
        )
        low_score_note = (
            "当前 <=3 历史证据 sparse/none：这只意味着 1/2/3 内部细分要保守；它不意味着可以放松 3/4 满意边界。若未过满意线，默认先给 3，只有严重不可用或明显错误时才降到 2/1。"
            if low_score_evidence_level in {"none", "sparse"} else
            "当前 <=3 历史证据足以支持 1/2/3 的相对严重度细分。"
        )
        anchor_instruction = (
            "【参考案例使用规则】\n"
            "1. 先找与当前回复整体质量最接近的案例，作为辅助刻度。\n"
            "2. 但若当前回复没过最低满意线，不要因为参考案例整体分布偏高就勉强给 SAT。\n"
            "3. 当 3/4 边界是弱推断时，优先看：核心问题是否回答、关键约束是否满足、用户特定要求是否被漏掉。\n"
            "4. 只有在已经明确过了 SAT gate 后，才让 calibration 和 4/5 边界去决定是否升到 5。\n\n"
            if anchor_turns else ""
        )
        prompt = (
            "你是一名个性化对话质量评估员。请给当前助手回复打 1-5 分。\n"
            "这是 memory v3.1 路线：保留 v3 的 calibration 优势，但重新加硬【3/4 最低满意线】。\n\n"
            f"【校准信息（用于整体刻度，不直接决定是否满意）】\n"
            f"程序校准摘要：{calibration_summary or memory.scoring_style}\n"
            f"历史平均分：{memory.avg_satisfaction_score:.2f}\n"
            f"分布：5分×{memory.score_distribution.score_5} / "
            f"4分×{memory.score_distribution.score_4} / "
            f"3分×{memory.score_distribution.score_3} / "
            f"2分×{memory.score_distribution.score_2} / "
            f"1分×{memory.score_distribution.score_1}\n\n"
            f"【边界规则（按证据充分性使用）】\n"
            f"3→4 边界：{rule_34}\n"
            f"4→5 边界：{rule_45}\n"
            f"低分细分提示：{low_score_note}\n"
            f"证据提醒：\n{evidence_block}\n\n"
            f"【真正影响评分的个性化要求】\n{user_reqs}\n"
            + (f"任务特定观察：\n{task_obs_lines}\n" if task_obs_lines else "")
            + f"偏好回复形式（仅供次要参考）：{memory.preferred_response_format}\n\n"
            + f"{anchor_section}"
            + anchor_instruction
            + f"【用户画像】{_format_profile(profile)}\n\n"
            + f"【任务背景】{task_context}\n\n"
            + f"【最近对话历史】\n{history_text}\n\n"
            + f"【待评估的助手回复】\n{assistant_reply}\n\n"
            + f"【可选原因标签】{reason_text}\n{reason_rule_block}\n"
            + "【评分步骤】\n"
            + "Step A. 先读校准信息，只把它当作整体刻度先验：这个用户通常偏高分还是偏低分。它不能直接替代满意/不满意判断。\n"
            + "Step B. 先做【强 3/4 gate】：判断当前回复是否已经过了最低满意线。\n"
            + "  - 必须先回答三个问题：\n"
            + "    1. 核心问题是否被直接回答？\n"
            + "    2. 关键约束 / 关键任务目标 / 用户特别在意的要求是否被满足？\n"
            + "    3. 剩余缺口是否只是普通不够细致，而不是会让用户仍然不满意的关键缺口？\n"
            + "  - 只要以上任一关键项明显失败，就不能给 >=4。\n"
            + "  - 若 3→4 边界是弱推断，不是放松 gate，而是改为更多依赖上述三个问题与真实案例。\n"
            + "Step C. 只有在 Step B 已明确通过 SAT gate 后，才允许进入 4/5 细化。\n"
            + "  - 若 4→5 边界是弱推断，默认保守给 4；只有明显超过一般 4 分完成度时才给 5。\n"
            + "Step D. 若 Step B 未通过 SAT gate，再细分 1/2/3。\n"
            + "  - 若低分证据 sparse/none，默认先给 3；只有严重不可用、明显错误、严重答非所问时才给 2/1。\n"
            + "  - 证据不足只影响 1/2/3 的内部细分，不影响你先把样本判为 <=3。\n"
            + "Step E. 选择最贴切的 reason。若最终分数 >=4，reason 必须是 `满意`。\n\n"
            + "注意：\n"
            + "- v3.1 的核心原则是：先过 SAT gate，再做 calibration 和 4/5 refinement；不能因为用户通常打分偏高，就让未过线的回复变成 SAT。\n"
            + "- `不够细致` 只有在仍然满足核心需求时才属于普通缺口；若它已经导致关键目标没完成，就仍然是 DSAT。\n"
            + "- 不要把“证据不足”误解成“默认偏 SAT”。\n\n"
            + "请严格输出 JSON，不要输出其他内容：\n"
            + "{\n"
            + '  "classification": 1-5 中的整数,\n'
            + f'  "reason": "{reason_json_rule}",\n'
            + '  "analysis": "按 StepA-E 简述：校准先验是什么；最低满意线是否通过；若通过为何是4或5，若未通过为何是3/2/1" \n'
            + "}\n"
        )
        return prompt

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


def build_turn_eval_v3_two_stage_gate_prompt(
    memory: UserMemory,
    profile: dict,
    task_context: str,
    history_window: list[str],
    assistant_reply: str,
    anchor_turns: list | None = None,
) -> str:
    """memory v3 两阶段版本的第一层：只判是否过 SAT gate（3/4）。"""
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
    calibration_summary = getattr(memory, "calibration_summary", memory.scoring_style)
    evidence_notes = list(getattr(memory, "evidence_notes", []))
    can_compare_3_vs_4 = bool(getattr(memory, "can_compare_3_vs_4", True))
    rule_34 = (
        memory.three_vs_four_distinction
        if can_compare_3_vs_4 else
        f"【弱推断，不能当硬规则】{memory.three_vs_four_distinction}"
    )
    evidence_block = (
        "\n".join(f"  - {note}" for note in evidence_notes)
        if evidence_notes else
        "  - 边界证据正常，可按规则使用"
    )
    anchor_block = _format_anchor_turns(anchor_turns or [])
    anchor_section = (anchor_block + "\n") if anchor_block else ""
    anchor_instruction = (
        "【参考案例使用规则】\n"
        "1. 本层只判断是否通过最低满意线。不要先想 5 分，只判断当前回复是否至少算满意。\n"
        "2. 若参考案例显示类似回复在该用户历史中经常落到 <=3，除非当前回复明显更好，否则不要轻易给 4。\n"
        "3. 若 3/4 边界是弱推断，优先看核心问题、关键约束、可用性和用户特定要求是否满足。\n\n"
        if anchor_turns else ""
    )
    return (
        "你是一名个性化满意度评估员。\n"
        "这是 memory v3 两阶段 pipeline 的第一层。你的任务只有一个：判断当前回复是否通过该用户的最低满意线。\n"
        "输出只能是：\n"
        "- `4` = 通过 SAT gate（至少满意）\n"
        "- `3` = 未通过 SAT gate（仍然不满意）\n\n"
        "这一层不能直接考虑 5 分，也不能因为用户整体偏高分就放松 gate。\n\n"
        f"【校准信息（只作背景）】\n"
        f"{calibration_summary}\n"
        f"历史平均分：{memory.avg_satisfaction_score:.2f}\n\n"
        f"【SAT gate 规则】\n"
        f"3→4 边界：{rule_34}\n"
        f"证据提醒：\n{evidence_block}\n\n"
        f"【真正影响评分的个性化要求】\n{user_reqs}\n"
        + (f"任务特定观察：\n{task_obs_lines}\n" if task_obs_lines else "")
        + "\n"
        + f"{anchor_section}"
        + anchor_instruction
        + f"【用户画像】{_format_profile(profile)}\n\n"
        + f"【任务背景】{task_context}\n\n"
        + f"【最近对话历史】\n{history_text}\n\n"
        + f"【待评估的助手回复】\n{assistant_reply}\n\n"
        + f"【可选原因标签】{reason_text}\n{reason_rule_block}\n"
        + "【第一层只做 SAT gate】\n"
        + "Step 1. 判断核心问题是否被直接回答。\n"
        + "Step 2. 判断关键约束、关键任务目标、该用户特别在意的要求是否被满足。\n"
        + "Step 3. 判断剩余缺口是否只是普通不够细致，而不是会让用户仍然不满意的关键缺口。\n"
        + "Step 4. 只要核心问题未回答、关键要求被漏掉、或可用性明显不足，就不能给 4。\n"
        + "Step 5. 只有确认已经过了最低满意线，才能给 4；否则给 3。\n\n"
        + "注意：\n"
        + "- 若 3/4 边界证据不足，这不等于可以默认偏 SAT；它只意味着你应更多依赖核心问题、关键要求和真实案例。\n"
        + "- 本层不区分 4 和 5。\n\n"
        + "请严格输出 JSON，不要输出其他内容：\n"
        + "{\n"
        + '  "classification": 只能是 3 或 4,\n'
        + f'  "reason": "{reason_json_rule}",\n'
        + '  "analysis": "用 1-2 句写明：核心问题是否回答、关键要求是否满足、为何通过或未通过 SAT gate" \n'
        + "}\n"
    )


def build_turn_eval_v3_two_stage_v2_gate_prompt(
    memory: UserMemory,
    profile: dict,
    task_context: str,
    history_window: list[str],
    assistant_reply: str,
    anchor_turns: list | None = None,
) -> str:
    """memory v3 两阶段 v2 的第一层：借鉴 selective-refute v2 的 SAT gate。"""
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
    calibration_summary = getattr(memory, "calibration_summary", memory.scoring_style)
    evidence_notes = list(getattr(memory, "evidence_notes", []))
    can_compare_3_vs_4 = bool(getattr(memory, "can_compare_3_vs_4", True))
    rule_34 = (
        memory.three_vs_four_distinction
        if can_compare_3_vs_4 else
        f"【弱推断，仅作参考】{memory.three_vs_four_distinction}"
    )
    evidence_block = (
        "\n".join(f"  - {note}" for note in evidence_notes)
        if evidence_notes else
        "  - 3/4 边界证据正常，可按规则使用"
    )
    anchor_block = _format_anchor_turns(anchor_turns or [])
    anchor_section = (anchor_block + "\n") if anchor_block else ""
    anchor_instruction = (
        "【参考案例使用规则】\n"
        "1. 只把案例理解为边界参考：真实分数 <=3 是【未达满意线案例】，>=4 是【达到满意线案例】。\n"
        "2. 只有当当前回复与两类案例都存在明显相似点、边界仍拿不准时，才考虑触发复核。\n"
        "3. 若当前回复整体明显站在某一边，就不要触发复核。\n\n"
        if anchor_turns else ""
    )
    return (
        "你是一名个性化满意度边界评估员。\n"
        "这是 memory v3 两阶段 v2 的第一层 SAT gate。你的任务是先判断当前回复是否达到该用户的【满意最低线】。\n"
        "输出只能是：\n"
        "- `4` = 满意（达到最低满意线）\n"
        "- `3` = 不满意（未达到最低满意线）\n\n"
        "除分数外，你还需要判断：这个样本是否【高度接近 3/4 边界】，需要进入 gate 复核。\n"
        "`needs_refute_review=true` 必须是少数情况；只有在你确实拿不准时才允许触发。\n\n"
        f"【校准信息（只作背景，不得放松 SAT gate）】\n"
        f"{calibration_summary}\n"
        f"历史平均分：{memory.avg_satisfaction_score:.2f}\n\n"
        f"【SAT gate 规则】\n"
        f"满意最低线（3→4 边界）：{rule_34}\n"
        f"证据提醒：\n{evidence_block}\n"
        f"更高要求（4→5，仅供背景参考）：{memory.four_vs_five_distinction}\n"
        f"用户特定要求：\n{user_reqs}\n"
        + (f"任务特定观察：\n{task_obs_lines}\n" if task_obs_lines else "")
        + "\n"
        + f"{anchor_section}"
        + anchor_instruction
        + f"【用户画像】{_format_profile(profile)}\n\n"
        + f"【任务背景】{task_context}\n\n"
        + f"【最近对话历史】\n{history_text}\n\n"
        + f"【待评估的助手回复】\n{assistant_reply}\n\n"
        + f"【可选原因标签】{reason_text}\n{reason_rule_block}\n"
        + "【第一遍只做严格筛选后的 SAT gate】\n"
        + "Step 1. 判断回复是否回答了核心问题，并基本满足关键约束。\n"
        + "Step 2. 判断它是否达到该用户的满意最低线：达到给 `4`，未达到给 `3`。\n"
        + "Step 3. 再判断是否真的需要复核。只有下面两类高不确定情形才允许 `needs_refute_review=true`：\n"
        + "  - 当前判成 `3`，但你怀疑问题主要只是边缘性的细节不足，未必真的低于满意线\n"
        + "  - 当前判成 `4`，但你怀疑它可能漏掉了一个关键要求，是否仍算满意拿不准\n"
        + "Step 4. 若主要证据已经明显站在一边，必须输出 `needs_refute_review=false`。\n\n"
        + "注意：\n"
        + "- 校准信息只能帮助你理解用户整体严格度，不能拿来抵消“核心问题没回答/关键要求没满足/可用性不足”。\n"
        + "- 若 3/4 边界证据不足，不要默认偏 SAT；此时更应依赖核心问题、关键要求、可用性和参考案例。\n"
        + "- 不要因为“还可以更好”就触发复核。\n"
        + "- 不要因为理由是 `不够细致` 就自动触发复核。\n"
        + "- 只有当一个具体可疑点是否属于关键失败拿不准时，才触发复核。\n"
        + "- `classification` 只能输出 `3` 或 `4`。\n"
        + "- `analysis` 只需 1-2 句，明确写出：当前为何判为 3 或 4、唯一的可疑点是什么、是否真的需要复核。\n\n"
        + "请严格输出 JSON，不要输出其他内容：\n"
        + "{\n"
        + '  "classification": 只能是 3 或 4,\n'
        + f'  "reason": "{reason_json_rule}",\n'
        + '  "analysis": "用 1-2 句写明当前为何判为 3 或 4、唯一的可疑点是什么，以及是否真的需要复核",\n'
        + '  "needs_refute_review": true 或 false\n'
        + "}\n"
    )


def build_turn_eval_v3_two_stage_v2_gate_followup_prompt(
    memory: UserMemory,
    profile: dict,
    task_context: str,
    history_window: list[str],
    assistant_reply: str,
    initial_classification: int,
    initial_reason: str,
    initial_analysis: str,
) -> str:
    """memory v3 两阶段 v2 的 gate 复核 prompt。"""
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
    calibration_summary = getattr(memory, "calibration_summary", memory.scoring_style)
    evidence_notes = list(getattr(memory, "evidence_notes", []))
    evidence_block = (
        "\n".join(f"  - {note}" for note in evidence_notes)
        if evidence_notes else
        "  - 3/4 边界证据正常，可按规则使用"
    )
    return (
        "你是一名个性化满意度边界复核员。\n"
        "这是 memory v3 两阶段 v2 的 gate 复核，只在第一遍认为样本高度接近 3/4 边界时触发。\n"
        "你的任务不是重新完整评分，而是核实：第一遍指出的唯一可疑点，是否真的足以推翻第一遍初判。\n\n"
        "输出只能是：\n"
        "- `4` = 满意（达到最低满意线）\n"
        "- `3` = 不满意（未达到最低满意线）\n\n"
        f"【校准信息（只作背景）】\n{calibration_summary}\n"
        f"【SAT gate 规则】\n"
        f"满意最低线（3→4 边界）：{memory.three_vs_four_distinction}\n"
        f"证据提醒：\n{evidence_block}\n"
        f"更高要求（4→5，仅供背景参考）：{memory.four_vs_five_distinction}\n"
        f"用户特定要求：\n{user_reqs}\n"
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
        + "Step 5. 校准信息不能单独构成改判理由；改判必须来自当前回复本身的明确证据。\n\n"
        + "请严格输出 JSON，不要输出其他内容：\n"
        + "{\n"
        + '  "classification": 只能是 3 或 4,\n'
        + f'  "reason": "{reason_json_rule}",\n'
        + '  "analysis": "用 1-2 句写明：是否发现足以推翻第一遍初判的明确反证；最终为何维持或改判" \n'
        + "}\n"
    )


def build_turn_eval_v3_two_stage_sat_refinement_prompt(
    memory: UserMemory,
    profile: dict,
    task_context: str,
    history_window: list[str],
    assistant_reply: str,
    gate_reason: str,
    gate_analysis: str,
    anchor_turns: list | None = None,
) -> str:
    """memory v3 两阶段版本第二层 SAT 分支：只细化 4/5。"""
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
    calibration_summary = getattr(memory, "calibration_summary", memory.scoring_style)
    can_compare_4_vs_5 = bool(getattr(memory, "can_compare_4_vs_5", True))
    rule_45 = (
        memory.four_vs_five_distinction
        if can_compare_4_vs_5 else
        f"【弱推断，默认保守给4】{memory.four_vs_five_distinction}"
    )
    anchor_block = _format_anchor_turns(anchor_turns or [])
    anchor_section = (anchor_block + "\n") if anchor_block else ""
    return (
        "你是一名个性化满意度评估员。\n"
        "这是 memory v3 两阶段 pipeline 的第二层 SAT 分支。第一层已确认当前回复至少满意。\n"
        "你的任务只是在 `4` 和 `5` 之间细化。\n\n"
        f"【校准信息】\n{calibration_summary}\n\n"
        f"【4/5 细化规则】\n{rule_45}\n"
        f"【真正影响评分的个性化要求】\n{user_reqs}\n"
        + (f"任务特定观察：\n{task_obs_lines}\n" if task_obs_lines else "")
        + "\n"
        + f"{anchor_section}"
        + f"【用户画像】{_format_profile(profile)}\n\n"
        + f"【任务背景】{task_context}\n\n"
        + f"【最近对话历史】\n{history_text}\n\n"
        + f"【待评估的助手回复】\n{assistant_reply}\n\n"
        + f"【第一层 gate 输出】reason={gate_reason}\n"
        + f"【第一层 gate 分析】{gate_analysis}\n\n"
        + f"【可选原因标签】{reason_text}\n{reason_rule_block}\n"
        + "【第二层 SAT 细化】\n"
        + "Step 1. 先把 4 当默认值：既然已经过了 SAT gate，除非有明确证据达到高满意门槛，否则保持 4。\n"
        + "Step 2. 只有当回复明显完整、个性化、可执行，并接近该用户的高满意案例时，才升到 5。\n"
        + "Step 3. 若 4/5 边界证据不足，默认保守给 4，而不是猜 5。\n\n"
        + "请严格输出 JSON，不要输出其他内容：\n"
        + "{\n"
        + '  "classification": 只能是 4 或 5,\n'
        + f'  "reason": "{reason_json_rule}",\n'
        + '  "analysis": "用 1-2 句写明：为何保持4，或为何已达到5分门槛" \n'
        + "}\n"
    )


def build_turn_eval_v3_two_stage_dsat_refinement_prompt(
    memory: UserMemory,
    profile: dict,
    task_context: str,
    history_window: list[str],
    assistant_reply: str,
    gate_reason: str,
    gate_analysis: str,
    anchor_turns: list | None = None,
) -> str:
    """memory v3 两阶段版本第二层 DSAT 分支：只细化 1/2/3。"""
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
    low_score_evidence_level = getattr(memory, "low_score_evidence_level", "moderate")
    evidence_note = (
        "低分证据 sparse/none：默认优先给 3；只有明显不可用、明显错误、严重答非所问时才给 2 或 1。"
        if low_score_evidence_level in {"none", "sparse"} else
        "低分证据充分：可以正常区分 1/2/3 的严重度。"
    )
    anchor_block = _format_anchor_turns(anchor_turns or [])
    anchor_section = (anchor_block + "\n") if anchor_block else ""
    return (
        "你是一名个性化满意度评估员。\n"
        "这是 memory v3 两阶段 pipeline 的第二层 DSAT 分支。第一层已确认当前回复没有通过最低满意线。\n"
        "你的任务只是在 `1/2/3` 之间细化严重度。\n\n"
        f"【低分细分提示】{evidence_note}\n"
        f"【真正影响评分的个性化要求】\n{user_reqs}\n"
        + (f"任务特定观察：\n{task_obs_lines}\n" if task_obs_lines else "")
        + "\n"
        + f"{anchor_section}"
        + f"【用户画像】{_format_profile(profile)}\n\n"
        + f"【任务背景】{task_context}\n\n"
        + f"【最近对话历史】\n{history_text}\n\n"
        + f"【待评估的助手回复】\n{assistant_reply}\n\n"
        + f"【第一层 gate 输出】reason={gate_reason}\n"
        + f"【第一层 gate 分析】{gate_analysis}\n\n"
        + f"【可选原因标签】{reason_text}\n{reason_rule_block}\n"
        + "【第二层 DSAT 细化】\n"
        + "Step 1. 既然第一层已判定未过 SAT gate，本层不能回到 4/5。\n"
        + "Step 2. 默认先考虑 3：即不满意，但仍有一定帮助。\n"
        + "Step 3. 只有当回复明显不可用、明显错误、严重答非所问，或几乎没有可执行价值时，才降到 2 或 1。\n"
        + "Step 4. 若低分证据 sparse/none，更要保守区分 1/2/3，不要轻易给极低分。\n\n"
        + "请严格输出 JSON，不要输出其他内容：\n"
        + "{\n"
        + '  "classification": 只能是 1、2 或 3,\n'
        + f'  "reason": "{reason_json_rule}",\n'
        + '  "analysis": "用 1-2 句写明：为什么是3，或为什么严重到2/1" \n'
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
