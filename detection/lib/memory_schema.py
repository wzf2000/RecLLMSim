"""Pydantic schemas and schema-local helpers for user memory."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

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
