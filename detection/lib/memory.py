"""
用户记忆模块（User Memory）

UserMemory 是 training-free agent 对目标用户的个性化偏好总结，
由 LLM 从历史 session（含满意度标签）中提炼，并可在预测过程中迭代更新。

主要组件：
  ScoreDistribution    — 满意度分布（固定 5 字段，兼容 OpenAI structured output）
  TaskObservation      — 单个任务类型的观察（替代 dict[str, str]）
  UserMemoryContent    — LLM 生成部分，严格兼容 OpenAI structured output
  UserMemory           — 完整记忆 = UserMemoryContent + 程序侧元信息
  build_memory_prompt  — 从历史 session 构建记忆的 prompt
  build_memory_update_prompt — 在已有记忆基础上整合新 session 后更新的 prompt
"""

from __future__ import annotations

import json

from pydantic import BaseModel, Field

from .personalized_data import SessionData
from .satisfaction_constants import get_reason_to_id

# ──────────────────────────────────────────────────────────────────────────────
# 辅助子模型（用于替代 dict，保证 OpenAI structured output 兼容）
# ──────────────────────────────────────────────────────────────────────────────

class ScoreDistribution(BaseModel):
    """满意度 1-5 分的出现次数（固定字段，兼容 OpenAI structured output）。"""
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
    """针对单个任务类型的关键观察（替代 dict[str, str]）。"""
    task_name: str = Field(description="任务类型名称，如旅行规划")
    observation: str = Field(description="该任务下用户特有的偏好或敏感点")


# ──────────────────────────────────────────────────────────────────────────────
# LLM 生成模型（严格兼容 OpenAI structured output）
#
# 设计约束（OpenAI 要求）：
#   1. 无 dict 类型字段（使用 list[SubModel] 替代）
#   2. 所有字段均为必填（无 default），自动进入 required 列表
#   3. 嵌套模型同理
# ──────────────────────────────────────────────────────────────────────────────

class UserMemoryContent(BaseModel):
    """
    LLM 生成的用户记忆内容。
    此类作为 response_format 传给 OpenAI structured output API。
    """

    # ── 基本统计 ──────────────────────────────────────────────────────────────
    avg_satisfaction_score: float = Field(
        ge=1.0, le=5.0,
        description="历史 session 中所有 assistant 轮的平均满意度分数",
    )
    score_distribution: ScoreDistribution = Field(
        description="满意度 1-5 分各自的出现次数",
    )

    # ── 满意 / 不满意触发因素 ──────────────────────────────────────────────────
    high_satisfaction_patterns: list[str] = Field(
        description=(
            "让该用户打出 4-5 分的回复特征，1-6 条。每条应具体，"
            "如【提供分阶段的详细执行计划】而非【回答详细】"
        ),
    )
    dissatisfaction_patterns: list[str] = Field(
        description=(
            "让该用户打出 1-3 分的回复特征，0-6 条。每条应具体指出缺陷类型，"
            "如【推荐内容不考虑用户当前零基础的实际情况】"
        ),
    )

    # ── 沟通偏好 ──────────────────────────────────────────────────────────────
    preferred_response_format: str = Field(
        description="用户偏好的回复组织形式，如【分步骤的结构化列表，每步附具体示例】",
    )
    preferred_detail_level: str = Field(
        description="用户对信息详细程度的偏好，如【高度详细，需要可执行的具体步骤】",
    )

    # ── 任务特定观察（list[TaskObservation] 替代 dict[str, str]）────────────
    task_specific_observations: list[TaskObservation] = Field(
        description=(
            "针对各历史任务类型的关键观察，每项包含 task_name 和 observation。"
            "每个历史任务类型应有一条记录（0-4 条）"
        ),
    )

    # ── 综合特征 ──────────────────────────────────────────────────────────────
    notable_user_characteristics: list[str] = Field(
        description=(
            "该用户区别于一般用户的显著特征，0-5 条。"
            "如【对资源推荐极其敏感，要求具体的网址或书名】"
        ),
    )


# ──────────────────────────────────────────────────────────────────────────────
# 完整 UserMemory（UserMemoryContent + 程序侧元信息）
# ──────────────────────────────────────────────────────────────────────────────

class UserMemory(UserMemoryContent):
    """
    完整用户记忆。
    LLM 只生成 UserMemoryContent 部分；元信息字段由程序在调用后填写。
    """
    source_tasks: list[str] = Field(
        default_factory=list,
        description="构建本记忆所使用的历史任务类型列表",
    )
    n_history_sessions: int = Field(
        default=0,
        description="构建本记忆时使用的历史 session 数量",
    )
    n_history_turns: int = Field(
        default=0,
        description="构建本记忆时使用的历史 assistant 轮总数",
    )

    @classmethod
    def from_content(
        cls,
        content: UserMemoryContent,
        source_tasks: list[str] | None = None,
        n_history_sessions: int = 0,
        n_history_turns: int = 0,
    ) -> "UserMemory":
        """将 LLM 生成的 UserMemoryContent 转换为带元信息的 UserMemory。"""
        data = content.model_dump()
        data["source_tasks"] = list(source_tasks or [])
        data["n_history_sessions"] = n_history_sessions
        data["n_history_turns"] = n_history_turns
        return cls(**data)


# ──────────────────────────────────────────────────────────────────────────────
# Prompt 构建
# ──────────────────────────────────────────────────────────────────────────────

_MAX_CONTENT_CHARS = 300   # 单条 assistant 回复截断长度
_MAX_SESSIONS_IN_PROMPT = 12  # 最多放入 prompt 的 session 数（避免超长）


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


def _format_session_for_memory(session: SessionData, idx: int) -> str:
    """将一个历史 session 格式化为 memory building prompt 中的片段。"""
    lines = [
        f"【Session {idx + 1}】任务：{session.task}",
        f"任务背景：{session.task_context[:200]}",
        "对话摘要（含满意度标注）：",
    ]
    assistant_turn_idx = 0
    conv_window: list[str] = []
    for utt in session.history:
        role_label = "用户" if utt["role"] == "user" else "助手"
        content = utt["content"][:_MAX_CONTENT_CHARS]
        if len(utt["content"]) > _MAX_CONTENT_CHARS:
            content += "…（截断）"
        conv_window.append(f"  {role_label}: {content}")

        if utt["role"] == "assistant":
            score = session.satisfaction_scores[assistant_turn_idx]
            reason = session.dissatisfaction_reasons[assistant_turn_idx]
            score_label = f"★{score}"
            if score <= 3:
                score_label += f"（不满意原因：{reason}）"
            conv_window.append(f"  [满意度: {score_label}]")
            assistant_turn_idx += 1

    lines.extend(conv_window)
    return "\n".join(lines)


def build_memory_prompt(
    user_id: str,
    profile: dict,
    history_sessions: list[SessionData],
) -> str:
    """
    构造 memory building prompt。

    输入：用户 profile + 历史 sessions（含满意度标注）
    输出：JSON 格式的 UserMemory
    """
    reason_labels = list(get_reason_to_id().keys())

    # 如果历史 session 太多，截取最近的若干个（按 session 顺序，保留多样性）
    sessions_to_use = history_sessions
    if len(history_sessions) > _MAX_SESSIONS_IN_PROMPT:
        # 尽量保持各任务类型均匀采样
        from collections import defaultdict
        by_task: dict[str, list[SessionData]] = defaultdict(list)
        for s in history_sessions:
            by_task[s.task].append(s)
        sessions_to_use = []
        per_task = max(1, _MAX_SESSIONS_IN_PROMPT // len(by_task))
        for task_sessions in by_task.values():
            sessions_to_use.extend(task_sessions[:per_task])
        sessions_to_use = sessions_to_use[:_MAX_SESSIONS_IN_PROMPT]

    session_texts = [
        _format_session_for_memory(s, i) for i, s in enumerate(sessions_to_use)
    ]
    session_block = "\n\n".join(session_texts)

    prompt = (
        "你是一名用户行为分析专家。你的任务是基于一名用户与 AI 助手的多个历史对话 session，"
        "总结该用户的满意度偏好特征，形成一份可用于后续预测的用户记忆（User Memory）。\n\n"
        f"【用户 ID】{user_id}\n"
        f"【用户画像】{_format_profile(profile)}\n\n"
        "【历史对话 Sessions（含满意度标注）】\n"
        f"{session_block}\n\n"
        "请仔细分析以上 session，重点关注：\n"
        "1. 哪类回复让该用户给出 4-5 分（满意）？\n"
        "2. 哪类回复让该用户给出 1-3 分（不满意）？\n"
        "3. 该用户对回复的组织形式、信息详细程度有什么偏好？\n"
        "4. 用户在不同任务类型中有哪些特定的敏感点或偏好？\n"
        "5. 该用户有哪些与众不同的显著特征？\n\n"
        f"可参考的不满意原因类别：{', '.join(reason_labels)}\n\n"
        "请严格按照 JSON Schema 输出结构化用户记忆，不要输出其他内容。"
    )
    return prompt


def build_memory_update_prompt(
    existing_memory: UserMemory,
    new_session: SessionData,
    turn_predictions: list[dict],
    use_oracle_labels: bool = False,
) -> str:
    """
    构造 memory update prompt。

    在已有记忆的基础上，整合一个新 session 的信息后更新记忆。

    参数
    ----
    existing_memory : UserMemory
        当前记忆快照。
    new_session : SessionData
        新观察到的 session。
    turn_predictions : list[dict]
        该 session 中各轮的预测结果，每条含：
          pred_score, pred_reason, analysis
        若 use_oracle_labels=True，则额外含 gold_score, gold_reason。
    use_oracle_labels : bool
        是否使用真实标签更新记忆（oracle 模式，用于分析记忆质量上界）。
    """
    existing_json = existing_memory.model_dump_json(indent=2)

    # 构建新 session 的描述
    session_lines = [
        f"任务：{new_session.task}",
        f"任务背景：{new_session.task_context[:200]}",
        "逐轮预测与实际（若有）：",
    ]
    assistant_turn_idx = 0
    history_window: list[str] = []
    for utt in new_session.history:
        role_label = "用户" if utt["role"] == "user" else "助手"
        content = utt["content"][:_MAX_CONTENT_CHARS]
        if len(utt["content"]) > _MAX_CONTENT_CHARS:
            content += "…（截断）"
        history_window.append(f"  {role_label}: {content}")

        if utt["role"] == "assistant" and assistant_turn_idx < len(turn_predictions):
            pred = turn_predictions[assistant_turn_idx]
            pred_score = pred.get("pred_score", "?")
            pred_reason = pred.get("pred_reason", "?")
            entry = f"  [预测满意度: ★{pred_score}，原因: {pred_reason}]"
            if use_oracle_labels:
                gold_score = pred.get("gold_score", "?")
                gold_reason = pred.get("gold_reason", "?")
                entry += f"  [真实满意度: ★{gold_score}，原因: {gold_reason}]"
            history_window.append(entry)
            assistant_turn_idx += 1

    session_lines.extend(history_window)
    session_text = "\n".join(session_lines)

    label_note = (
        "（注意：本次更新同时提供了真实满意度标签，请优先基于真实标签调整记忆）"
        if use_oracle_labels
        else "（注意：本次更新仅使用模型预测分数，无真实标签）"
    )

    prompt = (
        "你正在维护一份用户记忆（User Memory）。现在该用户完成了一个新的对话 session，"
        "请你基于新 session 的观察，更新并完善已有记忆。\n\n"
        f"【现有用户记忆】\n{existing_json}\n\n"
        f"【新 Session 信息】\n{session_text}\n\n"
        f"{label_note}\n\n"
        "更新要求：\n"
        "- 若新 session 印证了已有模式，可保留或加强描述\n"
        "- 若新 session 揭示了新的偏好或与已有模式矛盾，请相应修改\n"
        "- 统计字段（avg_satisfaction_score, score_distribution 等）需根据新数据更新\n"
        "- task_specific_observations 可新增条目，但不要删除已有条目\n\n"
        "请严格按照原 JSON Schema 输出更新后的完整用户记忆（不含 source_tasks / n_history_sessions / n_history_turns 元信息字段），不要输出其他内容。"
    )
    return prompt


def build_turn_eval_prompt(
    memory: UserMemory,
    profile: dict,
    task_context: str,
    history_window: list[str],
    assistant_reply: str,
) -> str:
    """
    构造单轮满意度预测 prompt（利用用户记忆）。

    参数
    ----
    memory : UserMemory
        当前用户记忆（可为 None 时退化为无记忆 baseline）。
    history_window : list[str]
        当前对话的最近若干轮，格式 ["用户: ...", "助手: ..."]。
    assistant_reply : str
        待评估的 assistant 回复。
    """
    reason_labels = list(get_reason_to_id().keys())
    reason_labels_text = "、".join(reason_labels)

    memory_section = (
        f"【用户记忆摘要】\n"
        f"  历史平均满意度：{memory.avg_satisfaction_score:.2f}\n"
        f"  满意触发因素：{'; '.join(memory.high_satisfaction_patterns)}\n"
        f"  不满意触发因素：{'; '.join(memory.dissatisfaction_patterns)}\n"
        f"  偏好回复形式：{memory.preferred_response_format}\n"
        f"  偏好详细程度：{memory.preferred_detail_level}\n"
        f"  用户显著特征：{'; '.join(memory.notable_user_characteristics)}\n"
    )
    if memory.task_specific_observations:
        obs_lines = "\n".join(
            f"    {obs.task_name}: {obs.observation}"
            for obs in memory.task_specific_observations
        )
        memory_section += f"  任务特定观察：\n{obs_lines}\n"

    history_text = "\n".join(history_window) if history_window else "（无历史对话）"

    prompt = (
        "你是一名个性化对话质量分析员。"
        "你了解当前用户的历史满意度偏好，请基于用户记忆对助手回复进行评估。\n\n"
        f"{memory_section}\n"
        f"【用户画像】{_format_profile(profile)}\n\n"
        f"【任务背景】{task_context}\n\n"
        f"【最近对话历史】\n{history_text}\n\n"
        f"【当前助手回复】{assistant_reply}\n\n"
        f"【可选原因标签】{reason_labels_text}\n\n"
        "请基于用户记忆中的满意/不满意触发因素，先进行推理，再预测满意度。\n"
        "请严格输出 JSON，不要输出其他内容：\n"
        "{\n"
        '  "classification": 1-5 中的整数,\n'
        '  "reason": "从可选原因标签中选择一个",\n'
        '  "analysis": "你的详细推理过程，需明确引用用户记忆中的哪条模式支撑了判断"\n'
        "}\n"
    )
    return prompt


def build_turn_eval_prompt_no_memory(
    profile: dict,
    task_context: str,
    history_window: list[str],
    assistant_reply: str,
) -> str:
    """
    无记忆版本的 turn 评估 prompt（用于 baseline 对比）。
    与 collect_api.py::build_prompt 逻辑对齐。
    """
    reason_labels = list(get_reason_to_id().keys())
    reason_labels_text = "、".join(reason_labels)
    history_text = "\n".join(history_window) if history_window else "（无历史对话）"

    prompt = (
        "你是一名会进行细粒度对话质量分析的评估员。\n"
        "请基于给定信息先进行推理，再同时预测：\n"
        "1) 当前用户对助手回复的满意度分数（1-5）\n"
        '2) 潜在原因（必须从给定标签中选择，包含【满意】）\n\n'
        f"【用户画像】{_format_profile(profile)}\n\n"
        f"【任务背景】{task_context}\n\n"
        f"【最近对话历史】\n{history_text}\n\n"
        f"【当前助手回复】{assistant_reply}\n\n"
        f"【可选原因标签】{reason_labels_text}\n\n"
        "请严格输出 JSON，不要输出其他内容：\n"
        "{\n"
        '  "classification": 1-5 中的整数,\n'
        '  "reason": "从可选原因标签中选择一个",\n'
        '  "analysis": "你的详细推理过程"\n'
        "}\n"
    )
    return prompt
