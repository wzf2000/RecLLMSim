"""Prompt builders for episodic-RAG satisfaction prediction."""

from __future__ import annotations

from .episodic_rag import EpisodicMemoryRecord, format_episodic_memories
from .memory_formatting import (
    _format_profile,
    _format_reason_json_rule,
    _format_reason_rule_block,
    _truncate,
)


def format_history_window(history_window: list[dict], max_chars: int = 900) -> str:
    if not history_window:
        return "（当前轮前没有更多对话上下文。）"
    lines: list[str] = []
    for utt in history_window:
        role = utt.get("role", "unknown")
        content = _truncate(utt.get("content", ""), 260)
        lines.append(f"{role}: {content}")
    text = "\n".join(lines)
    return text if len(text) <= max_chars else text[-max_chars:]


def build_episodic_rag_turn_prompt(
    profile: dict,
    task_context: str,
    history_window: list[dict],
    assistant_reply: str,
    retrieved_memories: list[EpisodicMemoryRecord],
    prompt_version: str = "episodic_rag",
) -> str:
    """Build an LLM prompt that predicts a full 1-5 score from raw memories."""
    if prompt_version == "episodic_rag_boundary_first":
        return build_episodic_rag_boundary_first_prompt(
            profile=profile,
            task_context=task_context,
            history_window=history_window,
            assistant_reply=assistant_reply,
            retrieved_memories=retrieved_memories,
        )
    if prompt_version != "episodic_rag":
        raise ValueError(f"Unsupported episodic-RAG prompt version: {prompt_version}")

    return f"""你正在预测一个具体用户对当前助手回复的满意度。

核心设定：
- 输出 1-5 分整数，1 最差，5 最好。
- 3/4 是最重要的满意/不满意边界：1-3 表示不满意，4-5 表示满意。
- 4/5 是满意程度细分：只有当回复明显超过用户历史中 4 分样例时才给 5。
- 历史证据不是总结，而是该用户过往每个已标注 assistant turn 的原始记忆检索结果；请优先比较当前回复与这些相似历史样例的质量差异。
- 不要因为检索证据里高分或低分更多就机械跟随比例；重点判断当前回复相对证据的具体优劣。

{_format_reason_rule_block()}

【用户画像】
{_format_profile(profile)}

【当前任务背景】
{task_context or "（无）"}

【当前对话上下文】
{format_history_window(history_window)}

【待评分助手回复】
{_truncate(assistant_reply, 1800)}

【检索到的该用户历史原始记忆证据】
{format_episodic_memories(retrieved_memories)}

请按以下步骤判断，但最终只输出 JSON：
1. 先判断当前回复是否跨过 3/4 满意边界。
2. 如果未跨过边界，在 1/2/3 中选择，并给出不满意原因。
3. 如果跨过边界，再比较是否只是合格满意 4，还是明显优于历史 4 分样例、可给 5。
4. reason 必须遵守：{_format_reason_json_rule()}。

输出严格 JSON，不要 Markdown，不要额外文字：
{{
  "classification": 4,
  "reason": "满意",
  "analysis": "用1-3句话说明最关键证据和边界判断。",
  "boundary_side": "sat",
  "evidence_confidence": "medium"
}}

字段约束：
- classification: 1/2/3/4/5
- boundary_side: "sat" 或 "dsat"，必须与 classification>=4 或 <=3 一致
- evidence_confidence: "low" / "medium" / "high"，表示检索证据对本次判断的支持强度
"""


def build_episodic_rag_boundary_first_prompt(
    profile: dict,
    task_context: str,
    history_window: list[dict],
    assistant_reply: str,
    retrieved_memories: list[EpisodicMemoryRecord],
) -> str:
    return f"""你正在基于该用户的原始历史记忆，预测当前助手回复的满意度。

这次必须采用 boundary-first 流程，不能直接凭整体印象给 4 分：

Phase 1：只判断 3/4 满意边界
- SAT: 当前回复足以让该用户满意，最终只能是 4 或 5。
- DSAT: 当前回复未满足该用户要求，最终只能是 1、2 或 3。
- 判断边界时优先比较检索证据中的 DSAT 样例与 SAT 样例：当前回复更像哪一侧？
- 如果当前回复存在与历史 DSAT 样例相同的实质缺陷，不要因为格式完整就判 SAT。

Phase 2：只在边界内部细分
- 如果 Phase 1=DSAT：
  - severe_dsat -> 1：不可用、严重偏离需求、几乎不能帮助用户。
  - clear_dsat -> 2：有明显帮助但关键需求缺失或错误。
  - near_boundary_dsat -> 3：接近满意但仍有实质短板。
- 如果 Phase 1=SAT：
  - qualified_sat -> 4：满足需求，但没有明显超过该用户历史 4 分样例。
  - strong_sat -> 5：明显接近或超过该用户历史 5 分样例，具体、可执行、贴合偏好。

重要约束：
- 不允许默认输出 4；必须先给出 SAT/DSAT 边界证据。
- 只有强证据表明“满足需求但不卓越”时才输出 4。
- 如果检索证据中同时有 SAT 和 DSAT，请指出最关键的对比证据 id。
- reason 必须遵守：{_format_reason_json_rule()}。

{_format_reason_rule_block()}

【用户画像】
{_format_profile(profile)}

【当前任务背景】
{task_context or "（无）"}

【当前对话上下文】
{format_history_window(history_window)}

【待评分助手回复】
{_truncate(assistant_reply, 1800)}

【检索到的该用户历史原始记忆证据】
{format_episodic_memories(retrieved_memories)}

只输出严格 JSON，不要 Markdown，不要额外文字：
{{
  "boundary_decision": "sat",
  "boundary_confidence": "medium",
  "score_refinement": "qualified_sat",
  "classification": 4,
  "reason": "满意",
  "analysis": "先说明3/4边界证据，再说明为什么细分到该分数。",
  "key_evidence_ids": ["User_0__任务__0.json__turn_0"]
}}

字段约束：
- boundary_decision: "sat" 或 "dsat"
- boundary_confidence: "low" / "medium" / "high"
- score_refinement: severe_dsat / clear_dsat / near_boundary_dsat / qualified_sat / strong_sat
- classification 必须与 score_refinement 完全一致：
  severe_dsat=1, clear_dsat=2, near_boundary_dsat=3, qualified_sat=4, strong_sat=5
- classification>=4 时 reason 必须是 "满意"；classification<=3 时 reason 必须是不满意原因
"""
