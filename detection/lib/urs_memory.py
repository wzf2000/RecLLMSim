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
    _format_profile,
    _format_reason_json_rule,
    _format_reason_rule_block,
    _truncate,
)
from .satisfaction_constants import get_reason_to_id


def _format_session_dialogue(history: list[dict], max_chars: int = 400) -> str:
    """把整段对话渲染成 "用户：... / 助手：..." 形式的多行文本。"""
    lines: list[str] = []
    for utt in history:
        role_label = "用户" if utt.get("role") == "user" else "助手"
        content = _truncate(utt.get("content", ""), max_chars)
        lines.append(f"{role_label}：{content}")
    return "\n".join(lines) if lines else "（空对话）"


def build_session_eval_prompt(
    memory: UserMemory,
    profile: dict,
    task_context: str,
    session_history: list[dict],
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

    prompt = (
        "你是一名个性化满意度评估员。\n"
        "请对下面【整段对话】给出一个 1-5 的整数分数，反映用户对本次对话的整体满意度。\n"
        "注意：是对整段对话打 1 个分数，不是逐轮打分。\n\n"
        f"{rubric}\n"
        f"【用户画像】{_format_profile(profile)}\n\n"
        f"【任务背景】{task_context}\n\n"
        f"【完整对话】\n{dialogue_text}\n\n"
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
) -> str:
    """URS session-level 评分 prompt — 无记忆 baseline。"""
    reason_labels = list(get_reason_to_id().keys())
    reason_text = "、".join(reason_labels)
    reason_rule_block = _format_reason_rule_block()
    reason_json_rule = _format_reason_json_rule()
    dialogue_text = _format_session_dialogue(session_history)

    prompt = (
        "你是一名会进行细粒度对话质量分析的评估员。\n"
        "请对下面【整段对话】给出一个 1-5 的整数分数，反映用户对本次对话的整体满意度。\n"
        "同时选取一个最贴切的原因标签（分数 >=4 必须为【满意】；分数 <=3 选择一个不满意原因）。\n\n"
        f"【用户画像】{_format_profile(profile)}\n\n"
        f"【任务背景】{task_context}\n\n"
        f"【完整对话】\n{dialogue_text}\n\n"
        f"【可选原因标签】{reason_text}\n{reason_rule_block}\n"
        "请严格输出 JSON，不要输出其他内容：\n"
        "{\n"
        '  "classification": 1-5 中的整数（对整段对话的单一分数）,\n'
        f'  "reason": "{reason_json_rule}",\n'
        '  "analysis": "你的详细推理过程"\n'
        "}\n"
    )
    return prompt
