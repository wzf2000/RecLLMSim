from __future__ import annotations

from typing import Literal

from .satisfaction_constants import SATISFIED_REASON, get_reason_to_id
from .memory_schema import UserMemory
from .memory_formatting import (
    _MAX_REPLY_CHARS,
    _format_profile,
    _format_reason_json_rule,
    _format_reason_rule_block,
    _truncate,
)
from .memory_eval_common import _format_anchor_turns


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


