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


