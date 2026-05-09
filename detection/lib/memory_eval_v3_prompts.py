from __future__ import annotations

from typing import Literal

from .satisfaction_constants import SATISFIED_REASON, get_reason_to_id
from .memory_schema import UserMemory
from .memory_formatting import (
    _format_profile,
    _format_reason_json_rule,
    _format_reason_rule_block,
)
from .memory_eval_common import _format_anchor_turns


def build_turn_eval_v3_prompt(
    memory: UserMemory,
    profile: dict,
    task_context: str,
    history_window: list[str],
    assistant_reply: str,
    anchor_turns: list | None = None,
    prompt_version: str = "v2",
) -> str:
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

