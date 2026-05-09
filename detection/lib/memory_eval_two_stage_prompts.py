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


