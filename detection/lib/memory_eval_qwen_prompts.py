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


def build_turn_eval_qwen_short_prompt(
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

