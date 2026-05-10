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


def build_turn_eval_base_prompt(
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
    # Anchor runs use a rank-match prior before applying the rubric sanity check.
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

