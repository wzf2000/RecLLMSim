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


def build_turn_eval_history_prior_prompt(
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

    if prompt_version in {"history_prior_delta_v2", "history_prior_delta_v3", "history_prior_delta_v3_1", "history_prior_delta_v3_episodic"}:
        is_v3 = prompt_version in {"history_prior_delta_v3", "history_prior_delta_v3_episodic"}
        is_v3_1 = prompt_version == "history_prior_delta_v3_1"
        is_v3_family = is_v3 or is_v3_1
        is_episodic = prompt_version == "history_prior_delta_v3_episodic"
        anchor_instruction = (
            "【参考案例使用规则】\n"
            + (
                "1. 这些案例来自该用户历史真实标注，是 episodic memory evidence；优先把它们用于 3/4 边界成对比较。\n"
                "2. 同时比较 DSAT-side evidence（<=3）与 SAT-side evidence（>=4）：当前回复更像哪一侧，必须结合具体缺陷/满足点说明。\n"
                "3. 不要机械复制案例分数；summary memory 给用户整体先验，episodic evidence 给当前 turn 的具体相似证据。\n"
                "4. 只有当当前回复与历史证据的差异具体、可引用，才给 high confidence。\n"
                if is_episodic else
                "1. 参考案例用于判断当前回复相对 history prior 的排序位置，而不是直接复制案例分数。\n"
                "2. 同时找最像当前回复的高分案例和低分案例，比较当前回复更接近哪一侧。\n"
                "3. 只有当当前回复相对历史常态明显更差/更好，且证据具体，才给 high confidence。\n"
            )
            + (
                "4. 对 v3 来说，boundary_score=3、classification<=3、delta_score<0 会作为程序侧 DSAT 投票信号；"
                "若当前回复确实未过满意线，请不要因为用户历史均分高而回避这些信号。\n\n"
                if is_v3_family and not is_episodic else
                "5. 对 v3 episodic 来说，boundary_score=3、classification<=3、delta_score<0 会作为程序侧 DSAT 投票信号；"
                "若 episodic evidence 显示当前回复更接近 <=3 一侧，请如实输出这些信号；若更接近 >=4 一侧，也不要被单个低分案例过度拉低。\n\n"
                if is_episodic else "\n"
            )
            if anchor_turns else ""
        )
        prompt = (
            "你是一名个性化满意度 residual/boundary judge。请把用户历史先验和当前回复质量分开判断。\n"
            + (
                "本版本会同时使用 summary memory 与 episodic memory：summary memory 提供用户整体评分先验，"
                "episodic memory 提供该用户历史真实标注的相似案例。请先锚定 history prior，再用成对历史证据判断当前回复是否过 3/4 满意线。\n\n"
                if is_episodic else
                "本版本会以 history prior 为 exact-score 锚点，但会用多个 DSAT 信号投票发现未过满意线的回复；"
                "请如实输出 residual、raw classification 与 3/4 boundary，不要被历史高均分吞掉当前失败证据。\n\n"
                if is_v3_family else
                "本版本不会让你的 delta 机械决定最终分；程序会以 history prior 为默认分数，"
                "只在你给出足够置信的 residual 或 boundary 证据时才移动/约束分数。\n\n"
            )
            + f"【History Prior（默认 exact-score 锚点）】\n"
            f"history_prior_score = {memory.avg_satisfaction_score:.2f}\n"
            f"历史分布：5分×{memory.score_distribution.score_5} / "
            f"4分×{memory.score_distribution.score_4} / "
            f"3分×{memory.score_distribution.score_3} / "
            f"2分×{memory.score_distribution.score_2} / "
            f"1分×{memory.score_distribution.score_1}\n"
            f"评分风格：{memory.scoring_style}\n\n"
            f"【个性化边界】\n"
            f"3→4 满意最低线：{memory.three_vs_four_distinction}\n"
            f"4→5 更高要求：{memory.four_vs_five_distinction}\n\n"
            f"【真正影响评分的个性化要求】\n{user_reqs}\n"
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
            + "【判断步骤】\n"
            + "Step 1. 固定 history_prior_score：直接使用上面给出的历史平均分，不要自行改写。\n"
            + "Step 2. 判断 residual delta：当前回复相对该用户历史常态是 below / around / above。\n"
            + "  - delta_score=-2：明显低于常态，存在严重关键失败。\n"
            + "  - delta_score=-1：低于常态，有一个清楚的关键缺口。\n"
            + "  - delta_score=0：大致符合常态。\n"
            + "  - delta_score=1：高于常态，更完整或更贴合需求。\n"
            + "  - delta_score=2：显著高于常态，接近历史强满意样本。\n"
            + "Step 3. 给 delta_confidence：\n"
            + "  - high：有具体、可引用的证据说明当前回复明显偏离历史常态。\n"
            + "  - medium：有方向性证据，但偏离不大或证据混合。\n"
            + "  - low：主要只是主观感觉、证据不足，或当前回复接近历史常态。\n"
            + "Step 4. 单独判断 3/4 boundary，并给 boundary_confidence：\n"
            + "  - high：核心问题是否满足非常明确；可以安全地作为硬约束。\n"
            + "  - medium：更像某一边，但仍有混合证据。\n"
            + "  - low：不确定，不能作为硬约束。\n"
            + "Step 5. 标记 strong evidence：\n"
            + "  - strong_failure_evidence=true 只在存在明确关键失败时使用，例如核心问题没答、关键约束被漏、内容空泛到不可用。\n"
            + "  - strong_excellence_evidence=true 只在明显超过该用户常态时使用，例如更完整、更贴合个性化要求、比相近历史高分案例更强。\n"
            + "Step 6. classification 填你按 soft reconstruction 直觉得到的诊断分，但最终 pred_score 会由程序重建。\n"
            + (
                "  v3.1 程序默认使用 round(history_prior_score)；只有 boundary_score=3、classification<=3、delta_score<0 三个 DSAT 信号全部成立时，"
                "才会把最终分约束到 <=3；两个信号只作诊断，不直接触发降分。否则仅在 high residual 或 medium 且 |delta_score|=2 时移动一档，"
                "并且只用 high-confidence boundary_score=4 做 >=4 约束。\n"
                if is_v3_1 else
                "  v3 程序默认使用 round(history_prior_score)；若 boundary_score=3、classification<=3、delta_score<0 中至少两个成立，"
                "会把最终分约束到 <=3；否则仅在 high residual 或 medium 且 |delta_score|=2 时移动一档，"
                "并且只用 high-confidence boundary_score=4 做 >=4 约束。\n"
                if is_v3 else
                "  程序默认使用 round(history_prior_score)，只在 high residual 或 medium 且 |delta_score|=2 时移动一档；"
                "只有 high-confidence boundary 才会强制 <=3 或 >=4。\n"
            )
            + "Step 7. 选择 reason：若你的诊断最终分 >=4，reason 必须是 `满意`；若 <=3，reason 必须是不满意原因。\n\n"
            + "注意：\n"
            + "- 不要把 ordinary improvement 写成 high confidence；high 必须有具体证据。\n"
            + "- 不要把“不是 5 分水平”当成未过 3/4 满意线。\n"
            + "- boundary_confidence=high 应该谨慎使用；只有核心满意/不满意非常明确时才给 high。\n\n"
            + "重要：不要输出 <think>、推理草稿、Markdown、解释文字或任何 JSON 外文本；只输出一个 JSON object。\n\n"
            + "请严格输出 JSON，不要输出其他内容：\n"
            + "{\n"
            + '  "classification": 1-5 中的整数,\n'
            + f'  "reason": "{reason_json_rule}",\n'
            + '  "analysis": "简述 prior、delta 证据与置信度、boundary 证据与置信度、strong evidence 标记原因",\n'
            + f'  "history_prior_score": {memory.avg_satisfaction_score:.2f},\n'
            + '  "delta_label": "below",\n'
            + '  "delta_score": -2 到 2 的整数,\n'
            + '  "delta_confidence": "medium",\n'
            + '  "passes_satisfaction_boundary": true 或 false,\n'
            + '  "boundary_score": 只能是 3 或 4,\n'
            + '  "boundary_confidence": "medium",\n'
            + '  "strong_failure_evidence": true 或 false,\n'
            + '  "strong_excellence_evidence": true 或 false\n'
            + "}\n"
        )
        return prompt

    if prompt_version == "history_prior_delta":
        anchor_instruction = (
            "【参考案例使用规则】\n"
            "1. 参考案例只用于判断当前回复相对历史先验是更差、相近还是更好。\n"
            "2. 优先找整体质量和当前回复最接近的案例，比较它与该用户历史平均水平的相对位置。\n"
            "3. 不要直接复制案例分数；本题先输出 residual delta，再由程序转回最终分。\n\n"
            if anchor_turns else ""
        )
        prompt = (
            "你是一名个性化满意度 residual judge。请不要直接自由打 1-5 分。\n"
            "本题必须先以该用户的 history prior 为起点，判断当前回复相对先验的 delta，"
            "再判断是否通过 3/4 满意边界。程序会根据你输出的 prior + delta 和 boundary 重建最终分。\n\n"
            f"【History Prior（必须作为起点）】\n"
            f"history_prior_score = {memory.avg_satisfaction_score:.2f}\n"
            f"历史分布：5分×{memory.score_distribution.score_5} / "
            f"4分×{memory.score_distribution.score_4} / "
            f"3分×{memory.score_distribution.score_3} / "
            f"2分×{memory.score_distribution.score_2} / "
            f"1分×{memory.score_distribution.score_1}\n"
            f"评分风格：{memory.scoring_style}\n\n"
            f"【个性化边界】\n"
            f"3→4 满意最低线：{memory.three_vs_four_distinction}\n"
            f"4→5 更高要求：{memory.four_vs_five_distinction}\n\n"
            f"【真正影响评分的个性化要求】\n{user_reqs}\n"
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
            + "【判断步骤】\n"
            + "Step 1. 固定 history_prior_score：直接使用上面给出的历史平均分，不要自行改写。\n"
            + "Step 2. 判断 residual delta：当前回复相对该用户通常得到的回复，是 below / around / above？\n"
            + "  - delta_score=-2：明显低于该用户历史常态，存在严重关键失败。\n"
            + "  - delta_score=-1：低于常态，有一个清楚的关键缺口或可用性损失。\n"
            + "  - delta_score=0：大致符合该用户历史常态。\n"
            + "  - delta_score=1：高于常态，明显更贴合需求或更完整。\n"
            + "  - delta_score=2：显著高于常态，接近该用户历史中的强满意样本。\n"
            + "Step 3. 单独判断 3/4 boundary：当前回复是否达到该用户的最低满意线？\n"
            + "  - 若核心问题没被回答、关键约束被忽略、内容空泛到影响使用，passes_satisfaction_boundary=false，boundary_score=3。\n"
            + "  - 若核心问题已回答且关键要求基本满足，passes_satisfaction_boundary=true，boundary_score=4。\n"
            + "Step 4. 最终分由程序重建：round(history_prior_score + delta_score) 后裁剪到 1-5；"
            + "若 boundary_score=3 则最终不超过3，若 boundary_score=4 则最终不低于4。\n"
            + "你仍需在 classification 中填入你按此规则得到的最终分，方便诊断。\n"
            + "Step 5. 选择 reason：若最终分 >=4，reason 必须是 `满意`；若最终分 <=3，reason 必须是不满意原因。\n\n"
            + "注意：\n"
            + "- history prior 解释用户整体偏高分或偏低分，delta 才解释当前回复比常态好/差。\n"
            + "- boundary 是硬约束：没过最低满意线时，即使 prior 很高也不能给 4/5；已过最低满意线时不能给 1/2/3。\n"
            + "- 不要把 5 分门槛误当成 4 分门槛；不够完美不等于未满意。\n\n"
            + "请严格输出 JSON，不要输出其他内容：\n"
            + "{\n"
            + '  "classification": 1-5 中的整数,\n'
            + f'  "reason": "{reason_json_rule}",\n'
            + '  "analysis": "按 Step1-5 简述：prior 是多少；delta 证据；是否过 3/4 boundary；最终分如何由 prior+delta+boundary 得到",\n'
            + f'  "history_prior_score": {memory.avg_satisfaction_score:.2f},\n'
            + '  "delta_label": "below",\n'
            + '  "delta_score": -2 到 2 的整数,\n'
            + '  "passes_satisfaction_boundary": true 或 false,\n'
            + '  "boundary_score": 只能是 3 或 4\n'
            + "}\n"
        )
        return prompt


def build_turn_eval_history_prior_episodic_refine_prompt(
    memory: UserMemory,
    profile: dict,
    task_context: str,
    history_window: list[str],
    assistant_reply: str,
    first_pass: dict,
    anchor_turns: list,
) -> str:
    reason_labels = list(get_reason_to_id().keys())
    reason_text = "、".join(reason_labels)
    reason_rule_block = _format_reason_rule_block()
    reason_json_rule = _format_reason_json_rule()
    history_text = "\n".join(history_window) if history_window else "（无历史）"
    anchor_block = _format_anchor_turns(anchor_turns)

    user_reqs = "\n".join(
        f"  - {r}" for r in memory.user_specific_requirements
    ) if memory.user_specific_requirements else "  （无特异性要求记录）"

    first_pass_block = (
        f"classification={first_pass.get('classification')}; "
        f"final_score={first_pass.get('final_score')}; "
        f"reason={first_pass.get('reason')}; "
        f"history_prior_score={first_pass.get('history_prior_score')}; "
        f"delta_label={first_pass.get('delta_label')}; "
        f"delta_score={first_pass.get('delta_score')}; "
        f"delta_confidence={first_pass.get('delta_confidence')}; "
        f"boundary_score={first_pass.get('boundary_score')}; "
        f"boundary_confidence={first_pass.get('boundary_confidence')}; "
        f"strong_failure_evidence={first_pass.get('strong_failure_evidence')}; "
        f"strong_excellence_evidence={first_pass.get('strong_excellence_evidence')}; "
        f"dsat_votes={first_pass.get('dsat_votes')}; "
        f"analysis={first_pass.get('analysis')}"
    )

    prompt = (
        "你是一名 episodic memory boundary reviewer。第一遍 judge 已经给出 history-prior 判断；"
        "你只需要基于该用户历史真实标注案例，复核当前回复更接近 3/4 边界哪一侧。\n\n"
        "【重要原则】\n"
        "1. summary memory 只作为背景；本轮重点是比较当前回复和 episodic evidence 的具体相似失败/满足点。\n"
        "2. 不要因为一个低分案例就判 3；只有当前回复的核心失败模式与 DSAT-side evidence 具体相同或更严重，才选择 dsat。\n"
        "3. 不要因为回复不完美就判 3；若核心问题已回答且更接近 SAT-side evidence，选择 sat。\n"
        "4. 若两侧都有相似点、证据不足或只是表面词相似，选择 mixed，confidence=low/medium。\n\n"
        f"【History Prior】\n"
        f"history_prior_score={memory.avg_satisfaction_score:.2f}\n"
        f"历史分布：5分×{memory.score_distribution.score_5} / "
        f"4分×{memory.score_distribution.score_4} / "
        f"3分×{memory.score_distribution.score_3} / "
        f"2分×{memory.score_distribution.score_2} / "
        f"1分×{memory.score_distribution.score_1}\n"
        f"评分风格：{memory.scoring_style}\n"
        f"3→4 满意最低线：{memory.three_vs_four_distinction}\n"
        f"4→5 更高要求：{memory.four_vs_five_distinction}\n"
        f"用户特异要求：\n{user_reqs}\n"
        f"偏好回复形式：{memory.preferred_response_format}\n\n"
        f"【第一遍判断】\n{first_pass_block}\n\n"
        f"{anchor_block}\n\n"
        f"【用户画像】{_format_profile(profile)}\n\n"
        f"【任务背景】{task_context}\n\n"
        f"【最近对话历史】\n{history_text}\n\n"
        f"【待复核的助手回复】\n{assistant_reply}\n\n"
        f"【可选原因标签】{reason_text}\n{reason_rule_block}\n"
        "【输出要求】\n"
        "- closest_evidence_side=\"dsat\"：当前回复的核心失败模式更接近 DSAT-side evidence，且会影响最低满意线。\n"
        "- closest_evidence_side=\"sat\"：当前回复核心需求已满足，更接近 SAT-side evidence。\n"
        "- closest_evidence_side=\"mixed\"：两侧证据混合或检索案例不够贴近，不应改动第一遍边界。\n"
        "- evidence_match_confidence=\"high\" 只能在相似点非常具体时使用。\n"
        "- classification 只输出 3 或 4，表示 episodic evidence 建议的 3/4 边界侧。\n"
        "- reason 规则：classification=4 时必须是 `满意`；classification=3 时必须是不满意原因。\n\n"
        "重要：不要输出 <think>、Markdown、解释文字或任何 JSON 外文本；只输出一个 JSON object。\n\n"
        "请严格输出 JSON，不要输出其他内容：\n"
        "{\n"
        '  "classification": 3 或 4,\n'
        f'  "reason": "{reason_json_rule}",\n'
        '  "analysis": "简述最相似的 DSAT/SAT 证据、当前回复与其关键相同/不同点，以及为何建议 3/4 或保持 mixed",\n'
        '  "closest_evidence_side": "mixed",\n'
        '  "evidence_match_confidence": "medium"\n'
        "}\n"
    )
    return prompt
