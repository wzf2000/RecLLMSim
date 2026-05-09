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


def build_turn_eval_boundary_prompt(
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

    if prompt_version == "boundary_34_selective_refute_v4":
        anchor_instruction = (
            "【参考案例使用规则】\n"
            "1. 只把案例当作 3/4 边界参考：真实分数 <=3 是【未达满意线案例】，>=4 是【达到满意线案例】。\n"
            "2. 不要只看一边的案例。若当前回复更像未达满意线案例，要敢于判 `3`；若更像达到满意线案例，也不要因不够优秀就压成 `3`。\n"
            "3. 案例用于校准边界，不用于追求 5 分标准。\n\n"
            if anchor_turns else ""
        )
        prompt = (
            "你是一名个性化满意度边界评估员。\n"
            "这是 selective-refute v4 的第一遍初判：目标是尽可能平衡地判断当前助手回复是否达到该用户的【满意最低线】。\n"
            "输出只能是：\n"
            "- `4` = 满意（达到最低满意线）\n"
            "- `3` = 不满意（未达到最低满意线）\n\n"
            "本题不要默认保护 `4`，也不要默认压成 `3`。你必须同时考虑：\n"
            "- 最强的“为什么它应该是 `3`”的证据\n"
            "- 最强的“为什么它至少已经到 `4`”的证据\n"
            "再决定哪一边更强。\n\n"
            f"【用户评分摘要】\n"
            f"评分风格：{memory.scoring_style}\n"
            f"历史平均分：{memory.avg_satisfaction_score:.2f}\n"
            f"满意最低线（3→4 边界）：{memory.three_vs_four_distinction}\n"
            f"更高要求（4→5，仅供背景参考）：{memory.four_vs_five_distinction}\n"
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
            + "【第一遍平衡边界判断规则】\n"
            + "Step 1. 先写出最强的 `3` 证据：\n"
            + "  - 核心问题是否未被回答？\n"
            + "  - 关键约束 / 关键任务目标 / 用户特别在意的要求是否被漏掉？\n"
            + "  - 缺口是否已经明显影响可用性，导致用户仍会觉得没被满足？\n"
            + "Step 2. 再写出最强的 `4` 证据：\n"
            + "  - 核心问题是否已经被回答？\n"
            + "  - 关键要求是否已经基本满足？\n"
            + "  - 剩余问题是否只是普通缺口，而不阻止用户把它当作基本满意的回复？\n"
            + "Step 3. 明确比较这两边哪一边更强：\n"
            + "  - 若最强的 `3` 证据更强，判 `3`\n"
            + "  - 若最强的 `4` 证据更强，判 `4`\n"
            + "Step 4. 只有当两边最强证据真的势均力敌时，才允许 `needs_refute_review=true`。\n"
            + "  - 明显偏向任一边时必须输出 `needs_refute_review=false`\n\n"
            + "注意：\n"
            + "- `不够细致` 既可能只是普通缺口，也可能已经影响可用性；不要默认把它归到任何一边。\n"
            + "- 友好语气、表面帮助性不能替代“核心问题是否真正回答”。\n"
            + "- 不要因为不是 5 分水平就压成 `3`，也不要因为看起来有帮助就放成 `4`。\n"
            + "- `classification` 只能输出 `3` 或 `4`。\n"
            + "- `analysis` 只需 1-2 句，必须同时提到：最强的 `3` 证据、最强的 `4` 证据，以及最终哪一边更强。\n\n"
            + "请严格输出 JSON，不要输出其他内容：\n"
            + "{\n"
            + '  "classification": 只能是 3 或 4,\n'
            + f'  "reason": "{reason_json_rule}",\n'
            + '  "analysis": "用 1-2 句写明：最强的 3 证据是什么，最强的 4 证据是什么，最终哪一边更强，以及是否需要复核",\n'
            + '  "needs_refute_review": true 或 false\n'
            + "}\n"
        )
        return prompt

    if prompt_version == "boundary_34_selective_refute_v3":
        anchor_instruction = (
            "【参考案例使用规则】\n"
            "1. 只把案例当作 3/4 边界参考：真实分数 <=3 是【未达满意线案例】，>=4 是【达到满意线案例】。\n"
            "2. 优先比较当前回复是否已经达到“这个用户愿意认为它基本有用、基本满意”的最低线，而不是和优秀案例比完整度。\n"
            "3. 若当前回复已经明显站在某一边，不要因为它不够优秀就把它拖回边界附近。\n\n"
            if anchor_turns else ""
        )
        prompt = (
            "你是一名个性化满意度边界评估员。\n"
            "这是 selective-refute v3 的第一遍初判：先尽可能准确地判断当前助手回复是否达到该用户的【满意最低线】。\n"
            "输出只能是：\n"
            "- `4` = 满意（达到最低满意线）\n"
            "- `3` = 不满意（未达到最低满意线）\n\n"
            "这一步最重要的不是区分“优秀”和“一般”，而是区分：\n"
            "- 只是普通缺口、还不够细，但已经达到最低满意线\n"
            "- 真正没过满意线，用户仍会觉得不满意\n\n"
            f"【用户评分摘要】\n"
            f"评分风格：{memory.scoring_style}\n"
            f"历史平均分：{memory.avg_satisfaction_score:.2f}\n"
            f"满意最低线（3→4 边界）：{memory.three_vs_four_distinction}\n"
            f"更高要求（4→5，仅供背景参考）：{memory.four_vs_five_distinction}\n"
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
            + "【第一遍边界判断规则】\n"
            + "Step 1. 先判断：回复是否真正回答了用户此刻最核心的问题。\n"
            + "  - 若核心问题没有被回答，优先判 `3`。\n"
            + "Step 2. 再判断：关键约束、关键任务目标、该用户特别在意的要求，是否至少被基本满足。\n"
            + "  - 若关键要求被漏掉，且这会明显影响可用性，优先判 `3`。\n"
            + "Step 3. 只有在核心问题已回答、关键要求也基本满足时，才去看剩余缺口属于哪类：\n"
            + "  - 【普通缺口】= 细节不足、还可更完整、还可更个性化，但不妨碍用户把它当作基本满意的答复，此时应判 `4`\n"
            + "  - 【关键缺口】= 缺失会让用户仍觉得没被满足、没法直接用、或明显偏离要求，此时应判 `3`\n"
            + "Step 4. 只有当你真的无法判断某个唯一可疑点到底是普通缺口还是关键缺口时，才允许 `needs_refute_review=true`。\n"
            + "  - 明显满意或明显不满意都必须输出 `needs_refute_review=false`\n"
            + "  - `needs_refute_review=true` 必须是少数情况\n\n"
            + "注意：\n"
            + "- `不够细致` 默认更接近【普通缺口】，除非它已经严重到让回复不可用或明显没满足核心要求。\n"
            + "- 不要因为它不是 5 分水平，就把一个本来已经过线的回复判成 3。\n"
            + "- 也不要因为回复语气友好、表面在帮忙，就把一个没回答核心问题的回复判成 4。\n"
            + "- `classification` 只能输出 `3` 或 `4`。\n"
            + "- `analysis` 只需 1-2 句，明确写出：核心问题是否被回答；关键要求是否被满足；当前可疑点为何属于普通缺口或关键缺口。\n\n"
            + "请严格输出 JSON，不要输出其他内容：\n"
            + "{\n"
            + '  "classification": 只能是 3 或 4,\n'
            + f'  "reason": "{reason_json_rule}",\n'
            + '  "analysis": "用 1-2 句写明：核心问题是否被回答，关键要求是否被满足，当前可疑点为何属于普通缺口或关键缺口，以及是否需要复核",\n'
            + '  "needs_refute_review": true 或 false\n'
            + "}\n"
        )
        return prompt

    if prompt_version == "boundary_34_selective_refute_v2":
        anchor_instruction = (
            "【参考案例使用规则】\n"
            "1. 只把案例理解为边界参考：真实分数 <=3 是【未达满意线案例】，>=4 是【达到满意线案例】。\n"
            "2. 只有当当前回复与两类案例都存在明显相似点、边界仍拿不准时，才考虑触发复核。\n"
            "3. 若当前回复整体明显站在某一边，就不要触发复核。\n\n"
            if anchor_turns else ""
        )
        prompt = (
            "你是一名个性化满意度边界评估员。\n"
            "这是 selective-refute v2 的第一遍初判：先温和判断当前助手回复是否达到该用户的【满意最低线】。\n"
            "输出只能是：\n"
            "- `4` = 满意（达到最低满意线）\n"
            "- `3` = 不满意（未达到最低满意线）\n\n"
            "除分数外，你还需要判断：这个样本是否【高度接近 3/4 边界】，需要进入第二遍复核。\n"
            "注意，`needs_refute_review=true` 必须是少数情况；只有在你确实拿不准时才允许触发。\n\n"
            f"【用户评分摘要】\n"
            f"评分风格：{memory.scoring_style}\n"
            f"历史平均分：{memory.avg_satisfaction_score:.2f}\n"
            f"满意最低线（3→4 边界）：{memory.three_vs_four_distinction}\n"
            f"更高要求（4→5，仅供背景参考）：{memory.four_vs_five_distinction}\n"
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
            + "【第一遍只做严格筛选后的边界判断】\n"
            + "Step 1. 判断回复是否回答了核心问题，并基本满足关键约束。\n"
            + "Step 2. 判断它是否达到该用户的满意最低线：达到给 `4`，未达到给 `3`。\n"
            + "Step 3. 再判断是否真的需要复核。只有下面两类高不确定情形才允许 `needs_refute_review=true`：\n"
            + "  - 当前判成 `3`，但你怀疑问题主要只是“边缘性的细节不足”，未必真的低于满意线\n"
            + "  - 当前判成 `4`，但你怀疑它可能漏掉了一个关键要求，是否仍算满意拿不准\n"
            + "Step 4. 若主要证据已经明显站在一边，必须输出 `needs_refute_review=false`。\n\n"
            + "注意：\n"
            + "- 不要因为“还可以更好”就触发复核。\n"
            + "- 不要因为理由是 `不够细致` 就自动触发复核。\n"
            + "- 只有当一个具体可疑点是否属于关键失败拿不准时，才触发复核。\n"
            + "- `classification` 只能输出 `3` 或 `4`。\n"
            + "- `analysis` 只需 1-2 句，明确写出：当前边界判断是什么；可疑点是什么；是否真的需要复核。\n\n"
            + "请严格输出 JSON，不要输出其他内容：\n"
            + "{\n"
            + '  "classification": 只能是 3 或 4,\n'
            + f'  "reason": "{reason_json_rule}",\n'
            + '  "analysis": "用 1-2 句写明当前为何判为 3 或 4、唯一的可疑点是什么，以及是否真的需要复核",\n'
            + '  "needs_refute_review": true 或 false\n'
            + "}\n"
        )
        return prompt

    if prompt_version == "boundary_34_selective_refute":
        anchor_instruction = (
            "【参考案例使用规则】\n"
            "1. 只把案例看成边界参考：真实分数 <=3 视为【未达满意线案例】，>=4 视为【达到满意线案例】。\n"
            "2. 案例只用于帮助你判断当前回复是否接近 3/4 边界，不要机械复用案例分数。\n"
            "3. 若当前回复明显优于未达满意线案例，或明显达到满意线，就不要触发二次复核。\n\n"
            if anchor_turns else ""
        )
        prompt = (
            "你是一名个性化满意度边界评估员。\n"
            "这是 selective-refute 的第一遍初判：先温和判断当前助手回复是否达到该用户的【满意最低线】。\n"
            "输出只能是：\n"
            "- `4` = 满意（达到最低满意线）\n"
            "- `3` = 不满意（未达到最低满意线）\n\n"
            "除分数外，你还需要判断这个样本是否【真的接近 3/4 边界】，从而需要进入二次反证复核。\n"
            "只有在证据混合、边界不稳时，才把 `needs_refute_review` 设为 `true`；明显满意或明显不满意都应设为 `false`。\n\n"
            f"【用户评分摘要】\n"
            f"评分风格：{memory.scoring_style}\n"
            f"历史平均分：{memory.avg_satisfaction_score:.2f}\n"
            f"满意最低线（3→4 边界）：{memory.three_vs_four_distinction}\n"
            f"更高要求（4→5，仅供背景参考）：{memory.four_vs_five_distinction}\n"
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
            + "【第一遍只做温和边界判断】\n"
            + "Step 1. 判断回复是否回答了核心问题，并基本满足关键约束。\n"
            + "Step 2. 判断它是否达到该用户的满意最低线：达到给 `4`，未达到给 `3`。\n"
            + "Step 3. 再判断这个案例是否【真的接近边界】。\n"
            + "  只有下面情况才把 `needs_refute_review=true`：\n"
            + "  - 回复大体有帮助，但有一个可能是关键缺陷的点，是否足以掉到 3 不确定\n"
            + "  - 当前判成 3，但主要问题可能只是“不够细致”，未必真的低于满意线\n"
            + "  - 当前判成 4，但可能漏掉了一个关键要求，是否仍算满意不确定\n"
            + "Step 4. 若结论已经很明显，就输出 `needs_refute_review=false`。\n\n"
            + "注意：\n"
            + "- `needs_refute_review=true` 应该是少数情况，不要把它当默认值。\n"
            + "- 不要因为回复不够优秀就自动触发复核；只有接近 3/4 边界时才触发。\n"
            + "- `classification` 只能输出 `3` 或 `4`。\n"
            + "- `analysis` 只需 1-2 句，写明当前判断依据，以及为什么需要或不需要二次复核。\n\n"
            + "请严格输出 JSON，不要输出其他内容：\n"
            + "{\n"
            + '  "classification": 只能是 3 或 4,\n'
            + f'  "reason": "{reason_json_rule}",\n'
            + '  "analysis": "用 1-2 句写明当前为何判为 3 或 4，以及是否接近 3/4 边界",\n'
            + '  "needs_refute_review": true 或 false\n'
            + "}\n"
        )
        return prompt

    if prompt_version == "boundary_34_refute_v2":
        anchor_instruction = (
            "【参考案例使用规则】\n"
            "1. 先把案例按边界用途理解：真实分数 <=3 是【未达满意线案例】，>=4 是【达到满意线案例】。\n"
            "2. 优先观察未达满意线案例中的【致命缺陷】是什么，再看达到满意线案例是否只是存在可改进的小缺口。\n"
            "3. 不要因为当前回复不如优秀案例完整，就直接判成 3。\n\n"
            if anchor_turns else ""
        )
        prompt = (
            "你是一名个性化满意度边界评估员。\n"
            "本题只判断当前助手回复是否达到该用户的【满意最低线】。\n"
            "输出只能是：\n"
            "- `4` = 满意（达到最低满意线）\n"
            "- `3` = 不满意（未达到最低满意线）\n\n"
            "你需要保留【反证检查】，但采用更温和的判定原则：\n"
            "只有当存在【明确且关键的失败】时，才允许判 `3`。\n"
            "如果回复已经回答了核心问题，关键约束也基本满足，而剩余问题只是“还不够细”“还可以更好”，应优先判 `4`。\n\n"
            f"【用户评分摘要】\n"
            f"评分风格：{memory.scoring_style}\n"
            f"历史平均分：{memory.avg_satisfaction_score:.2f}\n"
            f"满意最低线（3→4 边界）：{memory.three_vs_four_distinction}\n"
            f"更高要求（4→5，仅供背景参考）：{memory.four_vs_five_distinction}\n"
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
            + "【先做温和反证，再决定是否给 3】\n"
            + "Step 1. 先判断回复是否已经基本回答了用户的核心问题，并满足关键约束。\n"
            + "Step 2. 再检查是否存在【明确且关键的失败】。只有下面这些情况才足以判 `3`：\n"
            + "  - 没有直接回答主要问题\n"
            + "  - 明显忽略关键约束、条件或任务目标\n"
            + "  - 内容过于空泛，用户几乎无法据此采取行动\n"
            + "  - 漏掉了该用户最在意、且会显著影响满意度的要求\n"
            + "  - 存在会明显伤害可用性的缺口，而不是普通的“还不够细致”\n"
            + "Step 3. 明确区分两类问题：\n"
            + "  - 【致命缺陷】= 会让回复掉到 3\n"
            + "  - 【普通缺口】= 已达到最低满意线，但还不够优秀，仍应给 4\n"
            + "Step 4. 做简短反证：\n"
            + "  - 如果你能指出一个明确的【致命缺陷】，输出 `classification=3`\n"
            + "  - 如果只看到普通缺口，而没有致命缺陷，输出 `classification=4`\n"
            + "Step 5. 选择一个最贴切的原因标签。\n\n"
            + "注意：\n"
            + "- `不够细致` 本身不等于 `3`；只有它严重到导致回复不满足最低满意线时，才可以判 `3`。\n"
            + "- 不要因为它不如 5 分案例完整，就直接判 `3`。\n"
            + "- 如果回复已回答核心问题，且关键要求基本满足，应优先保护 `4`。\n"
            + "- `classification` 只能输出 `3` 或 `4`。\n"
            + "- `analysis` 只需简短说明：最强的降分证据是什么；它是否属于致命缺陷；最终为何判 3 或 4。\n\n"
            + "请严格输出 JSON，不要输出其他内容：\n"
            + "{\n"
            + '  "classification": 只能是 3 或 4,\n'
            + f'  "reason": "{reason_json_rule}",\n'
            + '  "analysis": "用 2-3 句写明：最强的降分证据是什么；它是否属于致命缺陷；最终为何判为 3 或 4。若只是普通缺口，应明确说明仍达到最低满意线" \n'
            + "}\n"
        )
        return prompt

    if prompt_version == "boundary_34_refute":
        anchor_instruction = (
            "【参考案例使用规则】\n"
            "1. 先把案例按边界用途理解：真实分数 <=3 是【未达满意线案例】，>=4 是【达到满意线案例】。\n"
            "2. 优先观察未达满意线案例缺了什么，再看达到满意线案例满足了什么。\n"
            "3. 不要机械复用案例分数；案例只用于帮助你发现“哪些缺口足以把回复判成 3”。\n\n"
            if anchor_turns else ""
        )
        prompt = (
            "你是一名个性化满意度边界评估员。\n"
            "本题只判断当前助手回复是否达到该用户的【满意最低线】。\n"
            "输出只能是：\n"
            "- `4` = 满意（达到最低满意线）\n"
            "- `3` = 不满意（未达到最低满意线）\n\n"
            "与旧版不同，本题必须先做【反证检查】。\n"
            "也就是说：先主动寻找足以把回复判成 `3` 的关键缺陷；"
            "只有当这些缺陷都不成立时，才允许给 `4`。\n\n"
            f"【用户评分摘要】\n"
            f"评分风格：{memory.scoring_style}\n"
            f"历史平均分：{memory.avg_satisfaction_score:.2f}\n"
            f"满意最低线（3→4 边界）：{memory.three_vs_four_distinction}\n"
            f"更高要求（4→5，仅供背景参考）：{memory.four_vs_five_distinction}\n"
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
            + "【先做失败检查，再决定是否给 4】\n"
            + "Step 1. 先检查是否存在任何一个【足以降到 3 分】的关键失败。\n"
            + "  重点检查：\n"
            + "  - 没有直接回答用户主要问题\n"
            + "  - 明显忽略关键约束、条件或任务目标\n"
            + "  - 内容太泛、太空，用户难以据此采取行动\n"
            + "  - 漏掉了该用户特别在意的要求或偏好格式\n"
            + "  - 存在会明显伤害满意度的缺口，而不只是“还可以更好”\n"
            + "Step 2. 做【反证】。\n"
            + "  问自己：如果我要把它判成 3，最强证据是什么？\n"
            + "  - 如果能找到明确且实质的证据，输出 `classification=3`\n"
            + "  - 只有当这些证据都站不住脚，才继续考虑 `classification=4`\n"
            + "Step 3. 只有同时满足下面两点，才能给 `4`：\n"
            + "  - 回复已经基本回答了用户问题，并满足关键要求\n"
            + "  - 没有发现任何一个足以把它拉回 3 的关键缺陷\n"
            + "Step 4. 选择一个最贴切的原因标签。\n\n"
            + "注意：\n"
            + "- 不要因为“语气像在帮忙”就给 4，关键是是否真正过了满意最低线。\n"
            + "- 也不要因为“还不够优秀”就给 3；只有出现了足以降到 3 的关键缺陷，才判 3。\n"
            + "- `classification` 只能输出 `3` 或 `4`。\n"
            + "- 在 `analysis` 里要明确写出：你检查过哪些降分证据，以及这些证据是否成立。\n\n"
            + "请严格输出 JSON，不要输出其他内容：\n"
            + "{\n"
            + '  "classification": 只能是 3 或 4,\n'
            + f'  "reason": "{reason_json_rule}",\n'
            + '  "analysis": "用 3-5 句写明：最可能把该回复判成 3 的关键缺陷是什么；这个缺陷是否成立；最终为什么判成 3 或 4。若使用参考案例，注明更接近未达满意线案例还是达到满意线案例" \n'
            + "}\n"
        )
        return prompt

    if prompt_version == "boundary_34":
        anchor_instruction = (
            "【参考案例使用规则】\n"
            "1. 只关心这些案例在满意边界上的含义：真实分数 <=3 视为【不满意案例】，>=4 视为【满意案例】。\n"
            "2. 不要尝试复用案例的精确分数，只判断当前回复更接近【不满意】还是【满意】。\n"
            "3. 你的任务不是判断这条回复有多优秀，而是判断：它有没有达到该用户的【最低满意线】。\n\n"
            if anchor_turns else ""
        )
        prompt = (
            "你是一名个性化满意度边界评估员。\n"
            "本题只判断当前助手回复是否达到该用户的【满意最低线】。\n"
            "请不要做 1/2/5 分细分，只输出：\n"
            "- `4` = 满意（达到最低满意线）\n"
            "- `3` = 不满意（未达到最低满意线）\n\n"
            f"【用户评分摘要】\n"
            f"评分风格：{memory.scoring_style}\n"
            f"历史平均分：{memory.avg_satisfaction_score:.2f}\n"
            f"满意最低线（3→4 边界）：{memory.three_vs_four_distinction}\n"
            f"补充说明（4→5 的更高要求，仅供参考，不作为本题判定目标）：{memory.four_vs_five_distinction}\n"
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
            + "Step 1. 先判断回复是否直接回答了用户问题，并满足关键约束。\n"
            + "Step 2. 再判断它是否达到该用户的【满意最低线】。\n"
            + "  - 若达到最低满意线，输出 `classification=4`\n"
            + "  - 若未达到最低满意线，输出 `classification=3`\n"
            + "Step 3. 选择一个最贴切的原因标签。\n\n"
            + "注意：\n"
            + "- 本题的目标是判定【满意 / 不满意】，不是区分 4 和 5。\n"
            + "- 除非回复明显没达到最低要求，否则不要因为“还不够优秀”就判成 3。\n"
            + "- `classification` 只能输出 `3` 或 `4`。\n\n"
            + "请严格输出 JSON，不要输出其他内容：\n"
            + "{\n"
            + '  "classification": 只能是 3 或 4,\n'
            + f'  "reason": "{reason_json_rule}",\n'
            + '  "analysis": "用 2-4 句写明：是否直接回答问题；是否达到最低满意线；最终为何判为 3 或 4。若使用参考案例，注明更接近满意案例还是不满意案例" \n'
            + "}\n"
        )
        return prompt

