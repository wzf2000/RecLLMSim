"""Prompt builders for URS episodic-retrieval satisfaction prediction."""

from __future__ import annotations

from .memory import _format_profile, _format_reason_json_rule, _format_reason_rule_block
from .satisfaction_constants import get_reason_to_id
from .urs_episodic import UrsEpisodicMemoryRecord, format_urs_episodic_memories
from .urs_memory import (
    _detect_session_language,
    _format_session_dialogue,
    _format_task_guard_block,
    _format_urs_calibration_block,
)

URS_EPISODIC_PROMPT_VERSIONS = {
    "urs_episodic_task_guarded",
    "urs_episodic_task_guarded_dsat_first",
    "urs_episodic_task_guarded_dsat_twostage",
}


def _split_evidence_blocks(
    retrieved_memories: list[UrsEpisodicMemoryRecord],
) -> tuple[str, str, str]:
    dsat = [m for m in retrieved_memories if m.score <= 3]
    sat = [m for m in retrieved_memories if m.score >= 4]
    other = [m for m in retrieved_memories if 3 < m.score < 4]
    return (
        format_urs_episodic_memories(dsat) if dsat else "（没有检索到 DSAT 历史失败案例。）",
        format_urs_episodic_memories(sat) if sat else "（没有检索到 SAT 历史满意案例。）",
        format_urs_episodic_memories(other) if other else "",
    )


def _common_context(
    task_context: str,
    session_history: list[dict],
) -> tuple[str, str, str, str]:
    dialogue_text = _format_session_dialogue(session_history)
    language = _detect_session_language(task_context, session_history)
    calibration_block = _format_urs_calibration_block(
        "urs_v2_calibrated_task_guarded",
        language=language,
    )
    task_guard_block = _format_task_guard_block(
        task_context,
        "urs_v2_calibrated_task_guarded",
    )
    return dialogue_text, language, calibration_block, task_guard_block


def build_urs_episodic_task_guarded_prompt(
    profile: dict,
    task_context: str,
    session_history: list[dict],
    retrieved_memories: list[UrsEpisodicMemoryRecord],
    prompt_version: str = "urs_episodic_task_guarded",
) -> str:
    if prompt_version not in URS_EPISODIC_PROMPT_VERSIONS:
        raise ValueError(f"Unsupported URS episodic prompt version: {prompt_version}")
    if prompt_version == "urs_episodic_task_guarded_dsat_twostage":
        raise ValueError("Two-stage prompt must use build_urs_dsat_failure_check_prompt first")

    reason_labels = list(get_reason_to_id().keys())
    reason_text = "、".join(reason_labels)
    reason_rule_block = _format_reason_rule_block()
    reason_json_rule = _format_reason_json_rule()
    dialogue_text, _, calibration_block, task_guard_block = _common_context(
        task_context,
        session_history,
    )

    if prompt_version == "urs_episodic_task_guarded_dsat_first":
        dsat_block, sat_block, other_block = _split_evidence_blocks(retrieved_memories)
        other_section = f"\n\n【Other retrieved evidence】\n{other_block}" if other_block else ""
        return (
            "你是一名个性化满意度评估员。\n"
            "请对下面【整段对话】给出一个 1-5 的整数分数，反映用户对本次对话的整体满意度。\n"
            "注意：这是 URS session-level 评分；每段对话只输出 1 个分数。\n\n"
            "【Memory 类型说明】\n"
            "- 这里不使用总结式 user memory，而是使用该用户其他 intent 下的原始历史对话作为 episodic retrieval memory。\n"
            "- 每条 episodic memory 都带有真实 session-level 满意度标签，可作为当前 3/4 满意边界的参照案例。\n"
            "- episodic memory 是证据，不是硬规则；当前 session 的直接质量证据和 task-aware guard 优先。\n"
            "- 本版本采用 DSAT-first failure check：在给 4/5 之前，必须先检查当前 session 是否与历史失败案例属于同类失败。\n\n"
            f"【用户画像】{_format_profile(profile)}\n\n"
            f"【Closest DSAT failure evidence（score <= 3）】\n{dsat_block}\n\n"
            f"【Closest SAT success evidence（score >= 4）】\n{sat_block}"
            f"{other_section}\n\n"
            f"【当前任务背景】{task_context}\n\n"
            f"【当前完整对话】\n{dialogue_text}\n\n"
            f"{calibration_block}"
            f"{task_guard_block}"
            "【DSAT-first episodic evidence 使用步骤】\n"
            "Step D1: 先只看当前 session 和 task-aware guard，判断核心请求是否被正确且有用地满足，得到 provisional score。\n"
            "Step D2: 检查 closest DSAT failure evidence：当前 session 是否出现同类失败？同类失败包括核心请求未回答、缺少用户明确要求的 artifact/source/product/route/format、泛泛建议、事实不可验证、格式明显不符、需要用户继续追问才能完成主要任务。\n"
            "Step D3: 若当前 session 与任一 DSAT evidence 的失败类型具体匹配，且该失败影响核心任务，则 classification 不能为 4/5。\n"
            "Step D4: 若要输出 classification >= 4，必须明确说明为什么当前 session 不属于 closest DSAT failure evidence 的同类失败，并引用当前 session 的具体满足证据。\n"
            "Step D5: 再检查 closest SAT success evidence：只有当前 session 在核心完成度上接近 SAT evidence 时，才支持给 4/5。\n"
            "Step D6: 不要因为历史中存在 DSAT evidence 就机械降分；必须是当前 intent/request/defect 具体相似才可影响 3/4 边界。\n"
            "Step D7: 若最终达到 4 分，再判断是否明显超过基本要求而给 5；5 分不要求完美。\n"
            "Step D8: 选取一个最贴切的原因标签（若 classification >= 4 必须为【满意】）。\n\n"
            f"【可选原因标签】{reason_text}\n{reason_rule_block}\n"
            "请严格输出 JSON，不要输出其他内容：\n"
            "{\n"
            '  "classification": 1-5 中的整数（对整段对话的单一分数）,\n'
            f'  "reason": "{reason_json_rule}",\n'
            '  "analysis": "按 StepD1-StepD8 格式说明判断过程；若 classification >= 4，必须包含 why_not_dsat_failure；若 classification <= 3，必须说明匹配的 failure evidence 或当前 session 直接缺陷"\n'
            "}\n"
        )

    episodic_block = format_urs_episodic_memories(retrieved_memories)
    return (
        "你是一名个性化满意度评估员。\n"
        "请对下面【整段对话】给出一个 1-5 的整数分数，反映用户对本次对话的整体满意度。\n"
        "注意：这是 URS session-level 评分；每段对话只输出 1 个分数。\n\n"
        "【Memory 类型说明】\n"
        "- 这里不使用总结式 user memory，而是使用该用户其他 intent 下的原始历史对话作为 episodic retrieval memory。\n"
        "- 每条 episodic memory 都带有真实 session-level 满意度标签，可作为当前 3/4 满意边界的参照案例。\n"
        "- episodic memory 是证据，不是硬规则；当前 session 的直接质量证据和 task-aware guard 优先。\n"
        "- 不要机械复制历史案例分数；只在历史案例与当前请求、缺陷类型或完成程度具体相似时才影响判断。\n\n"
        f"【用户画像】{_format_profile(profile)}\n\n"
        f"【Retrieved episodic memory evidence】\n{episodic_block}\n\n"
        f"【当前任务背景】{task_context}\n\n"
        f"【当前完整对话】\n{dialogue_text}\n\n"
        f"{calibration_block}"
        f"{task_guard_block}"
        "【Episodic evidence 使用步骤】\n"
        "Step M1: 先基于当前 session 证据和 task-aware guard 判断核心请求是否被满足，得到 provisional SAT/DSAT 判断。\n"
        "Step M2: 比较 retrieved evidence 中最相近的 SAT(>=4) 与 DSAT(<=3) 案例，说明当前 session 更接近哪一侧。\n"
        "Step M3: 只有当 episodic evidence 与当前 intent/请求/缺陷高度相关时，才允许改变 3/4 边界；否则保持 provisional 判断。\n"
        "Step M4: 若最终达到 4 分，再判断是否明显超过基本要求而给 5；5 分不要求完美。\n"
        "Step M5: 若最终低于 4 分，根据缺陷严重程度决定 1/2/3 分。\n"
        "Step M6: 选取一个最贴切的原因标签（若 classification >= 4 必须为【满意】）。\n\n"
        f"【可选原因标签】{reason_text}\n{reason_rule_block}\n"
        "请严格输出 JSON，不要输出其他内容：\n"
        "{\n"
        '  "classification": 1-5 中的整数（对整段对话的单一分数）,\n'
        f'  "reason": "{reason_json_rule}",\n'
        '  "analysis": "按 StepM1-StepM6 格式说明判断过程，必须引用当前 session 证据；若 episodic memory 影响判断，写出关键 evidence id 和相似点；若未采用 memory，也说明原因"\n'
        "}\n"
    )


def build_urs_dsat_failure_check_prompt(
    profile: dict,
    task_context: str,
    session_history: list[dict],
    retrieved_memories: list[UrsEpisodicMemoryRecord],
) -> str:
    dsat_block, sat_block, _ = _split_evidence_blocks(retrieved_memories)
    dialogue_text, _, _, task_guard_block = _common_context(task_context, session_history)
    return (
        "你是一名 URS satisfaction evaluator 的 failure-evidence arbiter。\n"
        "你的任务不是打 1-5 分，而是只判断：当前 session 是否与 retrieved DSAT 历史失败案例属于同类失败。\n\n"
        "【判断对象】\n"
        "- same_failure_as_dsat_evidence=true：当前 session 与某个 DSAT evidence 在核心失败类型上具体相同或高度相似，并且该失败影响主要任务完成。\n"
        "- same_failure_as_dsat_evidence=false：当前 session 没有同类核心失败，或相似点只是表面主题/表达风格/泛化偏好。\n\n"
        "【同类失败类型定义】\n"
        "- 核心请求没有被回答，或回答对象/任务错位。\n"
        "- 用户要求具体 artifact/source/product/route/price/format/code/关键元素，但回答缺失关键对象或约束。\n"
        "- 回答过于泛泛、只有背景或建议，用户仍必须继续追问才能完成主要任务。\n"
        "- 关键事实不可验证、明显错误、幻觉，或不当拒答。\n"
        "- 文本/创意任务漏掉明确格式、语气、主题、角色或关键情节。\n\n"
        "【非同类失败】\n"
        "- 只是同一个 intent，但失败点不同。\n"
        "- 只是都比较简短、都结构化、都不够详细。\n"
        "- DSAT evidence 来自无关 intent，且不能映射到当前请求的核心缺陷。\n"
        "- 当前 session 已经直接完成核心请求，只有轻微可改进点。\n\n"
        f"【用户画像】{_format_profile(profile)}\n\n"
        f"【Closest DSAT failure evidence（score <= 3）】\n{dsat_block}\n\n"
        f"【Closest SAT success evidence（score >= 4，仅作反例参考）】\n{sat_block}\n\n"
        f"【当前任务背景】{task_context}\n\n"
        f"【当前完整对话】\n{dialogue_text}\n\n"
        f"{task_guard_block}"
        "请严格输出 JSON，不要输出其他内容：\n"
        "{\n"
        '  "same_failure_as_dsat_evidence": true 或 false,\n'
        '  "matched_evidence_ids": ["匹配的 DSAT evidence id；没有则为空数组"],\n'
        '  "failure_type": "core_missing / missing_constraint / generic_unusable / unverifiable_or_wrong / format_mismatch / off_task / none / other",\n'
        '  "confidence": "low / medium / high",\n'
        '  "analysis": "先引用当前 session 的具体缺陷或满足证据，再逐条说明是否与 closest DSAT evidence 同类；不要输出 1-5 分"\n'
        "}\n"
    )


def build_urs_episodic_twostage_score_prompt(
    profile: dict,
    task_context: str,
    session_history: list[dict],
    retrieved_memories: list[UrsEpisodicMemoryRecord],
    failure_check: dict,
) -> str:
    reason_labels = list(get_reason_to_id().keys())
    reason_text = "、".join(reason_labels)
    reason_rule_block = _format_reason_rule_block()
    reason_json_rule = _format_reason_json_rule()
    dialogue_text, _, calibration_block, task_guard_block = _common_context(
        task_context,
        session_history,
    )
    dsat_block, sat_block, _ = _split_evidence_blocks(retrieved_memories)
    failure_check_json = {
        "same_failure_as_dsat_evidence": failure_check.get("same_failure_as_dsat_evidence"),
        "matched_evidence_ids": failure_check.get("matched_evidence_ids", []),
        "failure_type": failure_check.get("failure_type", "none"),
        "confidence": failure_check.get("confidence", "low"),
        "analysis": failure_check.get("analysis", ""),
    }
    return (
        "你是一名个性化满意度评估员。\n"
        "请对下面【整段对话】给出一个 1-5 的整数分数，反映用户对本次对话的整体满意度。\n"
        "注意：这是两阶段 URS episodic evaluator 的第二阶段。第一阶段已经独立判断当前 session 是否与历史 DSAT 失败案例同类。\n\n"
        f"【用户画像】{_format_profile(profile)}\n\n"
        f"【Stage-1 DSAT failure check（必须作为硬约束参考）】\n{failure_check_json}\n\n"
        "【Stage-1 使用规则】\n"
        "- 若 same_failure_as_dsat_evidence=true 且 confidence 为 medium/high，说明当前 session 存在与历史 DSAT 同类的核心失败；此时 classification 必须为 1/2/3，不能给 4/5。\n"
        "- 若 same_failure_as_dsat_evidence=true 但 confidence=low，只能作为弱风险信号，仍需依据当前 session 直接证据判断。\n"
        "- 若 same_failure_as_dsat_evidence=false，不代表自动满意；仍需按照 task-aware guard 和当前 session 证据评分。\n\n"
        f"【Closest DSAT failure evidence】\n{dsat_block}\n\n"
        f"【Closest SAT success evidence】\n{sat_block}\n\n"
        f"【当前任务背景】{task_context}\n\n"
        f"【当前完整对话】\n{dialogue_text}\n\n"
        f"{calibration_block}"
        f"{task_guard_block}"
        "【评分步骤】\n"
        "Step T1: 复述 Stage-1 的 failure decision，并说明它是否触发 1/2/3 硬门控。\n"
        "Step T2: 基于当前 session 证据判断核心请求是否被满足；若触发硬门控，不得输出 4/5。\n"
        "Step T3: 若未触发硬门控，再用 task-aware guard 判断 3/4 边界。\n"
        "Step T4: 若达到 4 分，再判断是否明显超过基本要求而给 5；5 分不要求完美。\n"
        "Step T5: 选取原因标签（classification >= 4 必须为【满意】；classification <= 3 必须选择不满意原因）。\n\n"
        f"【可选原因标签】{reason_text}\n{reason_rule_block}\n"
        "请严格输出 JSON，不要输出其他内容：\n"
        "{\n"
        '  "classification": 1-5 中的整数（对整段对话的单一分数）,\n'
        f'  "reason": "{reason_json_rule}",\n'
        '  "analysis": "按 StepT1-StepT5 格式说明判断过程，必须明确 Stage-1 是否触发硬门控"\n'
        "}\n"
    )
