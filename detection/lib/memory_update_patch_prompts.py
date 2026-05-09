"""User-memory update prompt builders and update evidence helpers."""

from __future__ import annotations

from __future__ import annotations

import json
from collections import defaultdict

from .personalized_data import SessionData
from .memory_schema import UserMemory
from .memory_formatting import _truncate
from .memory_update_common import (
    _build_update_evidence_bundle,
    _collect_update_turns,
    _format_update_examples,
)


def build_memory_update_prompt_v2_1(
    existing_memory: UserMemory,
    new_session: SessionData,
    turn_predictions: list[dict],
    use_oracle_labels: bool = False,
) -> str:
    """
    构造 memory update prompt（v2.1）。

    核心改动：
      1. 不再要求整份 memory 重写
      2. 统计量由代码侧更新
      3. verbal 字段只允许 patch 式修改
    """
    existing_json = existing_memory.model_dump_json(
        indent=2,
        exclude={"memory_version", "source_tasks", "n_history_sessions", "n_history_turns"},
    )
    turns = _collect_update_turns(new_session, turn_predictions, use_oracle_labels)
    by_score: dict[int, list[dict]] = defaultdict(list)
    for t in turns:
        by_score[t["score"]].append(t)

    low_turns = by_score[1] + by_score[2] + by_score[3]
    mid_turns = by_score[4]
    high_turns = by_score[5]

    session_stats = (
        f"本 session 分布："
        f"5分×{len(by_score[5])} / 4分×{len(by_score[4])} / 3分×{len(by_score[3])} / "
        f"2分×{len(by_score[2])} / 1分×{len(by_score[1])}"
    )
    label_note = (
        "本次提供了真实标签，可视为可靠证据。"
        if use_oracle_labels
        else "本次仅有模型预测标签，属于弱证据；除非形成清晰模式，否则不要改 verbal 边界字段。"
    )
    prompt = (
        "你正在维护一份 v2.1 用户记忆。请根据新 session 给出【字段级 patch】，而不是重写整份 memory。\n\n"
        f"【现有记忆】\n{existing_json}\n\n"
        f"【新 Session 概览】\n"
        f"任务：{new_session.task}\n"
        f"任务背景：{_truncate(new_session.task_context, 180)}\n"
        f"{session_stats}\n"
        f"【说明】{label_note}\n\n"
        f"{_format_update_examples('【<=3 证据（可能影响 3/4 边界或低分要求）】', low_turns)}\n\n"
        f"{_format_update_examples('【4 分证据（与 <=3 或 5 比较时使用）】', mid_turns)}\n\n"
        f"{_format_update_examples('【5 分证据（可能影响 4/5 边界）】', high_turns)}\n\n"
        "更新原则：\n"
        "1. avg_satisfaction_score 和 score_distribution 由程序自动更新，你不需要负责统计数字。\n"
        "2. 只有在新 session 提供了明确相邻分数证据时，才改写边界字段：\n"
        "   - 改写 three_vs_four_distinction 需要有清晰的 <=3 与 4 分对比证据\n"
        "   - 改写 four_vs_five_distinction 需要有清晰的 4 与 5 分对比证据\n"
        "3. 如果只是重复了现有模式，必须保持 verbal 字段不变。\n"
        "4. user_specific_requirements 只能新增真正会改变评分的个性化要求；禁止加入泛化要求。\n"
        "5. preferred_response_format 只有在出现新的稳定格式偏好时才改。\n"
        "6. task_specific_observations 只新增/覆盖当前 session 提供了明确新信息的任务观察。\n\n"
        "请严格按下面的 JSON Schema 输出 patch，不要输出其他内容：\n"
        "{\n"
        '  "update_scoring_style": true 或 false,\n'
        '  "scoring_style": "若不更新则原样复述现有值",\n'
        '  "update_three_vs_four": true 或 false,\n'
        '  "three_vs_four_distinction": "若不更新则原样复述现有值",\n'
        '  "update_four_vs_five": true 或 false,\n'
        '  "four_vs_five_distinction": "若不更新则原样复述现有值",\n'
        '  "add_user_specific_requirements": ["仅新增条目；若无则空列表"],\n'
        '  "update_preferred_response_format": true 或 false,\n'
        '  "preferred_response_format": "若不更新则原样复述现有值",\n'
        '  "add_or_update_task_specific_observations": [{"task_name": "...", "observation": "..."}],\n'
        '  "rationale": "1-3句说明主要依据与保持不变的原因"\n'
        "}\n"
    )
    return prompt


def build_memory_update_prompt_v2_4(
    existing_memory: UserMemory,
    new_session: SessionData,
    turn_predictions: list[dict],
    use_oracle_labels: bool = False,
) -> str:
    """
    构造 memory update prompt（v2.4）。

    设计目标：
      - 以 v2.1 的原始 turn 级证据和 patch schema 为基础
      - 不引入 v2.2 的 evidence bundle
      - 不引入 v2.3 的额外边界摘要，避免过度拉向 DSAT
      - 仅强调无相邻证据时不改 verbal boundary
    """
    existing_json = existing_memory.model_dump_json(
        indent=2,
        exclude={"memory_version", "source_tasks", "n_history_sessions", "n_history_turns"},
    )
    turns = _collect_update_turns(new_session, turn_predictions, use_oracle_labels)
    by_score: dict[int, list[dict]] = defaultdict(list)
    for t in turns:
        by_score[t["score"]].append(t)

    low_turns = by_score[1] + by_score[2] + by_score[3]
    mid_turns = by_score[4]
    high_turns = by_score[5]

    session_stats = (
        f"本 session 分布："
        f"5分×{len(by_score[5])} / 4分×{len(by_score[4])} / 3分×{len(by_score[3])} / "
        f"2分×{len(by_score[2])} / 1分×{len(by_score[1])}"
    )
    label_note = (
        "本次提供了真实标签，可视为可靠证据。"
        if use_oracle_labels
        else "本次仅有模型预测标签，属于弱证据；请保持 v2.1 的校准收益，但不要把单边样本升级成新的硬边界规则。"
    )
    prompt = (
        "你正在维护一份 v2.4 用户记忆。请根据新 session 给出【字段级 patch】，而不是重写整份 memory。\n\n"
        f"【现有记忆】\n{existing_json}\n\n"
        f"【新 Session 概览】\n"
        f"任务：{new_session.task}\n"
        f"任务背景：{_truncate(new_session.task_context, 180)}\n"
        f"{session_stats}\n"
        f"【说明】{label_note}\n\n"
        f"{_format_update_examples('【<=3 证据（可能影响 3/4 边界或低分要求）】', low_turns)}\n\n"
        f"{_format_update_examples('【4 分证据（与 <=3 或 5 比较时使用）】', mid_turns)}\n\n"
        f"{_format_update_examples('【5 分证据（可能影响 4/5 边界）】', high_turns)}\n\n"
        "更新原则：\n"
        "1. avg_satisfaction_score 和 score_distribution 由程序自动更新，你不需要负责统计数字。\n"
        "2. 尽量保持 v2.1 的轻量 patch 风格：只在有新信息时更新，不做大幅重写。\n"
        "3. three_vs_four_distinction 只有在本 session 同时存在 3 分和 4 分证据时才建议改写；"
        "若只有 <=3 或只有 4 分，请保持原样。\n"
        "4. four_vs_five_distinction 只有在本 session 同时存在 4 分和 5 分证据时才建议改写；"
        "若只有单边证据，请保持原样。\n"
        "5. 不要为了提高不满意识别而系统性压低分数；边界文字只能描述证据中真实出现的差异。\n"
        "6. user_specific_requirements 只能新增真正会改变评分的个性化要求；"
        "禁止加入“更详细、更具体、更清晰、更结构化、更实用”等泛化要求。\n"
        "7. preferred_response_format 只有在出现新的稳定格式偏好时才改。\n"
        "8. task_specific_observations 只新增/覆盖当前 session 提供了明确新信息的任务观察。\n\n"
        "请严格按下面的 JSON Schema 输出 patch，不要输出其他内容：\n"
        "{\n"
        '  "update_scoring_style": true 或 false,\n'
        '  "scoring_style": "若不更新则原样复述现有值",\n'
        '  "update_three_vs_four": true 或 false,\n'
        '  "three_vs_four_distinction": "若不更新则原样复述现有值",\n'
        '  "update_four_vs_five": true 或 false,\n'
        '  "four_vs_five_distinction": "若不更新则原样复述现有值",\n'
        '  "add_user_specific_requirements": ["仅新增条目；若无则空列表"],\n'
        '  "update_preferred_response_format": true 或 false,\n'
        '  "preferred_response_format": "若不更新则原样复述现有值",\n'
        '  "add_or_update_task_specific_observations": [{"task_name": "...", "observation": "..."}],\n'
        '  "rationale": "1-3句说明主要依据与保持不变的原因"\n'
        "}\n"
    )
    return prompt


def build_memory_update_prompt_v2_5(
    existing_memory: UserMemory,
    new_session: SessionData,
    turn_predictions: list[dict],
    use_oracle_labels: bool = False,
) -> str:
    """
    构造 memory update prompt（v2.5）。

    设计目标：
      - 继续使用 v2.1 的原始 turn 级证据与 patch schema
      - 不把 SAT drift 问题交给 verbal boundary 文本解决
      - 明确要求非 oracle update 不改 scoring_style
      - 统计均值的上移由 merge 侧轻量阻尼
    """
    existing_json = existing_memory.model_dump_json(
        indent=2,
        exclude={"memory_version", "source_tasks", "n_history_sessions", "n_history_turns"},
    )
    turns = _collect_update_turns(new_session, turn_predictions, use_oracle_labels)
    by_score: dict[int, list[dict]] = defaultdict(list)
    for t in turns:
        by_score[t["score"]].append(t)

    low_turns = by_score[1] + by_score[2] + by_score[3]
    mid_turns = by_score[4]
    high_turns = by_score[5]

    session_stats = (
        f"本 session 分布："
        f"5分×{len(by_score[5])} / 4分×{len(by_score[4])} / 3分×{len(by_score[3])} / "
        f"2分×{len(by_score[2])} / 1分×{len(by_score[1])}"
    )
    label_note = (
        "本次提供了真实标签，可视为可靠证据。"
        if use_oracle_labels
        else "本次仅有模型预测标签，属于弱证据；请不要根据预测标签改写 scoring_style 或提高用户整体宽松程度。"
    )
    prompt = (
        "你正在维护一份 v2.5 用户记忆。请根据新 session 给出【字段级 patch】，而不是重写整份 memory。\n\n"
        f"【现有记忆】\n{existing_json}\n\n"
        f"【新 Session 概览】\n"
        f"任务：{new_session.task}\n"
        f"任务背景：{_truncate(new_session.task_context, 180)}\n"
        f"{session_stats}\n"
        f"【说明】{label_note}\n\n"
        f"{_format_update_examples('【<=3 证据（可能影响 3/4 边界或低分要求）】', low_turns)}\n\n"
        f"{_format_update_examples('【4 分证据（与 <=3 或 5 比较时使用）】', mid_turns)}\n\n"
        f"{_format_update_examples('【5 分证据（可能影响 4/5 边界）】', high_turns)}\n\n"
        "更新原则：\n"
        "1. avg_satisfaction_score 和 score_distribution 由程序自动更新，你不需要负责统计数字。\n"
        "2. 非 oracle 场景下，scoring_style 默认保持原样；除非说明中明确写着真实标签可靠，否则 update_scoring_style 必须为 false。\n"
        "3. 不要因为本 session 预测分数偏高，就把用户描述成更宽松、更容易满意。\n"
        "4. three_vs_four_distinction 只有在出现清晰的 <=3 与 4 分对比证据时才改写。\n"
        "5. four_vs_five_distinction 只有在出现清晰的 4 与 5 分对比证据时才改写。\n"
        "6. user_specific_requirements 只能新增真正会改变评分的个性化要求；禁止加入泛化要求。\n"
        "7. preferred_response_format 只有在出现新的稳定格式偏好时才改。\n"
        "8. task_specific_observations 只新增/覆盖当前 session 提供了明确新信息的任务观察。\n\n"
        "请严格按下面的 JSON Schema 输出 patch，不要输出其他内容：\n"
        "{\n"
        '  "update_scoring_style": true 或 false,\n'
        '  "scoring_style": "若不更新则原样复述现有值",\n'
        '  "update_three_vs_four": true 或 false,\n'
        '  "three_vs_four_distinction": "若不更新则原样复述现有值",\n'
        '  "update_four_vs_five": true 或 false,\n'
        '  "four_vs_five_distinction": "若不更新则原样复述现有值",\n'
        '  "add_user_specific_requirements": ["仅新增条目；若无则空列表"],\n'
        '  "update_preferred_response_format": true 或 false,\n'
        '  "preferred_response_format": "若不更新则原样复述现有值",\n'
        '  "add_or_update_task_specific_observations": [{"task_name": "...", "observation": "..."}],\n'
        '  "rationale": "1-3句说明主要依据与保持不变的原因"\n'
        "}\n"
    )
    return prompt


def build_memory_update_prompt_v2_2(
    existing_memory: UserMemory,
    new_session: SessionData,
    turn_predictions: list[dict],
    use_oracle_labels: bool = False,
) -> str:
    """
    构造 memory update prompt（v2.2）。

    核心改动：
      1. 输入改成结构化 evidence bundle
      2. 要求模型输出 patch 时同时给出 confidence / support_count
      3. 让 non-oracle update 看到不确定性信号，避免把 noisy session 直接升级成硬规则
    """
    existing_json = existing_memory.model_dump_json(
        indent=2,
        exclude={"memory_version", "source_tasks", "n_history_sessions", "n_history_turns"},
    )
    evidence_bundle = _build_update_evidence_bundle(
        existing_memory=existing_memory,
        new_session=new_session,
        turn_predictions=turn_predictions,
        use_oracle_labels=use_oracle_labels,
    )
    evidence_json = json.dumps(evidence_bundle, ensure_ascii=False, indent=2)
    label_note = (
        "本次提供了真实标签，可将 evidence 视作强证据。"
        if use_oracle_labels
        else "本次 only 有模型预测标签。除非 evidence bundle 显示出稳定、清晰的边界模式，否则不要更新 verbal 边界字段。"
    )
    prompt = (
        "你正在维护一份 v2.2 用户记忆。请根据【结构化 evidence bundle】输出字段级 patch。\n\n"
        f"【现有记忆】\n{existing_json}\n\n"
        f"【Evidence Bundle】\n{evidence_json}\n\n"
        f"【说明】{label_note}\n\n"
        "更新原则：\n"
        "1. avg_satisfaction_score 和 score_distribution 由程序自动更新，你不需要负责统计数字。\n"
        "2. three_vs_four_distinction 只有在 evidence bundle 同时出现清晰的 <=3 与 4 分证据时才允许更新。\n"
        "3. four_vs_five_distinction 只有在 evidence bundle 同时出现清晰的 4 与 5 分证据时才允许更新。\n"
        "4. 若 uncertainty_signals 表明本 session 边界不稳定（例如大量 borderline mentions、频繁 3/4 或 4/5 flip），应降低 confidence，而不是勉强改写规则。\n"
        "5. user_specific_requirements 只新增真正会改变评分的个性化要求；像“更详细、更具体、更清晰”这类泛化词不要加入。\n"
        "6. preferred_response_format 只有在 evidence 明确显示稳定格式偏好时才更新；单个 session 默认不改。\n"
        "7. task_specific_observations 只对当前 session 明确提供新信息的任务给出 patch。\n\n"
        "请严格输出 v2.2 patch JSON，不要输出其他内容。字段要求：\n"
        "{\n"
        '  "update_scoring_style": true/false,\n'
        '  "scoring_style": "...",\n'
        '  "scoring_style_confidence": "low|medium|high",\n'
        '  "update_three_vs_four": true/false,\n'
        '  "three_vs_four_distinction": "...",\n'
        '  "three_vs_four_confidence": "low|medium|high",\n'
        '  "three_vs_four_evidence_count": 0-10,\n'
        '  "update_four_vs_five": true/false,\n'
        '  "four_vs_five_distinction": "...",\n'
        '  "four_vs_five_confidence": "low|medium|high",\n'
        '  "four_vs_five_evidence_count": 0-10,\n'
        '  "add_user_specific_requirements": [{"requirement":"...","confidence":"low|medium|high","support_count":0-10}],\n'
        '  "update_preferred_response_format": true/false,\n'
        '  "preferred_response_format": "...",\n'
        '  "preferred_response_format_confidence": "low|medium|high",\n'
        '  "preferred_response_format_support_count": 0-10,\n'
        '  "add_or_update_task_specific_observations": [{"task_name":"...","observation":"...","confidence":"low|medium|high","support_count":0-10}],\n'
        '  "rationale": "1-4 句说明哪些字段有足够 evidence，哪些字段应保持不变"\n'
        "}\n"
    )
    return prompt


def build_memory_update_prompt_v2_3(
    existing_memory: UserMemory,
    new_session: SessionData,
    turn_predictions: list[dict],
    use_oracle_labels: bool = False,
) -> str:
    """
    构造 memory update prompt（v2.3）。

    设计目标：
      - 回到 v2.1 的原始 turn 级证据输入
      - 只额外补充轻量边界摘要
      - 不再像 v2.2 那样把证据重压缩成抽象 bundle
    """
    existing_json = existing_memory.model_dump_json(
        indent=2,
        exclude={"memory_version", "source_tasks", "n_history_sessions", "n_history_turns"},
    )
    turns = _collect_update_turns(new_session, turn_predictions, use_oracle_labels)
    by_score: dict[int, list[dict]] = defaultdict(list)
    for t in turns:
        by_score[t["score"]].append(t)

    low_turns = by_score[1] + by_score[2] + by_score[3]
    mid_turns = by_score[4]
    high_turns = by_score[5]
    has_3_and_4 = bool(by_score[3] and by_score[4])
    has_4_and_5 = bool(by_score[4] and by_score[5])

    session_stats = (
        f"本 session 分布："
        f"5分×{len(by_score[5])} / 4分×{len(by_score[4])} / 3分×{len(by_score[3])} / "
        f"2分×{len(by_score[2])} / 1分×{len(by_score[1])}"
    )
    label_note = (
        "本次提供了真实标签，可视为可靠证据。"
        if use_oracle_labels
        else "本次仅有模型预测标签。请优先相信那些形成清晰 3/4 或 4/5 对比的 turn；"
             "若只是单边高分或单边低分，不要轻易改写长期边界规则。"
    )
    boundary_summary = (
        "【边界样本摘要】\n"
        f"- 是否同时出现 3 分和 4 分：{'是' if has_3_and_4 else '否'}\n"
        f"- 是否同时出现 4 分和 5 分：{'是' if has_4_and_5 else '否'}\n"
        f"- <=3 样本数：{len(low_turns)}\n"
        f"- 4 分样本数：{len(mid_turns)}\n"
        f"- 5 分样本数：{len(high_turns)}\n"
        "- 若没有相邻分数证据，只能微调 calibration 或补充 requirement，不能把边界文字改成新硬规则。"
    )

    prompt = (
        "你正在维护一份 v2.3 用户记忆。请根据新 session 给出【字段级 patch】，不要重写整份 memory。\n\n"
        f"【现有记忆】\n{existing_json}\n\n"
        f"【新 Session 概览】\n"
        f"任务：{new_session.task}\n"
        f"任务背景：{_truncate(new_session.task_context, 180)}\n"
        f"{session_stats}\n"
        f"【说明】{label_note}\n\n"
        f"{boundary_summary}\n\n"
        f"{_format_update_examples('【<=3 证据（可能影响 3/4 边界或低分要求）】', low_turns)}\n\n"
        f"{_format_update_examples('【4 分证据（与 <=3 或 5 比较时使用）】', mid_turns)}\n\n"
        f"{_format_update_examples('【5 分证据（可能影响 4/5 边界）】', high_turns)}\n\n"
        "更新原则：\n"
        "1. avg_satisfaction_score 和 score_distribution 由程序自动更新，你不需要负责统计数字。\n"
        "2. three_vs_four_distinction 只有在新 session 同时出现清晰的 3 分与 4 分证据时才建议改写；否则保持原样。\n"
        "3. four_vs_five_distinction 只有在新 session 同时出现清晰的 4 分与 5 分证据时才建议改写；否则保持原样。\n"
        "4. 若只是重复现有模式，必须保持 verbal 字段不变。\n"
        "5. user_specific_requirements 只新增真正会改变评分的个性化要求；不要加入“更详细、更具体、更清晰”这类泛化要求。\n"
        "6. preferred_response_format 只有在出现新的稳定格式偏好时才改。\n"
        "7. task_specific_observations 只新增/覆盖当前 session 提供了明确新信息的任务观察。\n\n"
        "请严格按下面的 JSON Schema 输出 patch，不要输出其他内容：\n"
        "{\n"
        '  "update_scoring_style": true 或 false,\n'
        '  "scoring_style": "若不更新则原样复述现有值",\n'
        '  "update_three_vs_four": true 或 false,\n'
        '  "three_vs_four_distinction": "若不更新则原样复述现有值",\n'
        '  "update_four_vs_five": true 或 false,\n'
        '  "four_vs_five_distinction": "若不更新则原样复述现有值",\n'
        '  "add_user_specific_requirements": ["仅新增条目；若无则空列表"],\n'
        '  "update_preferred_response_format": true 或 false,\n'
        '  "preferred_response_format": "若不更新则原样复述现有值",\n'
        '  "add_or_update_task_specific_observations": [{"task_name": "...", "observation": "..."}],\n'
        '  "rationale": "1-3句说明主要依据与保持不变的原因"\n'
        "}\n"
    )
    return prompt

