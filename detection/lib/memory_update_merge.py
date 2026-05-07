"""Merge structured memory update patches back into user memory."""

from __future__ import annotations

from .memory_schema import (
    MemoryUpdatePatchV2_1,
    MemoryUpdatePatchV2_2,
    ScoreDistribution,
    TaskObservation,
    UserMemory,
    _score_distribution_to_counts,
)
from .memory_update_prompts import (
    _dedupe_preserve_order,
    _is_generic_requirement,
)

def merge_memory_v2_1_patch(
    existing_memory: UserMemory,
    patch: MemoryUpdatePatchV2_1,
    turn_predictions: list[dict],
    use_oracle_labels: bool = False,
) -> UserMemory:
    """将 v2.1 patch 合并回 v2 memory，并由代码端更新统计量。"""
    scores = [
        int(pred["gold_score"] if use_oracle_labels else pred["pred_score"])
        for pred in turn_predictions
    ]
    old_counts = existing_memory.score_distribution.model_copy()
    counts_map = {
        1: old_counts.score_1,
        2: old_counts.score_2,
        3: old_counts.score_3,
        4: old_counts.score_4,
        5: old_counts.score_5,
    }
    for s in scores:
        counts_map[s] += 1
    new_dist = ScoreDistribution(
        score_1=counts_map[1],
        score_2=counts_map[2],
        score_3=counts_map[3],
        score_4=counts_map[4],
        score_5=counts_map[5],
    )
    old_total = sum(_score_distribution_to_counts(existing_memory.score_distribution).values())
    new_total = old_total + len(scores)
    weighted_sum = existing_memory.avg_satisfaction_score * old_total + sum(scores)
    new_avg = weighted_sum / new_total if new_total else existing_memory.avg_satisfaction_score

    reqs = _dedupe_preserve_order(
        list(existing_memory.user_specific_requirements)
        + list(patch.add_user_specific_requirements)
    )[:5]

    task_obs_map = {
        obs.task_name: obs.model_copy()
        for obs in existing_memory.task_specific_observations
    }
    for obs in patch.add_or_update_task_specific_observations:
        task_obs_map[obs.task_name] = obs

    return UserMemory(
        avg_satisfaction_score=round(new_avg, 4),
        score_distribution=new_dist,
        scoring_style=patch.scoring_style if patch.update_scoring_style else existing_memory.scoring_style,
        four_vs_five_distinction=(
            patch.four_vs_five_distinction
            if patch.update_four_vs_five else existing_memory.four_vs_five_distinction
        ),
        three_vs_four_distinction=(
            patch.three_vs_four_distinction
            if patch.update_three_vs_four else existing_memory.three_vs_four_distinction
        ),
        user_specific_requirements=reqs,
        preferred_response_format=(
            patch.preferred_response_format
            if patch.update_preferred_response_format else existing_memory.preferred_response_format
        ),
        task_specific_observations=list(task_obs_map.values()),
        memory_version="v2",
        source_tasks=list(existing_memory.source_tasks),
        n_history_sessions=existing_memory.n_history_sessions + 1,
        n_history_turns=existing_memory.n_history_turns + len(turn_predictions),
    )


def merge_memory_v2_2_patch(
    existing_memory: UserMemory,
    patch: MemoryUpdatePatchV2_2,
    turn_predictions: list[dict],
    use_oracle_labels: bool = False,
) -> UserMemory:
    """将 v2.2 patch 合并回 v2 memory，并用更硬的程序 gate 控制 verbal 字段更新。"""
    scores = [
        int(pred["gold_score"] if use_oracle_labels else pred["pred_score"])
        for pred in turn_predictions
    ]
    old_counts = existing_memory.score_distribution.model_copy()
    counts_map = {
        1: old_counts.score_1,
        2: old_counts.score_2,
        3: old_counts.score_3,
        4: old_counts.score_4,
        5: old_counts.score_5,
    }
    for s in scores:
        counts_map[s] += 1
    new_dist = ScoreDistribution(
        score_1=counts_map[1],
        score_2=counts_map[2],
        score_3=counts_map[3],
        score_4=counts_map[4],
        score_5=counts_map[5],
    )
    old_total = sum(_score_distribution_to_counts(existing_memory.score_distribution).values())
    new_total = old_total + len(scores)
    weighted_sum = existing_memory.avg_satisfaction_score * old_total + sum(scores)
    new_avg = weighted_sum / new_total if new_total else existing_memory.avg_satisfaction_score

    new_score_counts = {s: scores.count(s) for s in range(1, 6)}
    has_3_and_4 = new_score_counts[3] > 0 and new_score_counts[4] > 0
    has_4_and_5 = new_score_counts[4] > 0 and new_score_counts[5] > 0

    allow_scoring_style = (
        patch.update_scoring_style
        and patch.scoring_style_confidence in {"medium", "high"}
        and len(scores) >= 3
    )
    allow_three_vs_four = (
        patch.update_three_vs_four
        and has_3_and_4
        and patch.three_vs_four_confidence in {"medium", "high"}
        and patch.three_vs_four_evidence_count >= 2
    )
    allow_four_vs_five = (
        patch.update_four_vs_five
        and has_4_and_5
        and patch.four_vs_five_confidence in {"medium", "high"}
        and patch.four_vs_five_evidence_count >= 2
    )
    allow_preferred_format = (
        patch.update_preferred_response_format
        and patch.preferred_response_format_confidence == "high"
        and patch.preferred_response_format_support_count >= 2
    )

    reqs = list(existing_memory.user_specific_requirements)
    for item in patch.add_user_specific_requirements:
        if item.confidence == "low" or item.support_count < 2:
            continue
        if _is_generic_requirement(item.requirement):
            continue
        reqs.append(item.requirement)
    reqs = _dedupe_preserve_order(reqs)[:5]

    task_obs_map = {
        obs.task_name: obs.model_copy()
        for obs in existing_memory.task_specific_observations
    }
    for obs in patch.add_or_update_task_specific_observations:
        if obs.confidence == "low" or obs.support_count < 1:
            continue
        task_obs_map[obs.task_name] = TaskObservation(
            task_name=obs.task_name,
            observation=obs.observation,
        )

    return UserMemory(
        avg_satisfaction_score=round(new_avg, 4),
        score_distribution=new_dist,
        scoring_style=patch.scoring_style if allow_scoring_style else existing_memory.scoring_style,
        four_vs_five_distinction=(
            patch.four_vs_five_distinction
            if allow_four_vs_five else existing_memory.four_vs_five_distinction
        ),
        three_vs_four_distinction=(
            patch.three_vs_four_distinction
            if allow_three_vs_four else existing_memory.three_vs_four_distinction
        ),
        user_specific_requirements=reqs,
        preferred_response_format=(
            patch.preferred_response_format
            if allow_preferred_format else existing_memory.preferred_response_format
        ),
        task_specific_observations=list(task_obs_map.values()),
        memory_version="v2",
        source_tasks=list(existing_memory.source_tasks),
        n_history_sessions=existing_memory.n_history_sessions + 1,
        n_history_turns=existing_memory.n_history_turns + len(turn_predictions),
    )


def merge_memory_v2_3_patch(
    existing_memory: UserMemory,
    patch: MemoryUpdatePatchV2_1,
    turn_predictions: list[dict],
    use_oracle_labels: bool = False,
) -> UserMemory:
    """
    将 v2.3 patch 合并回 v2 memory。

    相比 v2.1：
      - 保留 field-level patch 框架
      - 只增加轻量 boundary gate
      - 继续过滤泛化 requirement
    """
    scores = [
        int(pred["gold_score"] if use_oracle_labels else pred["pred_score"])
        for pred in turn_predictions
    ]
    old_counts = existing_memory.score_distribution.model_copy()
    counts_map = {
        1: old_counts.score_1,
        2: old_counts.score_2,
        3: old_counts.score_3,
        4: old_counts.score_4,
        5: old_counts.score_5,
    }
    for s in scores:
        counts_map[s] += 1
    new_dist = ScoreDistribution(
        score_1=counts_map[1],
        score_2=counts_map[2],
        score_3=counts_map[3],
        score_4=counts_map[4],
        score_5=counts_map[5],
    )
    old_total = sum(_score_distribution_to_counts(existing_memory.score_distribution).values())
    new_total = old_total + len(scores)
    weighted_sum = existing_memory.avg_satisfaction_score * old_total + sum(scores)
    new_avg = weighted_sum / new_total if new_total else existing_memory.avg_satisfaction_score

    new_score_counts = {s: scores.count(s) for s in range(1, 6)}
    has_3_and_4 = new_score_counts[3] > 0 and new_score_counts[4] > 0
    has_4_and_5 = new_score_counts[4] > 0 and new_score_counts[5] > 0

    reqs = list(existing_memory.user_specific_requirements)
    for item in patch.add_user_specific_requirements:
        if _is_generic_requirement(item):
            continue
        reqs.append(item)
    reqs = _dedupe_preserve_order(reqs)[:5]

    task_obs_map = {
        obs.task_name: obs.model_copy()
        for obs in existing_memory.task_specific_observations
    }
    for obs in patch.add_or_update_task_specific_observations:
        task_obs_map[obs.task_name] = obs

    return UserMemory(
        avg_satisfaction_score=round(new_avg, 4),
        score_distribution=new_dist,
        scoring_style=patch.scoring_style if patch.update_scoring_style else existing_memory.scoring_style,
        four_vs_five_distinction=(
            patch.four_vs_five_distinction
            if patch.update_four_vs_five and has_4_and_5
            else existing_memory.four_vs_five_distinction
        ),
        three_vs_four_distinction=(
            patch.three_vs_four_distinction
            if patch.update_three_vs_four and has_3_and_4
            else existing_memory.three_vs_four_distinction
        ),
        user_specific_requirements=reqs,
        preferred_response_format=(
            patch.preferred_response_format
            if patch.update_preferred_response_format else existing_memory.preferred_response_format
        ),
        task_specific_observations=list(task_obs_map.values()),
        memory_version="v2",
        source_tasks=list(existing_memory.source_tasks),
        n_history_sessions=existing_memory.n_history_sessions + 1,
        n_history_turns=existing_memory.n_history_turns + len(turn_predictions),
    )


def merge_memory_v2_4_patch(
    existing_memory: UserMemory,
    patch: MemoryUpdatePatchV2_1,
    turn_predictions: list[dict],
    use_oracle_labels: bool = False,
) -> UserMemory:
    """
    将 v2.4 patch 合并回 v2 memory。

    相比 v2.1 只做轻量保护：
      - 统计量更新完全保持 v2.1
      - scoring_style / preferred_response_format / task observations 保持 v2.1
      - boundary 字段要求相邻分数证据
      - requirement 过滤泛化文本
    """
    scores = [
        int(pred["gold_score"] if use_oracle_labels else pred["pred_score"])
        for pred in turn_predictions
    ]
    old_counts = existing_memory.score_distribution.model_copy()
    counts_map = {
        1: old_counts.score_1,
        2: old_counts.score_2,
        3: old_counts.score_3,
        4: old_counts.score_4,
        5: old_counts.score_5,
    }
    for s in scores:
        counts_map[s] += 1
    new_dist = ScoreDistribution(
        score_1=counts_map[1],
        score_2=counts_map[2],
        score_3=counts_map[3],
        score_4=counts_map[4],
        score_5=counts_map[5],
    )
    old_total = sum(_score_distribution_to_counts(existing_memory.score_distribution).values())
    new_total = old_total + len(scores)
    weighted_sum = existing_memory.avg_satisfaction_score * old_total + sum(scores)
    new_avg = weighted_sum / new_total if new_total else existing_memory.avg_satisfaction_score

    new_score_counts = {s: scores.count(s) for s in range(1, 6)}
    has_3_and_4 = new_score_counts[3] > 0 and new_score_counts[4] > 0
    has_4_and_5 = new_score_counts[4] > 0 and new_score_counts[5] > 0

    reqs = list(existing_memory.user_specific_requirements)
    for item in patch.add_user_specific_requirements:
        if _is_generic_requirement(item):
            continue
        reqs.append(item)
    reqs = _dedupe_preserve_order(reqs)[:5]

    task_obs_map = {
        obs.task_name: obs.model_copy()
        for obs in existing_memory.task_specific_observations
    }
    for obs in patch.add_or_update_task_specific_observations:
        task_obs_map[obs.task_name] = obs

    return UserMemory(
        avg_satisfaction_score=round(new_avg, 4),
        score_distribution=new_dist,
        scoring_style=patch.scoring_style if patch.update_scoring_style else existing_memory.scoring_style,
        four_vs_five_distinction=(
            patch.four_vs_five_distinction
            if patch.update_four_vs_five and has_4_and_5
            else existing_memory.four_vs_five_distinction
        ),
        three_vs_four_distinction=(
            patch.three_vs_four_distinction
            if patch.update_three_vs_four and has_3_and_4
            else existing_memory.three_vs_four_distinction
        ),
        user_specific_requirements=reqs,
        preferred_response_format=(
            patch.preferred_response_format
            if patch.update_preferred_response_format else existing_memory.preferred_response_format
        ),
        task_specific_observations=list(task_obs_map.values()),
        memory_version="v2",
        source_tasks=list(existing_memory.source_tasks),
        n_history_sessions=existing_memory.n_history_sessions + 1,
        n_history_turns=existing_memory.n_history_turns + len(turn_predictions),
    )


def merge_memory_v2_5_patch(
    existing_memory: UserMemory,
    patch: MemoryUpdatePatchV2_1,
    turn_predictions: list[dict],
    use_oracle_labels: bool = False,
) -> UserMemory:
    """
    将 v2.5 patch 合并回 v2 memory。

    相比 v2.1：
      - 非 oracle 时冻结 scoring_style
      - 非 oracle 时对 avg_satisfaction_score 的上移做轻量阻尼
      - requirement 过滤泛化文本
      - 其他字段保持 v2.1 的轻量 patch 行为
    """
    scores = [
        int(pred["gold_score"] if use_oracle_labels else pred["pred_score"])
        for pred in turn_predictions
    ]
    old_counts = existing_memory.score_distribution.model_copy()
    counts_map = {
        1: old_counts.score_1,
        2: old_counts.score_2,
        3: old_counts.score_3,
        4: old_counts.score_4,
        5: old_counts.score_5,
    }
    for s in scores:
        counts_map[s] += 1
    new_dist = ScoreDistribution(
        score_1=counts_map[1],
        score_2=counts_map[2],
        score_3=counts_map[3],
        score_4=counts_map[4],
        score_5=counts_map[5],
    )
    old_total = sum(_score_distribution_to_counts(existing_memory.score_distribution).values())
    new_total = old_total + len(scores)
    weighted_sum = existing_memory.avg_satisfaction_score * old_total + sum(scores)
    raw_new_avg = weighted_sum / new_total if new_total else existing_memory.avg_satisfaction_score
    if use_oracle_labels or raw_new_avg <= existing_memory.avg_satisfaction_score:
        new_avg = raw_new_avg
    else:
        # Predicted-label updates often drift SAT-heavy; allow only a damped upward prior shift.
        new_avg = existing_memory.avg_satisfaction_score + 0.25 * (
            raw_new_avg - existing_memory.avg_satisfaction_score
        )

    reqs = list(existing_memory.user_specific_requirements)
    for item in patch.add_user_specific_requirements:
        if _is_generic_requirement(item):
            continue
        reqs.append(item)
    reqs = _dedupe_preserve_order(reqs)[:5]

    task_obs_map = {
        obs.task_name: obs.model_copy()
        for obs in existing_memory.task_specific_observations
    }
    for obs in patch.add_or_update_task_specific_observations:
        task_obs_map[obs.task_name] = obs

    return UserMemory(
        avg_satisfaction_score=round(new_avg, 4),
        score_distribution=new_dist,
        scoring_style=(
            patch.scoring_style
            if patch.update_scoring_style and use_oracle_labels
            else existing_memory.scoring_style
        ),
        four_vs_five_distinction=(
            patch.four_vs_five_distinction
            if patch.update_four_vs_five else existing_memory.four_vs_five_distinction
        ),
        three_vs_four_distinction=(
            patch.three_vs_four_distinction
            if patch.update_three_vs_four else existing_memory.three_vs_four_distinction
        ),
        user_specific_requirements=reqs,
        preferred_response_format=(
            patch.preferred_response_format
            if patch.update_preferred_response_format else existing_memory.preferred_response_format
        ),
        task_specific_observations=list(task_obs_map.values()),
        memory_version="v2",
        source_tasks=list(existing_memory.source_tasks),
        n_history_sessions=existing_memory.n_history_sessions + 1,
        n_history_turns=existing_memory.n_history_turns + len(turn_predictions),
    )
