"""
USS Cross-Dataset 个性化满意度推理适配层

USS 数据缺少持续的 user identity（dialogue_id 不复用），无法直接套 cross-task split。
本模块给出两条降级路线，把 USS 数据装成 PersonalizedSample-compatible 结构，
让 trace/collect_personalized.py::run_agent_on_sample 可以直接复用。

R1. Dialogue Warm-up（per-dialogue）
  把每条 dialogue 的前 n_warm 个 assistant turn 当作"历史"（满意度可见），
  剩余 turn 当作"目标"。memory cache 按 (dialogue_id, subset, model) 单独建。
  CDF 校准用 warm-up 的 score_distribution。

R2. Subset Population Rubric（per-subset）
  从 train split 随机采 K 条 dialogue 作为"人群历史"，构造一份 subset 级
  memory（cache key = `{subset}_population`），所有 test dialogue 共用。
  没有 per-user 个性化，验证的是 memory v2 rubric 在领域级是否仍带来增益。

两条路线产出的 PersonalizedSample 都满足：
  - sample.user / sample.target_task 唯一可作 cache key
  - sample.target_sessions[i].satisfaction_scores 长度 == 该 dialogue 的 assistant turn 数
  - sample.history_sessions 至少包含 1 个 SessionData（满足 min_history_sessions=1）

Warm-up 模式的"目标 session"包含完整 dialogue（含 warm-up 部分）；
trace/collect_uss.py 在拿到预测后**按 turn_idx 过滤掉 warm-up 部分**，
避免 prompt 里需要传"上下文起点 != 预测起点"的特殊参数。
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import replace

from loguru import logger

from .personalized_data import PersonalizedSample, SessionData
from .uss_data import ALL_DATASETS, _DEFAULT_USS_DIR, _DUMMY_PROFILE, _load_jsonl

# ──────────────────────────────────────────────────────────────────────────────
# 常量 / 工具
# ──────────────────────────────────────────────────────────────────────────────

USS_DATA_DIR: str = _DEFAULT_USS_DIR


def _score_to_reason(score: int) -> str:
    return "满意" if score >= 4 else "其它"


def _load_subset_records(
    subset: str,
    splits: list[str],
    data_dir: str = USS_DATA_DIR,
) -> list[dict]:
    """读取 USS 预处理 JSONL 并按 split 过滤。"""
    path = os.path.join(data_dir, f"{subset}.jsonl")
    if not os.path.exists(path):
        logger.warning(f"USS processed file missing: {path}")
        return []
    records = _load_jsonl(path)
    splits_set = set(splits)
    return [r for r in records if r.get("split") in splits_set]


def _group_by_dialogue(records: list[dict]) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = {}
    for r in records:
        groups.setdefault(r["dialogue_id"], []).append(r)
    for did in groups:
        groups[did].sort(key=lambda x: x.get("turn_idx", 0))
    return groups


def _dialogue_to_session_data(
    subset: str,
    dialogue_id: str,
    turn_records: list[dict],
    file_suffix: str = "full",
    score_slice: slice | None = None,
    history_slice: slice | None = None,
) -> SessionData | None:
    """
    把同一 dialogue 的 turn_records 拼成 SessionData。

    history 用最后一条 record 的 history + assistant_reply 拼出完整对话。
    score_slice / history_slice 可选，用于 warm-up 模式切出子序列。
    """
    if not turn_records:
        return None

    # 重建 [{role, content}, ...] 的完整对话
    full_history: list[dict] = list(turn_records[-1].get("history", []))
    full_history.append(
        {"role": "assistant", "content": turn_records[-1].get("assistant_reply", "")}
    )

    if history_slice is not None:
        full_history = full_history[history_slice]

    satisfaction_scores = [r["gold_score"] for r in turn_records]
    if score_slice is not None:
        satisfaction_scores = satisfaction_scores[score_slice]

    reasons = [_score_to_reason(s) for s in satisfaction_scores]

    first = turn_records[0]
    file_path = f"uss::{subset}::{dialogue_id}::{file_suffix}.json"

    return SessionData(
        user=dialogue_id,
        task=subset,
        file_path=file_path,
        task_context=first.get("task_context", "") or f"USS subset = {subset}",
        profile=_DUMMY_PROFILE,
        history=full_history,
        satisfaction_scores=satisfaction_scores,
        dissatisfaction_reasons=reasons,
        chat_model=first.get("model", "unknown"),
    )


def _truncate_history_to_assistant_count(
    history: list[dict],
    n_assistant_turns: int,
) -> list[dict]:
    """保留前 n_assistant_turns 个 assistant turn 及其前面的 user turn。"""
    out: list[dict] = []
    seen = 0
    for utt in history:
        out.append(utt)
        if utt.get("role") == "assistant":
            seen += 1
            if seen >= n_assistant_turns:
                break
    return out


# ──────────────────────────────────────────────────────────────────────────────
# R1. Dialogue Warm-up
# ──────────────────────────────────────────────────────────────────────────────

def build_uss_warmup_samples(
    subsets: list[str] | None = None,
    splits: list[str] = ("test",),
    warmup_turns: int = 5,
    min_warmup_turns: int = 3,
    min_target_turns: int = 1,
    data_dir: str = USS_DATA_DIR,
) -> list[PersonalizedSample]:
    """
    R1：每条 dialogue 拆 (warm-up, full)。

    返回的 PersonalizedSample：
      user           = dialogue_id
      target_task    = subset
      history_sessions = [warmup_pseudo_session]   （前 warmup_turns 个 assistant turn）
      target_sessions  = [full_dialogue_session]   （完整 dialogue，含 warm-up 部分）

    trace/collect_uss.py 拿到结果后会按 turn_idx >= warmup_turns 过滤，只保留
    "真正的目标" turn，warm-up turn 仅用于 memory 与 CDF 校准。

    过滤规则：
      - 该 dialogue assistant 总数 >= warmup_turns + min_target_turns
      - warm-up 长度至少 min_warmup_turns（防止 dialogue 太短）
    """
    if subsets is None:
        subsets = ALL_DATASETS

    samples: list[PersonalizedSample] = []
    skipped_too_short = 0

    for subset in subsets:
        records = _load_subset_records(subset, list(splits), data_dir=data_dir)
        for did, turns in _group_by_dialogue(records).items():
            n_assist = len(turns)
            n_warm = min(warmup_turns, n_assist - min_target_turns)
            if n_warm < min_warmup_turns:
                skipped_too_short += 1
                continue

            warmup_records = turns[:n_warm]
            warmup_session = _dialogue_to_session_data(
                subset=subset,
                dialogue_id=did,
                turn_records=warmup_records,
                file_suffix="warmup",
                history_slice=None,  # warmup_records 已只含前 n_warm turn
            )
            # warmup_session.history 通过 last record 重建——可能含到 warm-up 末尾的所有内容，
            # 但若有更多 turn 在 turn_records[-1].history 里就会包含进来。这里用 truncate 修正。
            if warmup_session is not None:
                warmup_session.history = _truncate_history_to_assistant_count(
                    warmup_session.history, n_warm
                )

            target_session = _dialogue_to_session_data(
                subset=subset,
                dialogue_id=did,
                turn_records=turns,
                file_suffix="target",
            )
            if warmup_session is None or target_session is None:
                continue

            samples.append(
                PersonalizedSample(
                    user=did,
                    profile=_DUMMY_PROFILE,
                    target_task=subset,
                    history_sessions=[warmup_session],
                    target_sessions=[target_session],
                )
            )

    logger.info(
        f"[uss_pipeline] R1 warmup samples: {len(samples)} "
        f"(skipped {skipped_too_short} dialogues for being too short)"
    )
    return samples


# ──────────────────────────────────────────────────────────────────────────────
# R2. Subset Population Rubric
# ──────────────────────────────────────────────────────────────────────────────

def build_uss_population_samples(
    subsets: list[str] | None = None,
    train_splits: list[str] = ("train",),
    test_splits: list[str] = ("test",),
    n_population_dialogues: int = 8,
    seed: int = 42,
    data_dir: str = USS_DATA_DIR,
) -> list[PersonalizedSample]:
    """
    R2：每个 subset 共享一份 population memory。

    对每个 subset：
      1. 从 train 中随机抽 n_population_dialogues 条 dialogue 当 history（标签可见）
      2. 每条 test dialogue 单独成 1 个 PersonalizedSample，其 history_sessions 共享步骤 1 的列表
      3. user = `{subset}_population`，所有 test dialogue 共用这个 cache key
         → memory 在第一次 build 后会被缓存，后续 dialogue 直接命中缓存

    返回 PersonalizedSample：
      user             = `{subset}_population`
      target_task      = subset
      history_sessions = K 条 train dialogue 的 SessionData（每个 subset 内共享同一对象）
      target_sessions  = [test_dialogue_session]
    """
    if subsets is None:
        subsets = ALL_DATASETS

    rng = random.Random(seed)
    samples: list[PersonalizedSample] = []

    for subset in subsets:
        train_records = _load_subset_records(subset, list(train_splits), data_dir=data_dir)
        train_groups = _group_by_dialogue(train_records)
        train_dialogues = list(train_groups.items())
        if not train_dialogues:
            logger.warning(f"[uss_pipeline] {subset}: no train dialogues, skip")
            continue

        # 采样人群基底 dialogues
        rng.shuffle(train_dialogues)
        sampled = train_dialogues[:n_population_dialogues]
        history_sessions: list[SessionData] = []
        for did, turns in sampled:
            sd = _dialogue_to_session_data(
                subset=subset, dialogue_id=did, turn_records=turns,
                file_suffix="population_basis",
            )
            if sd is not None:
                history_sessions.append(sd)
        if not history_sessions:
            logger.warning(f"[uss_pipeline] {subset}: no usable train dialogues, skip")
            continue

        # 收集 test dialogues
        test_records = _load_subset_records(subset, list(test_splits), data_dir=data_dir)
        test_groups = _group_by_dialogue(test_records)

        population_user = f"{subset}_population"
        for did, turns in test_groups.items():
            target_session = _dialogue_to_session_data(
                subset=subset, dialogue_id=did, turn_records=turns,
                file_suffix="target",
            )
            if target_session is None:
                continue
            # 注意：target_session.user 仍是真实 dialogue_id，但 PersonalizedSample.user
            # 用 population key 以共享 memory cache
            samples.append(
                PersonalizedSample(
                    user=population_user,
                    profile=_DUMMY_PROFILE,
                    target_task=subset,
                    history_sessions=history_sessions,
                    target_sessions=[target_session],
                )
            )

        logger.info(
            f"[uss_pipeline] R2 {subset}: population_basis={len(history_sessions)} dialogues, "
            f"test_blocks={len(test_groups)}"
        )

    logger.info(f"[uss_pipeline] R2 population samples total: {len(samples)}")
    return samples


# ──────────────────────────────────────────────────────────────────────────────
# 统计工具
# ──────────────────────────────────────────────────────────────────────────────

def uss_dataset_stats(samples: list[PersonalizedSample]) -> dict:
    n_blocks = len(samples)
    target_turns = sum(
        len(s.satisfaction_scores)
        for sample in samples
        for s in sample.target_sessions
    )
    by_subset: dict[str, int] = {}
    for sample in samples:
        by_subset[sample.target_task] = by_subset.get(sample.target_task, 0) + 1
    return {
        "n_blocks": n_blocks,
        "n_target_turns": target_turns,
        "blocks_by_subset": dict(sorted(by_subset.items())),
        "n_distinct_users": len({s.user for s in samples}),
    }
