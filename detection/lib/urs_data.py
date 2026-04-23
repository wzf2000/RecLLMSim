"""
URS 数据集加载 + cross-intent 划分

与 personalized_data.py 对齐的 API，核心差异：
  1. user_id 必须加语言前缀（zh_ / en_）——原始 JSON 中 zh 和 en 的 uid 命名空间独立
  2. label 是 session-level：每个 session 一个 1-5 分的 user_satisfaction
     在 SessionData 里 satisfaction_scores 恒为单元素列表 [session_score]
     dissatisfaction_reasons 同构，填 "满意" 或 "其它"
  3. 没有 profile 字段；profile 置空 dict（memory prompt 对空 profile 有 fallback）

Cross-Intent Split（对标原 Cross-Task Split）：
  对每个 (user, target_intent) 对：
    history_sessions = 该用户在其他 intent 下的所有 session（session-level 标签可见）
    target_sessions  = 该用户在目标 intent 下的所有 session（session-level 标签待预测）
  用户层面：GroupShuffleSplit 80% test / 20% train
"""

from __future__ import annotations

import json
import os
from typing import Literal

from loguru import logger
from sklearn.model_selection import GroupShuffleSplit

from .personalized_data import PersonalizedSample, SessionData

# ──────────────────────────────────────────────────────────────────────────────
# 常量 / 映射
# ──────────────────────────────────────────────────────────────────────────────

URS_DATA_DIR: str = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "urs",
)

URS_SOURCE_FILES: dict[str, str] = {
    "zh": "chinese_merged.json",
    "en": "english_processed.json",
}

# canonical intent 分类（zh ↔ en 统一成英文 slug）
URS_INTENT_CANONICAL: dict[str, str] = {
    # zh
    "解决专业问题": "professional",
    "检索信息":    "retrieval",
    "文本事务":    "text",
    "获取建议":    "advice",
    "寻求创意":    "creative",
    "娱乐休闲":    "leisure",
    "其他":        "other",
    # en
    "Solve Professional Problem": "professional",
    "Information Retrieval":      "retrieval",
    "Text Assistant":             "text",
    "Ask for Advice":             "advice",
    "Seek Creativity":            "creative",
    "Leisure":                    "leisure",
    "Others":                     "other",
}

URS_INTENT_LIST: list[str] = [
    "professional", "retrieval", "text", "advice", "creative", "leisure", "other",
]

# 满意度 5 档 ordinal 映射
URS_SATISFACTION_MAP: dict[str, int] = {
    # zh
    "很不满意":   1,
    "不满意":     2,
    "一般":       3,
    "满意":       4,
    "非常满意":   5,
    # en
    "very dissatisfied": 1,
    "dissatisfied":      2,
    "neutral":           3,
    "satisfied":         4,
    "very satisfied":    5,
}

Language = Literal["zh", "en"]


# ──────────────────────────────────────────────────────────────────────────────
# 数据加载
# ──────────────────────────────────────────────────────────────────────────────

def _session_to_session_data(
    item: dict,
    lang: Language,
    data_file: str,
    item_idx: int,
) -> SessionData | None:
    """把一条 URS item 转换成 SessionData（单 turn 语义）。"""
    sat_raw = item.get("user_satisfaction")
    score = URS_SATISFACTION_MAP.get(sat_raw)
    if score is None:
        return None

    intent_raw = item.get("intent_label", "")
    canonical = URS_INTENT_CANONICAL.get(intent_raw)
    if canonical is None:
        return None

    uid = str(item.get("user_id", ""))
    if not uid:
        return None
    user = f"{lang}_{uid}"

    history = [
        {"role": utt["role"], "content": utt["content"]}
        for utt in item.get("conversation_history", [])
        if utt.get("role") in ("user", "assistant")
    ]
    if not history:
        return None

    # task_context：用 title + intent 作为对话主题描述
    title = (item.get("title") or "").strip()
    task_context = f"[{canonical}] {title}" if title else f"[{canonical}]"

    # session-level 标签只装 1 个元素（pipeline 按 "1 assistant turn" 处理）
    reason = "满意" if score >= 4 else "其它"

    # 伪 file_path：保证 sample_id 唯一；URS 原始 JSON 没有独立文件名
    file_path = f"urs::{lang}::{item_idx:05d}.json"

    return SessionData(
        user=user,
        task=canonical,
        file_path=file_path,
        task_context=task_context,
        profile={},  # URS 无 profile
        history=history,
        satisfaction_scores=[score],
        dissatisfaction_reasons=[reason],
        chat_model=str(item.get("llm", "unknown")),
    )


def load_urs_sessions(
    data_dir: str = URS_DATA_DIR,
    languages: tuple[Language, ...] = ("zh", "en"),
) -> dict[str, dict[str, list[SessionData]]]:
    """
    加载 URS session。

    参数
    ----
    languages : tuple[str, ...]
        要加载的语言子集；默认同时加载 zh+en。
        加载多语言时，用户 id 自动加前缀（zh_1 / en_1）避免命名空间冲突。

    返回
    ----
    {user: {intent: [SessionData, ...]}}
    """
    result: dict[str, dict[str, list[SessionData]]] = {}

    for lang in languages:
        fname = URS_SOURCE_FILES.get(lang)
        if fname is None:
            logger.warning(f"[urs_data] unknown language: {lang}, skipped")
            continue
        fpath = os.path.join(data_dir, fname)
        if not os.path.exists(fpath):
            logger.warning(f"[urs_data] missing file: {fpath}, skipped")
            continue

        with open(fpath, "r", encoding="utf-8") as fp:
            data = json.load(fp)

        for idx, item in enumerate(data):
            sd = _session_to_session_data(item, lang, fname, idx)
            if sd is None:
                continue
            result.setdefault(sd.user, {}).setdefault(sd.task, []).append(sd)

    return result


# ──────────────────────────────────────────────────────────────────────────────
# Cross-Intent Split + 用户级 train/test 划分
# ──────────────────────────────────────────────────────────────────────────────

def build_urs_personalized_samples(
    split: str = "test",
    train_ratio: float = 0.2,
    seed: int = 42,
    min_history_sessions: int = 1,
    target_intents: list[str] | None = None,
    languages: tuple[Language, ...] = ("zh", "en"),
    data_dir: str = URS_DATA_DIR,
) -> list[PersonalizedSample]:
    """
    构建 URS 的个性化预测样本。

    与 build_personalized_samples 行为一致：
      history_sessions = 该用户在其他 intent 下的所有 session
      target_sessions  = 该用户在目标 intent 下的所有 session
    用户层面 GroupShuffleSplit 划分。
    """
    if target_intents is None:
        target_intents = URS_INTENT_LIST

    all_sessions = load_urs_sessions(data_dir=data_dir, languages=languages)

    all_samples: list[PersonalizedSample] = []
    for user, intent_sessions in all_sessions.items():
        # URS 无 profile，都是空 dict
        profile: dict = {}

        for target_intent in target_intents:
            target_sessions = intent_sessions.get(target_intent, [])
            if not target_sessions:
                continue

            history_sessions: list[SessionData] = []
            for other_intent, sessions in intent_sessions.items():
                if other_intent == target_intent:
                    continue
                history_sessions.extend(sessions)

            if len(history_sessions) < min_history_sessions:
                continue

            all_samples.append(
                PersonalizedSample(
                    user=user,
                    profile=profile,
                    target_task=target_intent,
                    history_sessions=history_sessions,
                    target_sessions=target_sessions,
                )
            )

    if not all_samples:
        return []

    if split == "all":
        logger.info(
            f"[urs_data] split=all, samples={len(all_samples)}, "
            f"users={len({s.user for s in all_samples})}"
        )
        return all_samples

    user_list = [s.user for s in all_samples]

    gss = GroupShuffleSplit(
        n_splits=1,
        train_size=train_ratio,
        test_size=1.0 - train_ratio,
        random_state=seed,
    )
    indices = list(range(len(all_samples)))
    train_rel, test_rel = next(gss.split(indices, groups=user_list))

    split_map = {
        "train": [all_samples[i] for i in train_rel],
        "test":  [all_samples[i] for i in test_rel],
    }
    selected = split_map[split]

    train_users = {all_samples[i].user for i in train_rel}
    test_users  = {all_samples[i].user for i in test_rel}
    logger.info(
        f"[urs_data] split={split}, languages={languages}, "
        f"train_users={len(train_users)}, test_users={len(test_users)}, "
        f"selected_samples={len(selected)}, "
        f"total_target_sessions={sum(len(s.target_sessions) for s in selected)}"
    )
    return selected


# ──────────────────────────────────────────────────────────────────────────────
# 工具函数
# ──────────────────────────────────────────────────────────────────────────────

def urs_dataset_stats(samples: list[PersonalizedSample]) -> dict:
    users = {s.user for s in samples}
    languages = {u.split("_", 1)[0] for u in users}
    target_sessions = sum(len(s.target_sessions) for s in samples)
    history_per_sample = [s.n_history_sessions for s in samples]
    return {
        "n_users": len(users),
        "languages": sorted(languages),
        "n_blocks": len(samples),
        "n_target_sessions": target_sessions,
        "avg_history_sessions": (
            sum(history_per_sample) / len(history_per_sample) if history_per_sample else 0
        ),
        "min_history_sessions": min(history_per_sample) if history_per_sample else 0,
        "max_history_sessions": max(history_per_sample) if history_per_sample else 0,
    }
