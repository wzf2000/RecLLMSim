from __future__ import annotations

import json
import os

from loguru import logger
from tenacity import before_sleep_log, retry, stop_after_attempt, wait_fixed

from lib.memory import UserMemory, UserMemoryContent
from lib.personalized_data import PersonalizedSample, SessionData
from lib.urs_memory import build_urs_memory_prompt, build_urs_memory_update_prompt
from trace.collect_personalized import _structured_parse


@retry(
    stop=stop_after_attempt(3),
    wait=wait_fixed(5),
    before_sleep=before_sleep_log(logger, log_level=40),
)
def _call_build_memory(prompt: str, model: str) -> UserMemoryContent:
    return _structured_parse(
        prompt,
        model,
        UserMemoryContent,
        temperature=0.3,
        timeout=120,
        system_msg="You are an expert user behavior analyst.",
    )


def build_user_memory_urs(
    sample: PersonalizedSample,
    model: str,
    memory_cache_dir: str | None = None,
) -> UserMemory:
    cache_key = f"{sample.user}__{sample.target_task}__{model.replace('/', '_')}"
    cache_path = (
        os.path.join(memory_cache_dir, f"{cache_key}.json")
        if memory_cache_dir else None
    )
    if cache_path and os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as fp:
                data = json.load(fp)
            mem = UserMemory(**data)
            if mem.memory_version == "v2":
                return mem
        except Exception as e:
            logger.debug(f"URS cache load failed ({e}), rebuilding: {cache_path}")

    prompt = build_urs_memory_prompt(
        user_id=sample.user,
        profile=sample.profile,
        history_sessions=sample.history_sessions,
    )
    content = _call_build_memory(prompt, model)
    memory = UserMemory.from_content(
        content,
        source_tasks=sample.history_tasks,
        n_history_sessions=sample.n_history_sessions,
        n_history_turns=sum(len(s.satisfaction_scores) for s in sample.history_sessions),
    )
    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as fp:
            json.dump(memory.model_dump(), fp, ensure_ascii=False, indent=2)
    return memory


# ──────────────────────────────────────────────────────────────────────────────
# 单 session 评估
# ──────────────────────────────────────────────────────────────────────────────



@retry(
    stop=stop_after_attempt(3),
    wait=wait_fixed(5),
    before_sleep=before_sleep_log(logger, log_level=40),
)
def _call_update_memory(prompt: str, model: str):
    from lib.memory import UserMemoryContent
    return _structured_parse(
        prompt, model, UserMemoryContent,
        temperature=0.3, timeout=120,
        system_msg="You are an expert user behavior analyst.",
    )


def update_memory_urs(
    memory: UserMemory,
    session: SessionData,
    turn_predictions: list[dict],
    model: str,
    use_oracle_labels: bool = False,
) -> UserMemory:
    prompt = build_urs_memory_update_prompt(
        existing_memory=memory,
        new_session=session,
        turn_predictions=turn_predictions,
        use_oracle_labels=use_oracle_labels,
    )
    content = _call_update_memory(prompt, model)
    # URS 下 assistant_turns=1，所以 n_history_turns += 1
    return UserMemory.from_content(
        content,
        source_tasks=memory.source_tasks,
        n_history_sessions=memory.n_history_sessions + 1,
        n_history_turns=memory.n_history_turns + 1,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Block-level 推理（一个 PersonalizedSample = 一个 (user, target_intent)）
# ──────────────────────────────────────────────────────────────────────────────

