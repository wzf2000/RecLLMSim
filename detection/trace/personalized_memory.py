"""Memory build/update helpers for personalized trace collection."""

from __future__ import annotations

import json
import os
from typing import Callable, Literal

from loguru import logger
from pydantic import BaseModel
from tenacity import before_sleep_log, retry, stop_after_attempt, wait_fixed

from lib.memory import (
    MemoryUpdatePatchV2_1,
    MemoryUpdatePatchV2_2,
    UserMemory,
    UserMemoryContent,
    UserMemoryContentV3,
    UserMemoryV3,
    build_memory_prompt,
    build_memory_prompt_v3,
    build_memory_update_prompt,
    build_memory_update_prompt_v2_1,
    build_memory_update_prompt_v2_2,
    build_memory_update_prompt_v2_3,
    build_memory_update_prompt_v2_4,
    build_memory_update_prompt_v2_5,
    build_memory_update_prompt_v3,
    merge_memory_v2_1_patch,
    merge_memory_v2_2_patch,
    merge_memory_v2_3_patch,
    merge_memory_v2_4_patch,
    merge_memory_v2_5_patch,
)
from lib.personalized_data import PersonalizedSample, SessionData

MemoryVersion = Literal["v2", "v3"]
MemoryUpdatePromptVersion = Literal["auto", "v2", "v2_1", "v2_2", "v2_3", "v2_4", "v2_5", "v3"]
StructuredParseFn = Callable[..., BaseModel]


@retry(
    stop=stop_after_attempt(3),
    wait=wait_fixed(5),
    before_sleep=before_sleep_log(logger, log_level=40),
)
def call_build_memory(
    prompt: str,
    model: str,
    parse_fn: StructuredParseFn,
    memory_version: MemoryVersion = "v2",
) -> UserMemoryContent | UserMemoryContentV3:
    try:
        return parse_fn(
            prompt,
            model,
            UserMemoryContent if memory_version == "v2" else UserMemoryContentV3,
            temperature=0.3,
            timeout=120,
            system_msg="You are an expert user behavior analyst.",
        )
    except Exception as e:
        logger.error(f"Memory building failed for {model}: {e}")
        raise e


def build_user_memory(
    sample: PersonalizedSample,
    model: str,
    parse_fn: StructuredParseFn,
    memory_cache_dir: str | None = None,
    memory_version: MemoryVersion = "v2",
) -> UserMemory | UserMemoryV3:
    """
    Build user memory for one sample, optionally loading/writing a cache file.

    Cache file name: {user}__{target_task}__{model}.json for v2, with the
    memory version suffix for later schemas.
    """
    cache_key = f"{sample.user}__{sample.target_task}__{model.replace('/', '_')}"
    cache_file = (
        f"{cache_key}.json"
        if memory_version == "v2"
        else f"{cache_key}__{memory_version}.json"
    )
    cache_path = (
        os.path.join(memory_cache_dir, cache_file)
        if memory_cache_dir
        else None
    )

    if cache_path and os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as fp:
                data = json.load(fp)
            mem = UserMemory(**data) if memory_version == "v2" else UserMemoryV3(**data)
            if mem.memory_version == memory_version:
                return mem
            logger.debug(f"Cache version mismatch ({mem.memory_version}), rebuilding: {cache_path}")
        except Exception as e:
            logger.debug(f"Cache load failed ({e}), rebuilding: {cache_path}")

    if memory_version == "v2":
        prompt = build_memory_prompt(
            user_id=sample.user,
            profile=sample.profile,
            history_sessions=sample.history_sessions,
        )
        content = call_build_memory(prompt, model, parse_fn=parse_fn, memory_version="v2")
        assert isinstance(content, UserMemoryContent)
        memory = UserMemory.from_content(
            content,
            source_tasks=sample.history_tasks,
            n_history_sessions=sample.n_history_sessions,
            n_history_turns=sum(s.assistant_turns for s in sample.history_sessions),
        )
    else:
        prompt = build_memory_prompt_v3(
            user_id=sample.user,
            profile=sample.profile,
            history_sessions=sample.history_sessions,
        )
        content = call_build_memory(prompt, model, parse_fn=parse_fn, memory_version="v3")
        assert isinstance(content, UserMemoryContentV3)
        memory = UserMemoryV3.from_content(
            content,
            source_tasks=sample.history_tasks,
            n_history_sessions=sample.n_history_sessions,
            n_history_turns=sum(s.assistant_turns for s in sample.history_sessions),
        )

    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as fp:
            json.dump(memory.model_dump(), fp, ensure_ascii=False, indent=2)

    return memory


@retry(
    stop=stop_after_attempt(3),
    wait=wait_fixed(5),
    before_sleep=before_sleep_log(logger, log_level=40),
)
def call_update_memory(
    prompt: str,
    model: str,
    parse_fn: StructuredParseFn,
    update_prompt_version: MemoryUpdatePromptVersion = "v2",
) -> UserMemoryContent | UserMemoryContentV3 | MemoryUpdatePatchV2_1 | MemoryUpdatePatchV2_2:
    response_model = (
        MemoryUpdatePatchV2_1 if update_prompt_version == "v2_1"
        else MemoryUpdatePatchV2_1 if update_prompt_version == "v2_3"
        else MemoryUpdatePatchV2_1 if update_prompt_version == "v2_4"
        else MemoryUpdatePatchV2_1 if update_prompt_version == "v2_5"
        else MemoryUpdatePatchV2_2 if update_prompt_version == "v2_2"
        else UserMemoryContent if update_prompt_version == "v2"
        else UserMemoryContentV3
    )
    return parse_fn(
        prompt,
        model,
        response_model,
        temperature=0.3,
        timeout=120,
        system_msg="You are an expert user behavior analyst.",
    )


def resolve_memory_update_prompt_version(
    memory_version: MemoryVersion,
    memory_update_prompt_version: MemoryUpdatePromptVersion,
) -> Literal["v2", "v2_1", "v2_2", "v2_3", "v2_4", "v2_5", "v3"]:
    if memory_update_prompt_version == "auto":
        return "v2" if memory_version == "v2" else "v3"
    if memory_version == "v2" and memory_update_prompt_version in {"v2", "v2_1", "v2_2", "v2_3", "v2_4", "v2_5"}:
        return memory_update_prompt_version
    if memory_version == "v3" and memory_update_prompt_version == "v3":
        return "v3"
    raise ValueError(
        f"Incompatible memory update prompt version: memory_version={memory_version}, "
        f"memory_update_prompt_version={memory_update_prompt_version}"
    )


def update_memory(
    memory: UserMemory | UserMemoryV3,
    session: SessionData,
    turn_predictions: list[dict],
    model: str,
    parse_fn: StructuredParseFn,
    use_oracle_labels: bool = False,
    memory_version: MemoryVersion = "v2",
    memory_update_prompt_version: MemoryUpdatePromptVersion = "auto",
) -> UserMemory | UserMemoryV3:
    """Update user memory after a session has been predicted."""
    resolved_update_version = resolve_memory_update_prompt_version(
        memory_version,
        memory_update_prompt_version,
    )
    if memory_version == "v2":
        assert isinstance(memory, UserMemory)
        if resolved_update_version == "v2_1":
            prompt = build_memory_update_prompt_v2_1(
                existing_memory=memory,
                new_session=session,
                turn_predictions=turn_predictions,
                use_oracle_labels=use_oracle_labels,
            )
            patch = call_update_memory(prompt, model, parse_fn=parse_fn, update_prompt_version="v2_1")
            assert isinstance(patch, MemoryUpdatePatchV2_1)
            return merge_memory_v2_1_patch(
                existing_memory=memory,
                patch=patch,
                turn_predictions=turn_predictions,
                use_oracle_labels=use_oracle_labels,
            )

        if resolved_update_version == "v2_2":
            prompt = build_memory_update_prompt_v2_2(
                existing_memory=memory,
                new_session=session,
                turn_predictions=turn_predictions,
                use_oracle_labels=use_oracle_labels,
            )
            patch = call_update_memory(prompt, model, parse_fn=parse_fn, update_prompt_version="v2_2")
            assert isinstance(patch, MemoryUpdatePatchV2_2)
            return merge_memory_v2_2_patch(
                existing_memory=memory,
                patch=patch,
                turn_predictions=turn_predictions,
                use_oracle_labels=use_oracle_labels,
            )

        if resolved_update_version == "v2_3":
            prompt = build_memory_update_prompt_v2_3(
                existing_memory=memory,
                new_session=session,
                turn_predictions=turn_predictions,
                use_oracle_labels=use_oracle_labels,
            )
            patch = call_update_memory(prompt, model, parse_fn=parse_fn, update_prompt_version="v2_3")
            assert isinstance(patch, MemoryUpdatePatchV2_1)
            return merge_memory_v2_3_patch(
                existing_memory=memory,
                patch=patch,
                turn_predictions=turn_predictions,
                use_oracle_labels=use_oracle_labels,
            )

        if resolved_update_version == "v2_4":
            prompt = build_memory_update_prompt_v2_4(
                existing_memory=memory,
                new_session=session,
                turn_predictions=turn_predictions,
                use_oracle_labels=use_oracle_labels,
            )
            patch = call_update_memory(prompt, model, parse_fn=parse_fn, update_prompt_version="v2_4")
            assert isinstance(patch, MemoryUpdatePatchV2_1)
            return merge_memory_v2_4_patch(
                existing_memory=memory,
                patch=patch,
                turn_predictions=turn_predictions,
                use_oracle_labels=use_oracle_labels,
            )

        if resolved_update_version == "v2_5":
            prompt = build_memory_update_prompt_v2_5(
                existing_memory=memory,
                new_session=session,
                turn_predictions=turn_predictions,
                use_oracle_labels=use_oracle_labels,
            )
            patch = call_update_memory(prompt, model, parse_fn=parse_fn, update_prompt_version="v2_5")
            assert isinstance(patch, MemoryUpdatePatchV2_1)
            return merge_memory_v2_5_patch(
                existing_memory=memory,
                patch=patch,
                turn_predictions=turn_predictions,
                use_oracle_labels=use_oracle_labels,
            )

        prompt = build_memory_update_prompt(
            existing_memory=memory,
            new_session=session,
            turn_predictions=turn_predictions,
            use_oracle_labels=use_oracle_labels,
        )
        content = call_update_memory(prompt, model, parse_fn=parse_fn, update_prompt_version="v2")
        assert isinstance(content, UserMemoryContent)
        return UserMemory.from_content(
            content,
            source_tasks=memory.source_tasks,
            n_history_sessions=memory.n_history_sessions + 1,
            n_history_turns=memory.n_history_turns + len(turn_predictions),
        )

    assert isinstance(memory, UserMemoryV3)
    prompt = build_memory_update_prompt_v3(
        existing_memory=memory,
        new_session=session,
        turn_predictions=turn_predictions,
        use_oracle_labels=use_oracle_labels,
    )
    content = call_update_memory(prompt, model, parse_fn=parse_fn, update_prompt_version="v3")
    assert isinstance(content, UserMemoryContentV3)
    return UserMemoryV3.from_content(
        content,
        source_tasks=memory.source_tasks,
        n_history_sessions=memory.n_history_sessions + 1,
        n_history_turns=memory.n_history_turns + len(turn_predictions),
    )
