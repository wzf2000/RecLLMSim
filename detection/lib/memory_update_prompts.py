"""Compatibility facade for user-memory update prompt builders."""

from __future__ import annotations

from .memory_update_common import (
    _build_update_evidence_bundle,
    _collect_update_turns,
    _count_analysis_borderline_mentions,
    _dedupe_preserve_order,
    _format_update_examples,
    _is_generic_requirement,
)
from .memory_update_patch_prompts import (
    build_memory_update_prompt_v2_1,
    build_memory_update_prompt_v2_2,
    build_memory_update_prompt_v2_3,
    build_memory_update_prompt_v2_4,
    build_memory_update_prompt_v2_5,
)
from .memory_update_v2_prompts import build_memory_update_prompt
from .memory_update_v3_prompts import build_memory_update_prompt_v3

__all__ = [
    "_build_update_evidence_bundle",
    "_collect_update_turns",
    "_count_analysis_borderline_mentions",
    "_dedupe_preserve_order",
    "_format_update_examples",
    "_is_generic_requirement",
    "build_memory_update_prompt",
    "build_memory_update_prompt_v2_1",
    "build_memory_update_prompt_v2_2",
    "build_memory_update_prompt_v2_3",
    "build_memory_update_prompt_v2_4",
    "build_memory_update_prompt_v2_5",
    "build_memory_update_prompt_v3",
]
