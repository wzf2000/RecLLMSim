"""
用户记忆模块（User Memory） v2 / v3

Compatibility facade for the user-memory schemas, prompt builders, and update
merge helpers.  Existing imports such as `from lib.memory import UserMemory` and
`from lib.memory import build_turn_eval_prompt` remain valid while the
implementation lives in smaller modules.
"""

from __future__ import annotations

from .memory_schema import (
    MemoryUpdatePatchV2_1,
    MemoryUpdatePatchV2_2,
    RequirementPatchV2_2,
    ScoreDistribution,
    TaskObservation,
    TaskObservationPatchV2_2,
    UserMemory,
    UserMemoryContent,
    UserMemoryContentV3,
    UserMemoryV3,
    _build_v3_calibration_summary,
    _build_v3_evidence_notes,
    _low_score_evidence_level,
    _score_distribution_to_counts,
)
from .memory_formatting import (
    _MAX_EXAMPLES_PER_SCORE,
    _MAX_REPLY_CHARS,
    _MAX_SESSIONS_PROMPT,
    _collect_turns_by_score,
    _format_profile,
    _format_reason_json_rule,
    _format_reason_rule_block,
    _format_score_group,
    _select_sessions,
    _truncate,
)
from .memory_build_prompts import (
    build_memory_prompt,
    build_memory_prompt_v3,
)
from .memory_update_prompts import (
    _build_update_evidence_bundle,
    _collect_update_turns,
    _count_analysis_borderline_mentions,
    _dedupe_preserve_order,
    _format_update_examples,
    _is_generic_requirement,
    build_memory_update_prompt,
    build_memory_update_prompt_v2_1,
    build_memory_update_prompt_v2_2,
    build_memory_update_prompt_v2_3,
    build_memory_update_prompt_v2_4,
    build_memory_update_prompt_v2_5,
    build_memory_update_prompt_v3,
)
from .memory_update_merge import (
    merge_memory_v2_1_patch,
    merge_memory_v2_2_patch,
    merge_memory_v2_3_patch,
    merge_memory_v2_4_patch,
    merge_memory_v2_5_patch,
)
from .memory_eval_prompts import (
    _format_anchor_turns,
    build_turn_eval_history_prior_episodic_refine_prompt,
    build_turn_eval_fullscale_dsat_refinement_prompt,
    build_turn_eval_fullscale_sat_refinement_prompt,
    build_turn_eval_prompt,
    build_turn_eval_prompt_no_memory,
    build_turn_eval_refute_followup_prompt,
    build_turn_eval_v3_two_stage_dsat_refinement_prompt,
    build_turn_eval_v3_two_stage_gate_prompt,
    build_turn_eval_v3_two_stage_sat_refinement_prompt,
    build_turn_eval_v3_two_stage_v2_gate_followup_prompt,
    build_turn_eval_v3_two_stage_v2_gate_prompt,
)
