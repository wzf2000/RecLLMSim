# memory_v2.1 update design

## Motivation

`detection/reports/memory/qwen_v2_update_diagnosis.md` shows that Qwen3 `memory_v2` update is not a no-op, but it is ineffective because:

- it changes around 18% of turns
- the changed turns are nearly evenly split between better and worse
- the update prompt rewrites the whole memory at once
- the actual supervision is too coarse to reliably sharpen `3/4` or `4/5` boundaries

So `v2.1 update` is designed to change the **update interface**, while keeping the original `v2` memory schema and initial cache unchanged.

## Key idea

`v2.1` keeps:

- initial memory building = `v2`
- turn evaluation prompt = unchanged unless the user chooses otherwise

Only the update stage changes.

The main design shift is:

- **statistics are updated deterministically in code**
- **verbal fields are updated by patch, not by full-memory rewrite**

## What changes in v2.1

### 1. Deterministic statistics update

The following fields are no longer entrusted to the LLM during update:

- `avg_satisfaction_score`
- `score_distribution`

They are updated directly from:

- oracle labels if `per_session_oracle`
- model predictions otherwise

This removes one major source of pointless full-memory regeneration.

### 2. Patch-style verbal update

The LLM now outputs a structured patch instead of a full `UserMemoryContent`.

Patch fields include:

- whether to update `scoring_style`
- whether to update `three_vs_four_distinction`
- whether to update `four_vs_five_distinction`
- which `user_specific_requirements` to add
- whether to update `preferred_response_format`
- which `task_specific_observations` to add or replace

This means the update stage can:

- leave most fields unchanged
- modify only the specific field supported by new evidence

### 3. Evidence is grouped by boundary role

The prompt no longer just dumps the whole new session.

Instead, it provides:

- session-level score distribution
- `<=3` evidence block
- `4` evidence block
- `5` evidence block

This is meant to make the model reason in terms of:

- `3/4` evidence
- `4/5` evidence
- low-score evidence

instead of diffuse full-session paraphrasing.

### 4. Conservative but targeted field update

The prompt explicitly requires:

- `three_vs_four_distinction` may only be changed when there is clear `<=3` vs `4` evidence
- `four_vs_five_distinction` may only be changed when there is clear `4` vs `5` evidence
- `user_specific_requirements` may only add truly discriminative user-specific constraints
- `preferred_response_format` should stay unchanged unless the session reveals a stable new format preference

This keeps the useful conservatism of `v2`, but makes it field-aware rather than whole-memory conservative.

## Cache reuse

Yes, the original `v2` memory cache can be reused.

Reason:

- `v2.1` does **not** change initial memory building
- it only changes **Phase 3: update**
- so the cache key for initial user memory remains the same `v2` cache file

Important caveat:

- updated memory states inside a run are still not persisted back to the cache directory
- cache reuse here means reusing the initial `build_user_memory()` output, not reusing intermediate updated states from previous runs

## New runtime parameter

A new parameter is added:

- `memory_update_prompt_version`

Choices:

- `auto`
- `v2`
- `v2_1`
- `v3`

Recommended for Qwen3 `memory_v2` update experiments:

- `memory_version=v2`
- `memory_update_prompt_version=v2_1`

## Recommended experiment

```bash
cd detection
model=Qwen/Qwen3-8B \
memory_update_mode=per_session_oracle \
memory_version=v2 \
memory_update_prompt_version=v2_1 \
output_jsonl=outputs/personalized/Qwen_Qwen3-8B_test_per_session_oracle_updv2_1.jsonl \
bash scripts/collect_personalized_vllm.sh
```

Also recommended:

```bash
cd detection
model=Qwen/Qwen3-8B \
memory_update_mode=per_session \
memory_version=v2 \
memory_update_prompt_version=v2_1 \
output_jsonl=outputs/personalized/Qwen_Qwen3-8B_test_per_session_updv2_1.jsonl \
bash scripts/collect_personalized_vllm.sh
```

## What to verify after running

The first questions to check are:

1. Does `v2.1` improve net gain over `none` for `per_session_oracle`?
2. Does it reduce the “better and worse nearly cancel out” pattern?
3. Does it improve `3/4` boundary behavior without heavily damaging the full 1-5 metrics?
4. Are the update-induced changes still mostly `3↔4` / `4↔5` nudges, or do they become more targeted?
