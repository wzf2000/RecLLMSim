# URS Task-Guarded Memgate Prompt Design

## Motivation

`urs_v2_calibrated_task_guarded` is the current best single-run URS evaluator.
The remaining risk is that summarized memory can still over-influence the 3/4 boundary, especially when history is thin, score buckets are nearly single-sided, or task-specific observations come from unrelated intents.

`urs_v2_calibrated_task_guarded_memgate` keeps the V1 task-aware calibration unchanged and adds a lightweight memory reliability gate.

## New Prompt Version

Added:

- `urs_prompt_version=urs_v2_calibrated_task_guarded_memgate`

Supported in:

- `scripts/collect_urs.sh`
- `scripts/score_urs_static_replay.sh`

## Design

The prompt computes memory confidence using the existing heuristic:

- `low`: fewer than 3 history sessions, or only one non-empty score bucket.
- `medium`: fewer than 5 sessions or only two score buckets.
- `high`: otherwise.

The added memory gate says:

- Current-session evidence always has priority.
- If the current answer clearly satisfies the core request and does not trigger the DSAT guard, memory cannot pull a score from `4/5` to `1/2/3`.
- If the current answer clearly fails the core request or triggers the DSAT guard, memory cannot pull a score from `1/2/3` to `4/5`.
- Low-confidence memory cannot change the SAT/DSAT boundary; it can only adjust inside `4/5` or inside `1/2/3`.
- Medium-confidence memory can affect the 3/4 boundary only when the preference is directly relevant to the current intent and request.
- High-confidence memory can adjust the boundary, but still cannot override current-session evidence or the task-aware guard.
- Generic preferences such as "detailed", "structured", or "actionable" cannot by themselves change the 3/4 boundary.

The rating steps are changed to:

1. Identify the current core request and hard constraints.
2. Make a preliminary SAT/DSAT decision using only current-session evidence and task-aware guard.
3. Evaluate memory confidence and relevance.
4. Apply memory only if the reliability gate allows crossing the 3/4 boundary.
5. Decide 4/5 or 1/2/3 within the final side.

## Commands

Memory version:

```bash
cd detection
model=Qwen/Qwen3-8B \
vllm_base_url=http://localhost:8000/v1 \
vllm_api_key=EMPTY \
memory_update_mode=none \
urs_prompt_version=urs_v2_calibrated_task_guarded_memgate \
max_workers=4 \
output_jsonl=outputs/urs/Qwen_Qwen3-8B_test_none_urs_v2_calibrated_task_guarded_memgate.jsonl \
bash scripts/collect_urs.sh
```

This version is intended for memory mode.
No-memory mode is not expected to be useful because the gate has no memory evidence to control.

## Expected Comparison

Compare against memory `urs_v2_calibrated_task_guarded`.

Desired behavior:

- Preserve the V1 retrieval/professional/advice DSAT gains.
- Reduce high-gold false-DSAT cases caused by unrelated or weak memory.
- Keep predicted SAT/DSAT distribution close to gold.
- Avoid the over-SAT drift seen in `task_guarded_v2`.
