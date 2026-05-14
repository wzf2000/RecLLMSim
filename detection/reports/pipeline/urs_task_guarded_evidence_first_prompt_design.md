# URS Task-Guarded Evidence-First Prompt Design

## Motivation

`urs_v2_calibrated_task_guarded` is currently the strongest URS predictor variant among the recent prompt trials.
`urs_v2_calibrated_task_guarded_memgate` did not improve over it: the prompt-side memory reliability gate improved a few tasks such as `text` and `leisure`, but weakened the useful gains on `advice`, `professional`, `retrieval`, and `creative`.

The failure mode suggests that broad natural-language memory gating makes the evaluator too conservative and distracts it from direct task-local evidence.
The next version therefore keeps the task-aware evidence checks from `task_guarded_v1`, but changes memory usage to an evidence-first policy.

## New Prompt Version

Prompt version:

- `urs_v2_calibrated_task_guarded_evidence_first`

Implemented in:

- `detection/lib/urs_memory.py`
- `detection/trace/urs/cli.py`
- `detection/trace/score_urs_static_replay.py`

## Design

The new prompt keeps the following parts from `task_guarded_v1`:

- URS session-level calibration.
- Task-aware guard by intent.
- DSAT guard for hard failure cases.
- Reason-label consistency rule: `classification >= 4` must use `满意`.

The main change is memory usage:

- The memory block is renamed from personalized scoring standard to historical memory.
- The prompt explicitly says memory is weak evidence, not the scoring rubric.
- The model must first produce a provisional score using only current-session evidence and task-aware guard.
- Memory is checked only after the provisional score is formed.
- By default, memory can only adjust within the same satisfaction side: `4 <-> 5` or `1 <-> 2 <-> 3`.
- Crossing the `3/4` boundary requires both directly relevant memory evidence and supporting current-session evidence.
- Generic memory preferences such as "detailed", "structured", or "actionable" cannot change the SAT/DSAT decision by themselves.

## Expected Behavior

This variant is intended to preserve the strong global and boundary performance of `task_guarded_v1`, while reducing cases where unrelated or weak memory pulls a correct answer below 4.

It is expected to be closer to `task_guarded_v1` than to `task_guarded_memgate`.
If it improves over `task_guarded_v1`, the likely gains should appear in `text`, `leisure`, or other cases where memory over-penalization was previously suspected.
If it underperforms, the result would indicate that `task_guarded_v1` already uses memory weakly enough and further prompt-side memory constraints are unnecessary.

## Suggested Run

Use local Qwen3-8B:

```bash
model=Qwen/Qwen3-8B \
vllm_base_url=http://localhost:8001/v1 \
vllm_api_key=EMPTY \
memory_update_mode=none \
urs_prompt_version=urs_v2_calibrated_task_guarded_evidence_first \
output_jsonl=outputs/urs/Qwen_Qwen3-8B_test_none_urs_v2_calibrated_task_guarded_evidence_first.jsonl \
max_workers=4 \
bash scripts/collect_urs.sh
```

Evaluate against previous variants:

```bash
result_files="cal=outputs/urs/Qwen_Qwen3-8B_test_none_urs_v2_calibrated.jsonl task_v1=outputs/urs/Qwen_Qwen3-8B_test_none_urs_v2_calibrated_task_guarded.jsonl memgate=outputs/urs/Qwen_Qwen3-8B_test_none_urs_v2_calibrated_task_guarded_memgate.jsonl evidence_first=outputs/urs/Qwen_Qwen3-8B_test_none_urs_v2_calibrated_task_guarded_evidence_first.jsonl" \
output_json=outputs/urs/qwen3_8b_urs_task_guarded_evidence_first_comparison.json \
bash scripts/eval_urs_predictor.sh
```

