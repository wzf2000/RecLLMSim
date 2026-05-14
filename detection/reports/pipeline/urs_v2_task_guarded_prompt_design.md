# URS V2 Task-Guarded Prompt Design

## Motivation

Recent URS experiments showed that language-specific prompting is not the main bottleneck.
The larger issue is unstable 3/4 boundary behavior:

- Some prompts over-penalize short or lightly truncated high-score answers.
- Some prompts over-credit answers that are topically relevant but miss a hard user constraint.
- URS memory can overgeneralize preferences from unrelated intents.

`urs_v2_calibrated_task_guarded` is added as a new prompt version that keeps the original calibrated URS rating scale while adding dataset-specific task guards.
Existing prompt versions are preserved.

## New Prompt Version

Added:

- `urs_prompt_version=urs_v2_calibrated_task_guarded`

Supported in:

- `scripts/collect_urs.sh`
- `scripts/score_urs_static_replay.sh`

## Design

The prompt adds three components.

### 1. Task-Aware Guard

The task slug is parsed from `task_context`, e.g. `[retrieval] ...`.
Each URS intent receives a short calibration rule:

- `retrieval`: prioritize factual correctness and directness; do not require extra structure for short factual answers; but hard constraints such as source/product/price/manufacturing-origin must be satisfied before assigning `4`.
- `professional`: require a correct and executable solution; generic advice or missing key steps should usually stay below `4`.
- `advice`: require advice to be situation-aware and actionable.
- `creative`: require user-specified topic, characters, style, format, and key plot elements.
- `text`: require a usable target text, not just writing advice.
- `leisure`: require recommendations or interaction aligned with user preferences.
- `other`: identify the concrete user goal first.

### 2. DSAT Guard

The prompt explicitly says that the following should usually not receive `4` or `5`:

- The core request is unanswered or the answer targets the wrong task.
- A requested artifact/source/product/route/price/constraint is missing.
- The answer is only generic background, empty advice, disclaimer, or "not found" without a useful alternative.
- The answer contains key factual errors, hallucinations, inappropriate refusal, or poor handling of sensitive topics.
- The user would need to ask again to complete the main task.

### 3. Memory Weak-Use Constraint

Memory is treated as weak evidence:

- Current-session evidence is primary.
- Memory only affects the score when directly relevant to the current intent and request.
- Generic preferences such as "detailed", "structured", or "actionable" should not downgrade a correct useful short answer from `4` to `3`.
- Preferences from unrelated intents should not be imported.

## Recommended Commands

Memory version:

```bash
cd detection
model=Qwen/Qwen3-8B \
vllm_base_url=http://localhost:8000/v1 \
vllm_api_key=EMPTY \
memory_update_mode=none \
urs_prompt_version=urs_v2_calibrated_task_guarded \
max_workers=4 \
output_jsonl=outputs/urs/Qwen_Qwen3-8B_test_none_urs_v2_calibrated_task_guarded.jsonl \
bash scripts/collect_urs.sh
```

No-memory ablation:

```bash
cd detection
model=Qwen/Qwen3-8B \
vllm_base_url=http://localhost:8000/v1 \
vllm_api_key=EMPTY \
no_memory=1 \
urs_prompt_version=urs_v2_calibrated_task_guarded \
max_workers=4 \
output_jsonl=outputs/urs/Qwen_Qwen3-8B_test_no_memory_urs_v2_calibrated_task_guarded.jsonl \
bash scripts/collect_urs.sh
```

Static replay scoring:

```bash
cd detection
judge_model=Qwen/Qwen3-8B \
judge_vllm_base_url=http://localhost:8000/v1 \
judge_vllm_api_key=EMPTY \
memory_update_mode=none \
urs_prompt_version=urs_v2_calibrated_task_guarded \
input_jsonl=outputs/urs_static_replay/<responses>.jsonl \
output_jsonl=outputs/urs_static_replay/<scores>.jsonl \
bash scripts/score_urs_static_replay.sh
```

## Expected Evaluation Focus

Compare against:

- `urs_v2_calibrated`
- `urs_v2_memory_guarded`
- `urs_v2_calibrated_langaware_v2`

Primary metrics:

- Overall Pearson / Spearman / QWK.
- Boundary F1-DSAT and false-SAT rate.
- Per-intent breakdown, especially `retrieval`, `professional`, and `text`.

The desired behavior is not simply higher DSAT prediction.
The goal is to recover DSAT cases with missing hard constraints while preserving high-score factual short-answer cases.
