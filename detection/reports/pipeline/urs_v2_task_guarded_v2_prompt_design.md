# URS V2 Task-Guarded V2 Prompt Design

## Motivation

`urs_v2_calibrated_task_guarded` is the strongest single URS evaluator so far, but its task-level breakdown shows regressions in `text` and some over-strict cases in `professional` and `advice`.
The main issue is that the guard sometimes treats simple but accepted answers as insufficient because they do not include extended structure or comprehensive details.

`urs_v2_calibrated_task_guarded_v2` keeps the successful parts of V1 while softening the over-strict task rules.

## Changes From V1

Unchanged:

- `retrieval` keeps the hard-constraint guard for source/product/route/price/manufacturing-origin/factual constraints.
- Generic DSAT guard remains.
- Memory remains weak evidence and cannot be imported across unrelated intents.

Softened:

- `professional`: simple concept, step, or professional questions can receive `4` if the core point is answered correctly; do not require all extended risks or advanced details.
- `advice`: simple requests can receive `4` with clear actionable advice; do not require exhaustive alternatives.
- `text`: distinguish full-text generation from local rewrite/explanation/advice; short or template-like but adequate text should not be lowered to `3`.

## Commands

Memory version:

```bash
cd detection
model=Qwen/Qwen3-8B \
vllm_base_url=http://localhost:8000/v1 \
vllm_api_key=EMPTY \
memory_update_mode=none \
urs_prompt_version=urs_v2_calibrated_task_guarded_v2 \
max_workers=4 \
output_jsonl=outputs/urs/Qwen_Qwen3-8B_test_none_urs_v2_calibrated_task_guarded_v2.jsonl \
bash scripts/collect_urs.sh
```

No-memory ablation:

```bash
cd detection
model=Qwen/Qwen3-8B \
vllm_base_url=http://localhost:8000/v1 \
vllm_api_key=EMPTY \
no_memory=1 \
urs_prompt_version=urs_v2_calibrated_task_guarded_v2 \
max_workers=4 \
output_jsonl=outputs/urs/Qwen_Qwen3-8B_test_no_memory_urs_v2_calibrated_task_guarded_v2.jsonl \
bash scripts/collect_urs.sh
```

## Expected Comparison

Compare primarily against memory `urs_v2_calibrated_task_guarded`.

Desired changes:

- Preserve or improve retrieval F1-DSAT.
- Recover `text` MAE/QWK and text F1-DSAT.
- Reduce high-gold over-penalization in `professional` and `advice`.
- Avoid returning to the over-SAT behavior seen in `langaware_v2`.
