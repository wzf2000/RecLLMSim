# URS V2 Memory-Guarded Prompt Design

## Motivation

Case-level diagnosis showed that URS memory can be harmful when:

- history is thin,
- history contains only one score bucket,
- intent-specific requirements are overgeneralized,
- memory prior overrides direct target-session evidence.

The previous `urs_v2_calibrated` prompt improves global score calibration but
still trusts memory too strongly. `urs_v2_memory_guarded` keeps the calibrated
score scale and adds explicit memory usage constraints.

## New Prompt Version

- `urs_prompt_version=urs_v2_memory_guarded`

This version applies only when memory is enabled. For no-memory runs, it is
equivalent to `urs_v2_calibrated` because there is no memory block to guard.

## Added Rules

The scoring prompt now computes a simple memory confidence label:

- `low`: fewer than 3 history sessions, or only one non-empty score bucket.
- `medium`: limited history or only two score buckets.
- `high`: at least 5 sessions and at least 3 score buckets.

The prompt tells the judge:

- Memory is a weak prior, not a substitute for target-session quality.
- Current session evidence overrides memory when they conflict.
- Do not transfer concrete requirements across unrelated intents.
- All-positive or all-neutral histories should not automatically inflate or
  deflate the target score.
- If memory conflicts with target-session evidence, mention the conflict in
  analysis and prioritize the current session.

## Expected Effect

Compared with `urs_v2_calibrated`, this version should:

- preserve improved global calibration,
- reduce obvious memory-induced errors,
- improve DSAT precision/recall balance,
- improve user-aware metrics if harmful over-personalization is reduced.

## Recommended Command

```bash
cd detection
model=Qwen/Qwen3-8B \
vllm_base_url=http://localhost:8000/v1 \
vllm_api_key=EMPTY \
memory_update_mode=none \
urs_prompt_version=urs_v2_memory_guarded \
max_workers=4 \
output_jsonl=outputs/urs/Qwen_Qwen3-8B_test_none_urs_v2_memory_guarded.jsonl \
bash scripts/collect_urs.sh
```

Evaluate with:

```bash
result_file=outputs/urs/Qwen_Qwen3-8B_test_none_urs_v2_memory_guarded.jsonl \
output_json=outputs/urs/Qwen_Qwen3-8B_test_none_urs_v2_memory_guarded_eval.json \
bash scripts/eval_urs_predictor.sh
```
