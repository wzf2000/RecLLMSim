# Memory V2.4 Update Design

## Motivation

`per_session + v2.1` remains the best non-oracle update variant for overall
1-5 prediction:

- it improves MAE/RMSE over `Qwen none`
- it keeps the raw turn-level evidence that appears useful
- it avoids the large regression caused by the v2.2 evidence bundle

However, `v2.1` is SAT-leaning on the 3/4 boundary:

- it improves boundary accuracy and F1-SAT
- but it does not improve F1-DSAT over `none`
- its false SAT rate is higher than `none`

`v2.3` showed that a boundary-aware variant can recover some DSAT signal, but
it over-corrected and worsened full 1-5 metrics. Therefore v2.4 is designed as
a lighter modification based on v2.1 rather than v2.3.

## Design

`memory_update_prompt_version=v2_4` keeps the v2.1 structure:

- raw turn-level evidence grouped by predicted score
- same `MemoryUpdatePatchV2_1` schema
- code-side update of `avg_satisfaction_score` and `score_distribution`
- no v2.2-style structured evidence bundle
- no v2.3-style extra boundary summary block

The only added constraints are:

1. `three_vs_four_distinction` should only be updated when the new session has
   both 3-point and 4-point evidence.
2. `four_vs_five_distinction` should only be updated when the new session has
   both 4-point and 5-point evidence.
3. generic requirements such as "更详细", "更具体", "更清晰", "更结构化",
   and similar broad preferences are filtered during merge.
4. the prompt explicitly warns against systematically lowering scores only to
   improve dissatisfied-case detection.

## Implementation

Added functions:

- `build_memory_update_prompt_v2_4(...)`
- `merge_memory_v2_4_patch(...)`

Updated entry points:

- `detection/trace/collect_personalized.py`
- `detection/scripts/collect_personalized.sh`
- `detection/scripts/collect_personalized_vllm.sh`

`v2_4` uses the same response model as v2.1:

- `MemoryUpdatePatchV2_1`

## Expected Behavior

The desired result is not to maximize DSAT at all costs. The target is:

- preserve most of the v2.1 MAE/RMSE gain
- reduce v2.1's SAT-heavy boundary drift
- avoid v2.3's harmful `4 -> 3` over-correction

Expected success pattern:

- MAE close to v2.1 and better than none
- boundary F1-DSAT closer to none/v2.3 than v2.1
- false SAT lower than v2.1
- QWK/correlation not worse than v2.1 by much

Expected failure pattern:

- if MAE regresses toward v2.3, the boundary guard is still too influential
- if F1-DSAT remains at v2.1 level, the change is too weak to affect boundary

## Suggested Experiment

Small subset first:

```bash
cd detection

model=Qwen/Qwen3-8B \
memory_update_mode=per_session \
memory_update_prompt_version=v2_4 \
memory_version=v2 \
turn_eval_prompt_version=v2 \
limit_users=20 \
max_workers=4 \
output_jsonl=outputs/personalized/Qwen_Qwen3-8B_test_per_session_updv2_4_u20.jsonl \
bash scripts/collect_personalized_vllm.sh
```

Full run:

```bash
cd detection

model=Qwen/Qwen3-8B \
memory_update_mode=per_session \
memory_update_prompt_version=v2_4 \
memory_version=v2 \
turn_eval_prompt_version=v2 \
max_workers=4 \
output_jsonl=outputs/personalized/Qwen_Qwen3-8B_test_per_session_updv2_4.jsonl \
bash scripts/collect_personalized_vllm.sh
```

Evaluation:

```bash
cd detection

result_files="none=outputs/personalized/Qwen_Qwen3-8B_test_none.jsonl \
v2_1=outputs/personalized/Qwen_Qwen3-8B_test_per_session_updv2_1.jsonl \
v2_3=outputs/personalized/Qwen_Qwen3-8B_test_per_session_updv2_3.jsonl \
v2_4=outputs/personalized/Qwen_Qwen3-8B_test_per_session_updv2_4.jsonl" \
output_json=outputs/personalized/qwen_v2_4_update_compare.json \
bash scripts/eval_personalized.sh
```
