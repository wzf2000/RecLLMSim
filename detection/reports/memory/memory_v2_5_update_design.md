# Memory V2.5 Update Design

## Motivation

`v2.4` was intended to protect the 3/4 boundary by gating verbal boundary
updates, but the 20-user subset showed the opposite behavior:

- predicted SAT increased to the highest level among compared variants
- `F1-DSAT` dropped sharply
- false SAT became worse

This suggests that SAT drift is not mainly caused by the verbal boundary fields.
The more likely sources are:

- `scoring_style` being rewritten toward a more lenient prior
- upward movement of `avg_satisfaction_score` from noisy predicted labels
- generic requirements and task observations being interpreted as easy-to-satisfy preferences

`v2.5` therefore starts again from `v2.1`, but constrains SAT-drift sources more
directly.

## Design

`memory_update_prompt_version=v2_5` keeps the v2.1-style update format:

- raw turn-level evidence grouped by predicted score
- same `MemoryUpdatePatchV2_1` output schema
- no v2.2 evidence bundle
- no v2.3 boundary summary

The merge behavior changes in three ways:

1. **Freeze `scoring_style` in non-oracle update**
   - For `per_session`, the patch cannot update `scoring_style`.
   - For `per_session_oracle`, `scoring_style` can still update.

2. **Dampen upward average-score drift in non-oracle update**
   - If the code-side updated average score moves downward, accept it.
   - If it moves upward from predicted labels, only apply 25% of the upward
     shift.
   - This keeps non-oracle updates from quickly raising the user's satisfaction
     prior due to SAT-biased predictions.

3. **Filter generic requirements**
   - Requirements containing broad fragments such as "详细", "具体", "清晰",
     "结构化", "实用", "全面", "完整", "分点", "有条理" are not merged.

Other fields keep v2.1 behavior:

- `three_vs_four_distinction`
- `four_vs_five_distinction`
- `preferred_response_format`
- `task_specific_observations`

This makes v2.5 a targeted SAT-drift control rather than another boundary-text
experiment.

## Expected Behavior

Compared with v2.1, v2.5 should:

- reduce SAT-heavy drift
- reduce false SAT
- recover some `F1-DSAT`
- hopefully preserve most of v2.1's MAE gain

Potential failure modes:

- If MAE worsens but DSAT improves, the average-score damping is too strong.
- If SAT drift remains, the source is likely task observations or turn-eval
  prompt consumption rather than scoring style or average score.
- If all metrics are close to v2.1, the frozen/damped fields were not the main
  active mechanism.

## Suggested Experiment

Small subset first:

```bash
cd detection

model=Qwen/Qwen3-8B \
memory_update_mode=per_session \
memory_update_prompt_version=v2_5 \
memory_version=v2 \
turn_eval_prompt_version=v2 \
limit_users=20 \
max_workers=4 \
output_jsonl=outputs/personalized/Qwen_Qwen3-8B_test_per_session_updv2_5_u20.jsonl \
bash scripts/collect_personalized_vllm.sh
```

Evaluation should use the same `sample_id` subset as v2.5 and compare against:

- `none`
- `v2.1`
- `v2.3`
- `v2.4`
- `oracle_v2.1`
