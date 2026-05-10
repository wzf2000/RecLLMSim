# Mixed Memory/Predictor Model Results

Date: 2026-05-10

This note compares recent runs where the memory-building model and turn-level
predictor model are decoupled. All subset results below use the same 20-user test
subset with 1594 evaluated turns, `memory_version=v2`, `memory_update_mode=none`,
and `turn_eval_prompt_version=v2`.

## Compared Runs

| short name | result file | predictor | memory model |
|---|---|---|---|
| q36_same_raw | `detection/outputs/personalized/Qwen_Qwen3.6-35B-A3B_test_none_v2_u20.jsonl` | Qwen3.6-35B-A3B | Qwen3.6-35B-A3B |
| q36_pred_q8_mem | `detection/outputs/personalized/Qwen36_judge_Qwen8B_memory_v2_none_u20.jsonl` | Qwen3.6-35B-A3B | Qwen3-8B |
| q8_pred_q36_mem | `detection/outputs/personalized/Qwen8B_judge_Qwen36_memory_v2_none_u20.jsonl` | Qwen3-8B | Qwen3.6-35B-A3B |
| q8_pred_gpt54_mem | `detection/outputs/personalized/qwen3_8b_v2_none_mem_gpt54mini_u20.jsonl` | Qwen3-8B | gpt-5.4-mini-2026-03-17 |
| q8_pred_gpt54_mem_full | `detection/outputs/personalized/qwen3_8b_v2_none_mem_gpt54mini.jsonl` | Qwen3-8B | gpt-5.4-mini-2026-03-17 |

`q8_pred_gpt54_mem_full` is a 90-user/full run with 6474 turns. It is not directly
comparable with the 20-user subset, but it is useful for checking whether the
subset trend is stable.

## Raw 20-User Results

| run | MAE ↓ | Pearson ↑ | Spearman ↑ | QWK ↑ | Boundary Acc ↑ | F1-DSAT ↑ | False SAT ↓ | False DSAT ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| q36_same_raw | 0.7842 | 0.3411 | 0.3335 | 0.3083 | 0.7120 | 0.4168 | 0.4364 | 0.2548 |
| q36_pred_q8_mem | 0.7171 | 0.3066 | 0.3109 | 0.2857 | 0.7553 | 0.3689 | 0.6082 | 0.1635 |
| q8_pred_q36_mem | 0.6744 | 0.3442 | 0.3447 | 0.3276 | 0.7836 | 0.3979 | 0.6082 | 0.1289 |
| q8_pred_gpt54_mem | 0.6386 | 0.3286 | 0.3351 | 0.2998 | 0.8074 | 0.2742 | 0.8007 | 0.0568 |

Raw observations:

- `q8_pred_q36_mem` is the best balanced raw mixed run. It has the strongest raw
  QWK among mixed runs, the best raw Pearson/Spearman, and much better DSAT
  handling than `q8_pred_gpt54_mem`.
- `q8_pred_gpt54_mem` has the lowest raw MAE and highest boundary accuracy, but
  it over-predicts satisfaction. Its predicted SAT count is 1462/1594, and
  False SAT reaches 0.8007, so it misses most dissatisfied turns.
- `q36_pred_q8_mem` improves MAE over `q36_same_raw`, but weakens ranking/QWK and
  boundary DSAT detection. This suggests the weaker memory model is a bottleneck
  when paired with the stronger predictor.
- `q36_same_raw` has the highest DSAT recall/F1 among raw configurations but is
  too pessimistic overall, resulting in high MAE and low boundary accuracy.

## Calibrated 20-User Results

CDF and mean-shift calibration were run for the three mixed configurations. The
existing Qwen3.6 same-model CDF result is included as the current same-model
reference.

| run | MAE ↓ | Pearson ↑ | Spearman ↑ | QWK ↑ | Boundary Acc ↑ | F1-DSAT ↑ | False SAT ↓ | False DSAT ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| q36_same_cdf | 0.6405 | 0.3977 | 0.4238 | 0.3966 | 0.7923 | 0.4121 | 0.6014 | 0.1197 |
| q36pred_q8mem_cdf | 0.6518 | 0.3750 | 0.3930 | 0.3740 | 0.7911 | 0.4085 | 0.6048 | 0.1205 |
| q8pred_q36mem_cdf | 0.6694 | 0.3596 | 0.3854 | 0.3586 | 0.7861 | 0.3943 | 0.6186 | 0.1236 |
| q8pred_gptmem_cdf | 0.6619 | 0.3640 | 0.3815 | 0.3630 | 0.7911 | 0.4085 | 0.6048 | 0.1205 |
| q36pred_q8mem_ms | 0.6468 | 0.3757 | 0.3951 | 0.3642 | 0.7829 | 0.3930 | 0.6151 | 0.1282 |
| q8pred_q36mem_ms | 0.6380 | 0.3589 | 0.3732 | 0.3438 | 0.8024 | 0.4023 | 0.6357 | 0.0998 |
| q8pred_gptmem_ms | 0.6449 | 0.3149 | 0.3290 | 0.2878 | 0.8005 | 0.2673 | 0.8007 | 0.0652 |

Calibrated observations:

- The strongest overall configuration remains `q36_same_cdf`. It has the best
  Pearson, Spearman, QWK, and a strong boundary F1-DSAT.
- The closest mixed competitor is `q36pred_q8mem_cdf`. It nearly matches boundary
  metrics, but it is still behind `q36_same_cdf` on all global/ranking metrics.
- `q8pred_q36mem_ms` has the best calibrated MAE and boundary accuracy, but its
  QWK and rank correlations are weaker than the CDF variants. It is useful if
  the priority is absolute error, not ranking/personalization signal.
- `q8pred_gptmem` remains satisfaction-biased after calibration. CDF reduces the
  problem substantially, but it still does not beat `q36_same_cdf`; mean-shift
  leaves the DSAT issue mostly unresolved.

## Full-Run Check

For `q8_pred_gpt54_mem_full` on 90 users / 6474 turns:

| run | MAE ↓ | Pearson ↑ | Spearman ↑ | QWK ↑ | Boundary Acc ↑ | F1-DSAT ↑ | False SAT ↓ | False DSAT ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| q8_pred_gpt54_mem_full | 0.6069 | 0.3480 | 0.3380 | 0.3307 | 0.8143 | 0.2987 | 0.7687 | 0.0654 |

This confirms the subset behavior: the GPT-memory + Qwen3-8B predictor setup is
strong on MAE and overall SAT accuracy, but it under-detects dissatisfied turns.
It is therefore risky as a benchmark judge if DSAT sensitivity matters.

## Recommendation

For the current pipeline, use `Qwen3.6-35B-A3B` as both memory builder and
predictor, with CDF calibration, when compute permits. It is the most defensible
choice across global metrics, user-aware/ranking-oriented behavior, and 3/4
boundary quality.

If cost is a constraint, the best mixed alternative is `Qwen3-8B predictor +
Qwen3.6-35B-A3B memory` in raw form, or with mean-shift when MAE is the main
target. However, it does not surpass the same-model Qwen3.6 + CDF setup.

