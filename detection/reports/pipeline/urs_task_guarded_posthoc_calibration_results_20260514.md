# URS Task-Guarded Post-Hoc Calibration Results

## Setup

This report evaluates post-hoc calibration on the current strongest URS predictor:

- Base prediction: `outputs/urs/Qwen_Qwen3-8B_test_none_urs_v2_calibrated_task_guarded.jsonl`
- Memory cache: `outputs/urs/memory_cache`
- Calibration methods: `mean_shift` and `cdf`
- `min_history_turns=1`

Generated outputs:

- `outputs/urs/Qwen_Qwen3-8B_test_none_urs_v2_calibrated_task_guarded_calMS.jsonl`
- `outputs/urs/Qwen_Qwen3-8B_test_none_urs_v2_calibrated_task_guarded_calCDF.jsonl`
- `outputs/urs/qwen3_8b_urs_task_guarded_posthoc_comparison.json`

Calibration coverage:

- Total blocks: 339
- Calibrated blocks: 145
- Fallback identity blocks: 194
- Fallback reason: small block

The ordinary calibration script groups by `(user, target_task, model)`.
On URS, many user-task blocks contain only one test sample, so a large fraction of samples cannot be rank-calibrated within block.

## Overall Results

| version | MAE | RMSE | Pearson | Spearman | QWK | Boundary Acc | F1-SAT | F1-DSAT | False-SAT | False-DSAT |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| task_guarded_v1 | 0.6952 | 1.0119 | 0.2760 | 0.2883 | 0.2424 | 0.6627 | 0.7398 | 0.5207 | 0.4780 | 0.2612 |
| mean_shift | 0.7466 | 1.0517 | 0.2820 | 0.2953 | 0.2669 | 0.6644 | 0.7487 | 0.4948 | 0.5317 | 0.2296 |
| cdf | 0.8099 | 1.1355 | 0.2603 | 0.2698 | 0.2571 | 0.6318 | 0.7122 | 0.4893 | 0.4976 | 0.2982 |
| evidence_first | 0.7209 | 1.0245 | 0.2473 | 0.2483 | 0.2143 | 0.6438 | 0.7219 | 0.5048 | 0.4829 | 0.2876 |

## Interpretation

Standard post-hoc calibration is not useful as a direct replacement for `task_guarded_v1`.

`mean_shift` slightly improves correlation and QWK, but it worsens MAE and reduces F1-DSAT.
It also increases False-SAT, meaning more truly dissatisfied sessions are pushed into the satisfied side.

`cdf` is clearly worse overall.
It increases MAE substantially and reduces boundary accuracy and F1-DSAT.

The likely reason is that URS has sparse per-user-task blocks.
The current calibration method was originally designed for denser turn-level blocks, where rank-to-CDF mapping has multiple predictions per user/task.
On URS, many blocks contain only one record, and even calibrated blocks have weak local ranking information.

## Recommendation

Do not use `mean_shift` or `cdf` as the default URS evaluator post-processing.

If post-hoc adjustment is still needed, the next attempt should be a boundary-specific arbitration rule rather than full 1-5 distribution remapping.
For example, keep `task_guarded_v1` as the base score and only use `evidence_first` when:

- the two variants disagree across the 3/4 boundary,
- the base prediction is near the boundary,
- and the task is one where `evidence_first` showed improvements, such as `leisure`, `professional`, or `text`.

This is more aligned with the current error profile than forcing each sparse user-task block to match its historical score distribution.

