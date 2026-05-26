# URS Post-Hoc Arbitration Dev Search Results

## Setup

This report records the dev-selected post-hoc arbitration rule and its fixed test-set performance.

Dev search input:

- Base: `outputs/urs/dev/Qwen_Qwen3-8B_dev_none_urs_v2_calibrated_task_guarded.jsonl`
- Auxiliary: `outputs/urs/dev/Qwen_Qwen3-8B_dev_none_urs_v2_calibrated_task_guarded_evidence_first.jsonl`
- Grid output: `outputs/urs/dev/qwen3_8b_urs_posthoc_arbitration_grid_dev.json`

Test output:

- `outputs/urs/posthoc_dev_selected/urs_posthoc_selected_upgrade.jsonl`
- `outputs/urs/qwen3_8b_urs_posthoc_dev_selected_test.json`
- `outputs/urs/qwen3_8b_urs_posthoc_dev_selected_test_compare.json`

## Dev-Selected Rule

The best dev rule under the balanced search metric was:

- Strategy: `selected_upgrade`
- Selected tasks: `advice`, `creative`, `retrieval`
- Changed dev records: 7

Dev metrics:

| metric | value |
|---|---:|
| MAE | 0.7297 |
| Spearman | 0.2163 |
| QWK | 0.1509 |
| Boundary Acc | 0.6351 |
| F1-DSAT | 0.5846 |
| False-SAT | 0.3448 |
| False-DSAT | 0.3778 |

The top dev trials were dominated by `selected_upgrade`.
This suggests the small dev split favored avoiding false dissatisfaction, but the resulting rule is SAT-biased.

## Fixed Test Results

| version | MAE | RMSE | Pearson | Spearman | QWK | Boundary Acc | F1-SAT | F1-DSAT | False-SAT | False-DSAT |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| task_guarded_v1 | 0.6952 | 1.0119 | 0.2760 | 0.2883 | 0.2424 | 0.6627 | 0.7398 | 0.5207 | 0.4780 | 0.2612 |
| test_selected_boundary | 0.6935 | 1.0043 | 0.2928 | 0.3061 | 0.2572 | 0.6644 | 0.7358 | 0.5399 | 0.4390 | 0.2797 |
| dev_selected_upgrade | 0.6935 | 1.0094 | 0.2681 | 0.2732 | 0.2309 | 0.6644 | 0.7525 | 0.4787 | 0.5610 | 0.2137 |
| evidence_first | 0.7209 | 1.0245 | 0.2473 | 0.2483 | 0.2143 | 0.6438 | 0.7219 | 0.5048 | 0.4829 | 0.2876 |

The dev-selected rule does not generalize well.
It improves F1-SAT and reduces False-DSAT, but it substantially hurts F1-DSAT and increases False-SAT.
This is the wrong tradeoff for a satisfaction evaluator where dissatisfied-session detection matters.

## Interpretation

The dev split has only 74 target sessions.
It is too small for reliable task-subset selection, especially when rules may change only 5-10 records.
The selected rule overfits the dev split and becomes SAT-biased on test.

The test-selected `selected_boundary_aux` remains the best diagnostic result, but it should be described carefully because the task whitelist was informed by test-set analysis.
For a clean reporting protocol, the safer default is still the raw `task_guarded_v1` evaluator.
The post-hoc boundary arbitration can be reported as an exploratory analysis or tuned on a larger dev split.

## Recommendation

Do not use the current 74-session dev split alone to select task-specific arbitration rules.

Better options:

- Use cross-validation over users and report averaged dev selection performance.
- Search only over strategy direction with a fixed task whitelist derived from qualitative analysis, not from small dev metrics.
- Increase the dev size by changing the original `train_ratio` if URS is used primarily for method development rather than final test reporting.
- Keep `task_guarded_v1` as the clean main evaluator and report `selected_boundary_aux` as an upper-bound or exploratory post-hoc variant.

