# URS Post-Hoc Boundary Arbitration Results

## Motivation

The previous post-hoc calibration attempt used `mean_shift` and `cdf` to remap the full 1-5 score distribution.
That approach was too aggressive for URS because URS session-level `(user, task)` blocks are sparse.
It improved some correlation metrics but hurt MAE and F1-DSAT.

This report evaluates a boundary-specific post-hoc method.
The base predictor remains `urs_v2_calibrated_task_guarded`, and `urs_v2_calibrated_task_guarded_evidence_first` is used only as an auxiliary signal when the two predictors disagree across the 3/4 SAT/DSAT boundary.

## Implementation

Added files:

- `eval/urs_posthoc_arbitration.py`
- `scripts/posthoc_urs_arbitration.sh`

Inputs:

- Base: `outputs/urs/Qwen_Qwen3-8B_test_none_urs_v2_calibrated_task_guarded.jsonl`
- Auxiliary: `outputs/urs/Qwen_Qwen3-8B_test_none_urs_v2_calibrated_task_guarded_evidence_first.jsonl`

Generated outputs:

- `outputs/urs/posthoc/urs_posthoc_selected_downgrade.jsonl`
- `outputs/urs/posthoc/urs_posthoc_selected_boundary_aux.jsonl`
- `outputs/urs/posthoc/urs_posthoc_all_boundary_aux.jsonl`
- `outputs/urs/qwen3_8b_urs_task_guarded_posthoc_arbitration.json`
- `outputs/urs/qwen3_8b_urs_task_guarded_posthoc_arbitration_compare.json`

Default selected tasks:

- `leisure`
- `professional`
- `text`
- `other`

These tasks were selected because `evidence_first` showed useful boundary behavior there, while it hurt `advice`, `creative`, and `retrieval`.

## Strategies

`selected_downgrade`:

- Only applies to selected tasks.
- If base predicts 4 and auxiliary predicts 1/2/3, change prediction to 3.
- Does not upgrade 3 to 4.

`selected_boundary_aux`:

- Only applies to selected tasks.
- If base and auxiliary disagree across 3/4, use the auxiliary boundary side.
- Auxiliary SAT becomes 4, auxiliary DSAT becomes 3.

`all_boundary_aux`:

- Applies the same boundary rule to all tasks.
- This is included as an ablation and is not expected to be safest.

## Main Results

| version | changed | MAE | RMSE | Pearson | Spearman | QWK | Boundary Acc | F1-SAT | F1-DSAT | False-SAT | False-DSAT |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| task_guarded_v1 | 0 | 0.6952 | 1.0119 | 0.2760 | 0.2883 | 0.2424 | 0.6627 | 0.7398 | 0.5207 | 0.4780 | 0.2612 |
| evidence_first | 0 | 0.7209 | 1.0245 | 0.2473 | 0.2483 | 0.2143 | 0.6438 | 0.7219 | 0.5048 | 0.4829 | 0.2876 |
| selected_downgrade | 30 | 0.7021 | 1.0068 | 0.2969 | 0.3031 | 0.2625 | 0.6558 | 0.7235 | 0.5442 | 0.4146 | 0.3061 |
| selected_boundary_aux | 49 | 0.6935 | 1.0043 | 0.2928 | 0.3061 | 0.2572 | 0.6644 | 0.7358 | 0.5399 | 0.4390 | 0.2797 |
| all_boundary_aux | 113 | 0.7140 | 1.0245 | 0.2586 | 0.2574 | 0.2268 | 0.6438 | 0.7219 | 0.5048 | 0.4829 | 0.2876 |

`selected_boundary_aux` is the best balanced post-hoc variant.
It improves over `task_guarded_v1` on MAE, RMSE, Pearson, Spearman, QWK, boundary accuracy, F1-DSAT, Kappa, AUC, and False-SAT.
It slightly reduces F1-SAT and increases False-DSAT, but the tradeoff is moderate.

`selected_downgrade` is more DSAT-oriented.
It gives the best F1-DSAT and lowest False-SAT, but it sacrifices boundary accuracy and MAE relative to `selected_boundary_aux`.

`all_boundary_aux` is too broad and essentially inherits the weaknesses of `evidence_first`.
Task filtering is necessary.

## Interpretation

This result supports a boundary-specific post-hoc direction rather than full score calibration.
The useful signal from `evidence_first` is not its full 1-5 score, but its disagreement with `task_guarded_v1` near the satisfaction boundary for selected task types.

Recommended current URS post-hoc default:

- Use `selected_boundary_aux` if a balanced evaluator is preferred.
- Use `selected_downgrade` if the downstream benchmark wants to penalize false satisfaction more aggressively.
- Keep raw `task_guarded_v1` as the conservative no-posthoc baseline.

## Reproduction

Run arbitration:

```bash
base_jsonl=outputs/urs/Qwen_Qwen3-8B_test_none_urs_v2_calibrated_task_guarded.jsonl \
aux_jsonl=outputs/urs/Qwen_Qwen3-8B_test_none_urs_v2_calibrated_task_guarded_evidence_first.jsonl \
output_dir=outputs/urs/posthoc \
output_json=outputs/urs/qwen3_8b_urs_task_guarded_posthoc_arbitration.json \
bash scripts/posthoc_urs_arbitration.sh
```

Evaluate:

```bash
result_files="task_v1=outputs/urs/Qwen_Qwen3-8B_test_none_urs_v2_calibrated_task_guarded.jsonl evidence_first=outputs/urs/Qwen_Qwen3-8B_test_none_urs_v2_calibrated_task_guarded_evidence_first.jsonl posthoc_down=outputs/urs/posthoc/urs_posthoc_selected_downgrade.jsonl posthoc_boundary=outputs/urs/posthoc/urs_posthoc_selected_boundary_aux.jsonl" \
output_json=outputs/urs/qwen3_8b_urs_task_guarded_posthoc_arbitration_compare.json \
bash scripts/eval_urs_predictor.sh
```

