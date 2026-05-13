# URS Ensemble Evaluator Test

## Purpose

Test whether multiple URS predictor variants can be combined offline to improve
evaluator reliability without additional LLM calls.

## Implementation

Added:

- `detection/eval/urs_ensemble.py`
- `detection/scripts/eval_urs_ensemble.sh`

The script reads multiple aligned URS predictor JSONL files, ensembles their
`pred_score`, and evaluates the ensemble against URS gold labels.

Supported strategies:

- `mean`
- `median`
- `majority_sat`
- `mean_if_confident`

Confidence labels are based on score span and SAT/DSAT agreement across runs:

- `high`: score span <= 1 and all runs agree on SAT/DSAT side.
- `medium`: score span <= 2 or all runs agree on SAT/DSAT side.
- `low`: large disagreement.

## Input Runs

Used five existing Qwen3-8B URS predictor outputs:

- `mem_v2`: `outputs/urs/Qwen_Qwen3-8B_test_none_local.jsonl`
- `mem_cal`: `outputs/urs/Qwen_Qwen3-8B_test_none_urs_v2_calibrated.jsonl`
- `mem_guard`: `outputs/urs/Qwen_Qwen3-8B_test_none_urs_v2_memory_guarded_local.jsonl`
- `no_mem_v2`: `outputs/urs/Qwen_Qwen3-8B_test_no_memory_v2.jsonl`
- `no_mem_cal`: `outputs/urs/Qwen_Qwen3-8B_test_no_memory_urs_v2_calibrated.jsonl`

All runs align on `584` URS test sessions.

Output files:

- `outputs/urs/qwen3_8b_urs_ensemble_eval.json`
- `outputs/urs/qwen3_8b_urs_ensemble_predictions.jsonl`

## Results

| Strategy | n | MAE | Pearson | Spearman | QWK | Acc | F1-DSAT | F1-SAT |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| mean | 584 | 0.7432 | 0.2667 | 0.2516 | 0.2249 | 0.6147 | 0.5077 | 0.6835 |
| median | 584 | 0.7500 | 0.2282 | 0.2298 | 0.1948 | 0.6147 | 0.4898 | 0.6905 |
| majority_sat | 584 | 0.7500 | 0.2307 | 0.2320 | 0.1980 | 0.6147 | 0.4898 | 0.6905 |
| mean_if_confident | 578 | 0.7405 | 0.2701 | 0.2539 | 0.2281 | 0.6142 | 0.5099 | 0.6819 |

Confidence distribution for full strategies:

- high: `151`
- medium: `427`
- low: `6`

## Comparison To Single Predictors

Best single global judge so far:

- `mem_calibrated`: MAE `0.7260`, Pearson `0.2358`, Spearman `0.2495`,
  QWK `0.2073`, Acc `0.6370`, F1-DSAT `0.4592`, F1-SAT `0.7268`

The `mean` ensemble:

- improves Pearson: `0.2358 -> 0.2667`
- improves QWK: `0.2073 -> 0.2249`
- improves F1-DSAT: `0.4592 -> 0.5077`
- slightly improves Spearman: `0.2495 -> 0.2516`
- worsens MAE: `0.7260 -> 0.7432`
- worsens boundary accuracy and F1-SAT.

`mean_if_confident` gives the best Pearson/QWK/F1-DSAT among ensemble variants,
but drops 6 low-confidence samples.

## Takeaway

Offline ensemble is promising for URS evaluator reliability. It reduces the
single-prompt bias of `mem_calibrated` and produces a more balanced evaluator,
especially for DSAT detection and ordinal correlation.

For benchmark scoring:

- Use `mem_calibrated` when MAE / SAT-favoring global score is preferred.
- Use `mean` or `mean_if_confident` ensemble when balanced reliability and
  F1-DSAT matter.
- Report low-confidence rate if using confidence-filtered scores.
