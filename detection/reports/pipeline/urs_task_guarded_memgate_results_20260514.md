# URS Task-Guarded Memgate Results

## Setup

This report compares the URS satisfaction evaluator variants using local `Qwen/Qwen3-8B` with `memory_update_mode=none`.

Compared outputs:

- `outputs/urs/Qwen_Qwen3-8B_test_none_urs_v2_calibrated.jsonl`
- `outputs/urs/Qwen_Qwen3-8B_test_none_urs_v2_calibrated_task_guarded.jsonl`
- `outputs/urs/Qwen_Qwen3-8B_test_none_urs_v2_calibrated_task_guarded_v2.jsonl`
- `outputs/urs/Qwen_Qwen3-8B_test_none_urs_v2_calibrated_task_guarded_memgate.jsonl`

Aggregated comparison file:

- `outputs/urs/qwen3_8b_urs_task_guarded_memgate_comparison.json`

All variants are evaluated on 584 URS test samples.

## Overall Results

| version | MAE | RMSE | Pearson | Spearman | QWK | Boundary Acc | F1-SAT | F1-DSAT | False-SAT | False-DSAT |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| calibrated | 0.7260 | 1.0353 | 0.2358 | 0.2495 | 0.2073 | 0.6370 | 0.7268 | 0.4592 | 0.5610 | 0.2559 |
| task_guarded_v1 | 0.6952 | 1.0119 | 0.2760 | 0.2883 | 0.2424 | 0.6627 | 0.7398 | 0.5207 | 0.4780 | 0.2612 |
| task_guarded_v2 | 0.7209 | 1.0378 | 0.2058 | 0.2224 | 0.1746 | 0.6353 | 0.7294 | 0.4409 | 0.5902 | 0.2427 |
| task_guarded_memgate | 0.7432 | 1.0534 | 0.2275 | 0.2290 | 0.2042 | 0.6336 | 0.7214 | 0.4650 | 0.5463 | 0.2691 |

`task_guarded_memgate` does not improve over `task_guarded_v1`.
It only slightly improves F1-DSAT over the calibrated baseline, but it is worse than `task_guarded_v1` on MAE, RMSE, Pearson, Spearman, QWK, boundary accuracy, F1-SAT, and F1-DSAT.

## Task Breakdown

F1-DSAT by task:

| task | calibrated | task_guarded_v1 | task_guarded_v2 | task_guarded_memgate |
|---|---:|---:|---:|---:|
| advice | 0.4800 | 0.5789 | 0.4545 | 0.4474 |
| creative | 0.3256 | 0.4091 | 0.3000 | 0.3000 |
| leisure | 0.4615 | 0.4848 | 0.4615 | 0.5128 |
| other | 0.4000 | 0.2857 | 0.3333 | 0.4000 |
| professional | 0.4156 | 0.5417 | 0.4324 | 0.4681 |
| retrieval | 0.4950 | 0.5714 | 0.5098 | 0.4835 |
| text | 0.5385 | 0.4561 | 0.4074 | 0.5455 |

The memgate prompt helps `text` and `leisure`, but loses most of the useful gains from `task_guarded_v1` on `advice`, `professional`, `retrieval`, and `creative`.
This suggests that the current prompt-side memory gate makes the model more conservative and weakens the direct task-aware evidence checks.

## Interpretation

The best current URS evaluator among these prompt variants is still `urs_v2_calibrated_task_guarded`.
The memory reliability gate should not replace it as the default evaluator.

The result also suggests that the main URS error mode is not simply "memory crosses the satisfaction boundary too aggressively".
For this dataset, task-local response quality cues appear more reliable than natural-language memory gating inside the same prompt.

If memory reliability is revisited, it should preferably be implemented as an external arbitration or post-hoc ensemble rule rather than another broad natural-language instruction inside the predictor prompt.
For example, one can compare `task_guarded_v1` with calibrated or no-memory predictions and only apply a fallback when the prediction disagreement is large and memory evidence is weak.

