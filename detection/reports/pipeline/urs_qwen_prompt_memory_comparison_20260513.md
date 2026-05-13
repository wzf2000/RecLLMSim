# URS Qwen3-8B Prompt And Memory Comparison

## Setup

Compared four URS satisfaction predictor runs:

- `mem_v2_local`: `outputs/urs/Qwen_Qwen3-8B_test_none_local.jsonl`
- `mem_calibrated`: `outputs/urs/Qwen_Qwen3-8B_test_none_urs_v2_calibrated.jsonl`
- `no_mem_v2`: `outputs/urs/Qwen_Qwen3-8B_test_no_memory_v2.jsonl`
- `no_mem_calibrated`: `outputs/urs/Qwen_Qwen3-8B_test_no_memory_urs_v2_calibrated.jsonl`

All runs cover the same URS test set:

- `584` sessions
- `116` users
- `339` user-intent blocks

Detailed JSON summary:

- `outputs/urs/qwen3_8b_urs_prompt_memory_comparison.json`

## Main Metrics

| Run | MAE | RMSE | Pearson | Spearman | QWK | Boundary Acc | F1-DSAT | F1-SAT |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| mem_v2_local | 0.8305 | 1.1018 | 0.2312 | 0.2354 | 0.1868 | 0.5394 | 0.5386 | 0.5402 |
| mem_calibrated | 0.7260 | 1.0353 | 0.2358 | 0.2495 | 0.2073 | 0.6370 | 0.4592 | 0.7268 |
| no_mem_v2 | 1.0497 | 1.3428 | 0.1588 | 0.1295 | 0.1396 | 0.4812 | 0.5152 | 0.4420 |
| no_mem_calibrated | 0.7928 | 1.0925 | 0.1621 | 0.1527 | 0.1411 | 0.6301 | 0.3415 | 0.7429 |

## Score Distributions

Gold distribution:

- `1`: 19
- `2`: 47
- `3`: 139
- `4`: 250
- `5`: 129

Predicted distributions:

| Run | 1 | 2 | 3 | 4 | 5 |
|---|---:|---:|---:|---:|---:|
| mem_v2_local | 3 | 9 | 366 | 185 | 21 |
| mem_calibrated | 1 | 11 | 175 | 368 | 29 |
| no_mem_v2 | 11 | 87 | 322 | 87 | 77 |
| no_mem_calibrated | 0 | 11 | 112 | 395 | 66 |

## Interpretation

### Calibrated prompt helps memory-based global metrics

Compared with `mem_v2_local`, `mem_calibrated` improves:

- MAE: `0.8305 -> 0.7260`
- RMSE: `1.1018 -> 1.0353`
- Pearson: `0.2312 -> 0.2358`
- Spearman: `0.2354 -> 0.2495`
- QWK: `0.1868 -> 0.2073`
- Boundary accuracy: `0.5394 -> 0.6370`

It also fixes the strongest score-3 collapse:

- score-3 predictions: `366 -> 175`
- score-4 predictions: `185 -> 368`

This is the best run by MAE, RMSE, Spearman, QWK, and boundary accuracy.

### The tradeoff is weaker DSAT detection

The same calibrated prompt makes the model more willing to predict SAT:

- pred SAT: `206 -> 397`
- pred DSAT: `378 -> 187`

This improves F1-SAT, but reduces F1-DSAT:

- F1-SAT: `0.5402 -> 0.7268`
- F1-DSAT: `0.5386 -> 0.4592`

So the calibrated prompt is better for overall satisfaction scoring, but worse
if DSAT recall is the primary target.

### No-memory has better user-aware metrics but worse global accuracy

`no_mem_v2` performs worse globally, but its user-aware correlations are
positive:

- PU-Pearson: `0.1965`
- WC-Pearson: `0.2003`

In contrast, memory-based runs still have negative user-aware correlations.
This suggests the current URS memory summary may inject noisy or misleading
personalization, even when it helps global calibration.

### Language gap remains

For `mem_calibrated`:

- zh: MAE `0.6923`, QWK `0.2347`, F1-DSAT `0.4688`
- en: MAE `0.7818`, QWK `0.1750`, F1-DSAT `0.4412`

Chinese remains stronger than English, though the calibrated prompt improves
both language subsets in global metrics.

## Recommendation

Use `mem_calibrated` as the default URS judge if the goal is benchmark scoring
with reasonable global behavior.

For DSAT-focused analysis, keep `mem_v2_local` as a secondary diagnostic because
it has better F1-DSAT.

For studying whether URS memory is useful, compare against `no_mem_v2` because
it has better user-aware metrics despite worse global metrics. This indicates
that the current memory formulation is not yet a reliable source of
personalized signal on URS.
