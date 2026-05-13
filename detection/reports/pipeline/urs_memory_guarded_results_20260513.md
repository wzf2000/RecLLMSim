# URS Memory-Guarded Prompt Results

## Setup

New run:

- `outputs/urs/Qwen_Qwen3-8B_test_none_urs_v2_memory_guarded_local.jsonl`
- Model: `Qwen/Qwen3-8B`
- Memory update mode: `none`
- Prompt version: `urs_v2_memory_guarded`
- Coverage: `584` sessions, `116` users, `339` user-intent blocks

Comparison JSON:

- `outputs/urs/qwen3_8b_urs_memory_guarded_comparison.json`

## Main Results

| Run | MAE | RMSE | Pearson | Spearman | QWK | Boundary Acc | F1-DSAT | F1-SAT |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| mem_v2_local | 0.8305 | 1.1018 | 0.2312 | 0.2354 | 0.1868 | 0.5394 | 0.5386 | 0.5402 |
| mem_calibrated | 0.7260 | 1.0353 | 0.2358 | 0.2495 | 0.2073 | 0.6370 | 0.4592 | 0.7268 |
| mem_guarded | 0.7483 | 1.0558 | 0.1851 | 0.1928 | 0.1590 | 0.6164 | 0.4563 | 0.7037 |
| no_mem_v2 | 1.0497 | 1.3428 | 0.1588 | 0.1295 | 0.1396 | 0.4812 | 0.5152 | 0.4420 |
| no_mem_calibrated | 0.7928 | 1.0925 | 0.1621 | 0.1527 | 0.1411 | 0.6301 | 0.3415 | 0.7429 |

## Score Distribution

| Run | 1 | 2 | 3 | 4 | 5 |
|---|---:|---:|---:|---:|---:|
| mem_v2_local | 3 | 9 | 366 | 185 | 21 |
| mem_calibrated | 1 | 11 | 175 | 368 | 29 |
| mem_guarded | 1 | 6 | 200 | 356 | 21 |
| no_mem_v2 | 11 | 87 | 322 | 87 | 77 |
| no_mem_calibrated | 0 | 11 | 112 | 395 | 66 |

## Interpretation

`urs_v2_memory_guarded` worked as intended mechanically: it reduced some
extreme SAT inflation compared with `mem_calibrated`:

- pred SAT: `397 -> 377`
- pred DSAT: `187 -> 207`
- score-3 predictions: `175 -> 200`

However, it did not improve the main metrics:

- MAE worsened: `0.7260 -> 0.7483`
- Pearson worsened: `0.2358 -> 0.1851`
- Spearman worsened: `0.2495 -> 0.1928`
- QWK worsened: `0.2073 -> 0.1590`
- Boundary accuracy worsened: `0.6370 -> 0.6164`
- F1-DSAT stayed almost unchanged: `0.4592 -> 0.4563`

The guarded instructions made the model more cautious about memory, but this
mostly moved predictions from `4` to `3` without recovering enough true DSAT
cases. It also weakened ranking quality, especially on English.

## Language Breakdown

| Run | Lang | MAE | Pearson | QWK | F1-DSAT |
|---|---|---:|---:|---:|---:|
| mem_calibrated | en | 0.7818 | 0.1961 | 0.1750 | 0.4412 |
| mem_guarded | en | 0.8000 | 0.0991 | 0.0837 | 0.4173 |
| mem_calibrated | zh | 0.6923 | 0.2779 | 0.2347 | 0.4688 |
| mem_guarded | zh | 0.7170 | 0.2373 | 0.2032 | 0.4762 |

The guarded prompt slightly improves Chinese F1-DSAT but degrades most other
metrics. English degrades substantially.

## Recommendation

Do not use `urs_v2_memory_guarded` as the default URS judge.

Current preferred choices:

- Main URS benchmark judge: `mem_calibrated`
- DSAT diagnostic judge: `mem_v2_local`
- Personalization sanity check: `no_mem_v2`

The result suggests that prompt-level memory guarding alone is insufficient.
Future URS memory improvements should likely change the memory representation
or retrieval mechanism, not just add cautionary prompt text.
