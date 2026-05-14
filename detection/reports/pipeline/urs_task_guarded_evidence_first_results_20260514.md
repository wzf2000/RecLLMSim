# URS Task-Guarded Evidence-First Results

## Setup

This report evaluates `urs_v2_calibrated_task_guarded_evidence_first` on the URS test split with local `Qwen/Qwen3-8B`.

Compared outputs:

- `outputs/urs/Qwen_Qwen3-8B_test_none_urs_v2_calibrated.jsonl`
- `outputs/urs/Qwen_Qwen3-8B_test_none_urs_v2_calibrated_task_guarded.jsonl`
- `outputs/urs/Qwen_Qwen3-8B_test_none_urs_v2_calibrated_task_guarded_memgate.jsonl`
- `outputs/urs/Qwen_Qwen3-8B_test_none_urs_v2_calibrated_task_guarded_evidence_first.jsonl`

Aggregated comparison:

- `outputs/urs/qwen3_8b_urs_task_guarded_evidence_first_comparison.json`

All variants are evaluated on 584 URS test samples.

## Overall Results

| version | MAE | RMSE | Pearson | Spearman | QWK | Boundary Acc | F1-SAT | F1-DSAT | False-SAT | False-DSAT |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| calibrated | 0.7260 | 1.0353 | 0.2358 | 0.2495 | 0.2073 | 0.6370 | 0.7268 | 0.4592 | 0.5610 | 0.2559 |
| task_guarded_v1 | 0.6952 | 1.0119 | 0.2760 | 0.2883 | 0.2424 | 0.6627 | 0.7398 | 0.5207 | 0.4780 | 0.2612 |
| memgate | 0.7432 | 1.0534 | 0.2275 | 0.2290 | 0.2042 | 0.6336 | 0.7214 | 0.4650 | 0.5463 | 0.2691 |
| evidence_first | 0.7209 | 1.0245 | 0.2473 | 0.2483 | 0.2143 | 0.6438 | 0.7219 | 0.5048 | 0.4829 | 0.2876 |

`evidence_first` is better than `memgate` on most metrics and improves F1-DSAT over the calibrated baseline.
However, it still does not beat `task_guarded_v1` on global metrics or boundary metrics.

## User-Aware Metrics

| version | PU Pearson | PU F1-DSAT | WC Pearson | WC binary Pearson |
|---|---:|---:|---:|---:|
| calibrated | -0.1965 | 0.2893 | -0.0587 | -0.1140 |
| task_guarded_v1 | 0.0126 | 0.3727 | 0.0154 | 0.0239 |
| memgate | -0.1721 | 0.3389 | -0.0656 | -0.0523 |
| evidence_first | 0.1079 | 0.4067 | 0.0828 | 0.0643 |

`evidence_first` is the best among these variants on the user-aware metrics shown above.
This suggests that forcing the model to separate current-session evidence from memory may improve relative per-user behavior, even though it hurts or fails to improve absolute/global scoring compared with `task_guarded_v1`.

## Task Breakdown

Each cell shows `MAE / F1-DSAT`.

| task | task_guarded_v1 | memgate | evidence_first |
|---|---:|---:|---:|
| advice | 0.6250 / 0.5789 | 0.7396 / 0.4474 | 0.6875 / 0.5316 |
| creative | 0.7031 / 0.4091 | 0.7188 / 0.3000 | 0.7656 / 0.2564 |
| leisure | 0.7083 / 0.4848 | 0.7292 / 0.5128 | 0.6875 / 0.5714 |
| other | 0.7500 / 0.2857 | 0.6250 / 0.4000 | 0.6250 / 0.4000 |
| professional | 0.7778 / 0.5417 | 0.8632 / 0.4681 | 0.8291 / 0.5577 |
| retrieval | 0.6606 / 0.5714 | 0.7091 / 0.4835 | 0.6727 / 0.5106 |
| text | 0.7093 / 0.4561 | 0.6860 / 0.5455 | 0.6977 / 0.4912 |

`evidence_first` partially recovers the `memgate` damage on `advice`, `professional`, `retrieval`, and `text`.
It improves over `task_guarded_v1` on `leisure`, `other`, `professional` F1-DSAT, and user-aware metrics.
But it degrades `creative` sharply and remains worse than `task_guarded_v1` on the overall score.

## Interpretation

The evidence-first idea is directionally better than the stronger prompt-side memory gate, but it is still not a better default than `task_guarded_v1`.

The result suggests two separate objectives:

- If the goal is the strongest single URS evaluator by global and boundary metrics, use `urs_v2_calibrated_task_guarded`.
- If the goal is user-aware behavior, `evidence_first` is worth keeping as a candidate or ensemble component.

The next promising direction is not another longer prompt.
Instead, use `task_guarded_v1` as the base prediction and treat `evidence_first` as an arbitration signal only when the two disagree near the 3/4 boundary.
This may preserve the stronger global metrics of `task_guarded_v1` while recovering some user-aware benefit from `evidence_first`.

