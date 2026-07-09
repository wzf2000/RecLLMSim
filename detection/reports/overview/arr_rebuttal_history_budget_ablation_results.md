# ARR Rebuttal: Full History-Budget Ablation Results

本文档记录 Reviewer fUbw W4 对 history amount robustness 的补充实验结果。
实验使用全量 test split 上满足 `min_history_sessions=4` 的 user-target blocks，并比较 memory construction 使用 `K=1,2,4` 个 source-history sessions 和使用全部 source-history sessions 的结果。

## 1. Source Files

- `detection/outputs/personalized/qwen3_8b_history_budget_k1_full.jsonl`
- `detection/outputs/personalized/qwen3_8b_history_budget_k2_full.jsonl`
- `detection/outputs/personalized/qwen3_8b_history_budget_k4_full.jsonl`
- `detection/outputs/personalized/qwen3_8b_history_budget_all_full.jsonl`
- `detection/outputs/personalized/qwen3_8b_history_budget_full_eval.json`

The four settings use the same 6,449 target turns from 88 users and 350 user-target blocks.
The experiment uses raw 1--5 evaluator outputs in `pred_score`; it does not apply the reference-CDF post-hoc calibration used in the final benchmark table.
Therefore, the main comparison should be made within this ablation rather than directly against the refCDF values in the paper.

## 2. History Coverage Check

| Setting | Target turns | Users | Blocks | History sessions used | Original history sessions |
|---|---:|---:|---:|---:|---:|
| K=1 | 6,449 | 88 | 350 | 1.00 avg | 12.62 avg |
| K=2 | 6,449 | 88 | 350 | 2.00 avg | 12.62 avg |
| K=4 | 6,449 | 88 | 350 | 4.00 avg | 12.62 avg |
| All | 6,449 | 88 | 350 | 12.62 avg | 12.62 avg |

For all K-limited settings, source histories were selected with `round_robin_task`.
This keeps small budgets deterministic and avoids taking all histories from a single source scenario when possible.

## 3. Main Metrics

| Setting | MAE ↓ | RMSE ↓ | Pearson ↑ | Spearman ↑ | QWK ↑ | Bin. Acc. ↑ | Low-side F1 ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|
| K=1 | 0.7262 | 1.0452 | 0.2728 | 0.2536 | 0.2642 | 0.7624 | 0.3345 |
| K=2 | 0.6807 | 0.9843 | 0.2855 | 0.2735 | 0.2762 | 0.7787 | 0.3083 |
| K=4 | 0.6758 | 0.9761 | 0.2737 | 0.2602 | 0.2640 | 0.7918 | 0.2920 |
| All | 0.6624 | 0.9568 | 0.3005 | 0.2889 | 0.2874 | 0.7944 | 0.3065 |

The full-history setting has the best full-score metrics: MAE, RMSE, Pearson, Spearman, and QWK.
The trend is clearest for MAE/RMSE and becomes smaller after K=2.
Binary accuracy also improves with more history.
Low-side F1 is not monotonic because K=1 predicts more low-side labels, increasing recall at the cost of more false low-side predictions.

## 4. User-Level Bootstrap Confidence Intervals

We resampled users with replacement 1,000 times using seed 42.
Each resample includes all target turns for sampled users.

| Setting | MAE ↓ | RMSE ↓ | Pearson ↑ | Spearman ↑ | QWK ↑ | Low-side F1 ↑ |
|---|---:|---:|---:|---:|---:|---:|
| K=1 | 0.7262 [0.6852, 0.7717] | 1.0452 [0.9950, 1.0961] | 0.2728 [0.2179, 0.3256] | 0.2536 [0.2032, 0.3018] | 0.2642 [0.2108, 0.3142] | 0.3345 [0.2801, 0.3823] |
| K=2 | 0.6807 [0.6429, 0.7213] | 0.9843 [0.9413, 1.0287] | 0.2855 [0.2336, 0.3378] | 0.2735 [0.2248, 0.3245] | 0.2762 [0.2244, 0.3267] | 0.3083 [0.2583, 0.3594] |
| K=4 | 0.6758 [0.6405, 0.7129] | 0.9761 [0.9348, 1.0199] | 0.2737 [0.2236, 0.3245] | 0.2602 [0.2148, 0.3041] | 0.2640 [0.2150, 0.3138] | 0.2920 [0.2443, 0.3418] |
| All | 0.6624 [0.6295, 0.6979] | 0.9568 [0.9153, 0.9982] | 0.3005 [0.2550, 0.3480] | 0.2889 [0.2493, 0.3308] | 0.2874 [0.2428, 0.3333] | 0.3065 [0.2682, 0.3430] |

## 5. Difference Against All-History

For MAE/RMSE, positive values mean the K-limited setting is worse than all-history.
For Pearson, Spearman, QWK, and Low-side F1, positive values mean all-history is better than the K-limited setting.

| K-limited setting | ΔMAE ↓ | ΔRMSE ↓ | ΔPearson ↑ | ΔSpearman ↑ | ΔQWK ↑ | ΔLow-side F1 ↑ |
|---|---:|---:|---:|---:|---:|---:|
| K=1 vs All | 0.0637 [0.0385, 0.0899] | 0.0884 [0.0598, 0.1214] | 0.0278 [-0.0128, 0.0657] | 0.0353 [-0.0055, 0.0732] | 0.0232 [-0.0156, 0.0602] | -0.0280 [-0.0611, 0.0070] |
| K=2 vs All | 0.0183 [-0.0016, 0.0378] | 0.0275 [0.0082, 0.0460] | 0.0150 [-0.0151, 0.0439] | 0.0155 [-0.0175, 0.0456] | 0.0112 [-0.0184, 0.0386] | -0.0018 [-0.0337, 0.0278] |
| K=4 vs All | 0.0133 [-0.0050, 0.0321] | 0.0193 [0.0003, 0.0396] | 0.0268 [0.0021, 0.0513] | 0.0288 [0.0012, 0.0566] | 0.0234 [-0.0007, 0.0470] | 0.0144 [-0.0132, 0.0435] |

The most robust difference is between K=1 and all-history on MAE/RMSE.
The gains from K=2 or K=4 to all-history are smaller, but RMSE consistently favors all-history.
For Pearson and Spearman, all-history is clearly better than K=4 under the bootstrap interval, while the all-history advantage over K=2 is directionally positive but not clearly separated.

## 6. Prediction Distribution

| Setting | Score 1 | Score 2 | Score 3 | Score 4 | Score 5 | Low-side predictions |
|---|---:|---:|---:|---:|---:|---:|
| Gold | 100 | 219 | 699 | 2,429 | 3,002 | 1,018 |
| K=1 | 32 | 267 | 985 | 3,274 | 1,891 | 1,284 |
| K=2 | 24 | 116 | 905 | 3,494 | 1,910 | 1,045 |
| K=4 | 25 | 102 | 752 | 3,656 | 1,914 | 879 |
| All | 27 | 94 | 773 | 3,786 | 1,769 | 894 |

K=1 is more aggressive in predicting score 3 and therefore obtains higher low-side F1, but it also has lower full-score accuracy and worse MAE/RMSE.
As more history is added, the evaluator becomes less likely to over-predict low-side scores and better aligned with the full 1--5 scale.

## 7. Interpretation for Rebuttal

This ablation supports a moderate and defensible conclusion.
The evaluator benefits from having more user history, especially when moving from one source conversation to richer user evidence.
The all-history setting gives the best overall full-score performance and the best raw-score correlation metrics.
At the same time, the gains saturate quickly after K=2, and low-side F1 is not monotonic because sparse histories produce more low-side predictions.

Suggested wording:

> We add a full-split history-budget ablation that keeps the target turns fixed and limits memory construction to K source conversations per user-target block.
> Using all available histories gives the best full-score metrics, improving MAE from 0.7262 at K=1 to 0.6624 and Pearson from 0.2728 to 0.3005.
> The improvement is largest when moving away from extremely sparse histories, while the gap between K=2/K=4 and all-history is smaller.
> This suggests that the evaluator benefits from richer user evidence but does not require a very large amount of history to remain usable.
> Low-side F1 is not monotonic because K=1 predicts more low-side turns; we will report this tradeoff explicitly rather than claiming uniform improvement across all metrics.

Recommended paper/rebuttal use:

- Use the main metric table and prediction distribution table in internal rebuttal notes.
- In the OpenReview response, report only the compact trend: K=1, K=2, K=4, All for MAE/Pearson/QWK/Low-side F1.
- Avoid claiming monotonic improvement on every metric.
- Emphasize that additional user history helps full-score evaluation and that sparse-history personalization remains a limitation.
