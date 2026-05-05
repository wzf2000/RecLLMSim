# Memory V2.1 Update Results

## Setup

Goal: test whether the new `memory_update_prompt_version=v2_1` can make Qwen3 `per_session_oracle` update genuinely useful, instead of producing near-zero net gain as in the old v2 update.

Compared files:

- `detection/outputs/personalized/Qwen_Qwen3-8B_test_none.jsonl`
- `detection/outputs/personalized/Qwen_Qwen3-8B_test_per_session_oracle.jsonl`
- `detection/outputs/personalized/Qwen_Qwen3-8B_test_per_session_oracle_updv2_1.jsonl`

Evaluation summary JSON:

- `detection/outputs/personalized/qwen_v2_1_oracle_compare.json`

## Main Result

`per_session_oracle + v2.1` is the first Qwen update variant that clearly beats `none` on the main personalized benchmark.

### Global metrics

| Method | MAE | RMSE | Pearson | Spearman | QWK |
|---|---:|---:|---:|---:|---:|
| `Qwen none` | `0.7110` | `1.0122` | `0.2967` | `0.2820` | `0.2815` |
| `Qwen per_session_oracle (old)` | `0.7141` | `1.0239` | `0.2737` | `0.2563` | `0.2632` |
| `Qwen per_session_oracle + v2.1` | **`0.6934`** | **`0.9951`** | **`0.3045`** | **`0.2931`** | **`0.2889`** |

Compared with `Qwen none`, `v2.1 oracle` gives:

- `MAE: 0.7110 -> 0.6934`  (`-0.0176`)
- `RMSE: 1.0122 -> 0.9951`
- `Pearson: 0.2967 -> 0.3045`
- `Spearman: 0.2820 -> 0.2931`
- `QWK: 0.2815 -> 0.2889`

So this is not a narrow MAE-only gain. It improves both absolute error and ordinal/ranking consistency.

## Boundary Metrics

| Method | Acc | F1-SAT | F1-DSAT | Kappa | AUC | False SAT | False DSAT |
|---|---:|---:|---:|---:|---:|---:|---:|
| `Qwen none` | `0.7555` | `0.8504` | `0.3312` | `0.1824` | `0.6444` | `0.6459` | `0.1617` |
| `Qwen per_session_oracle (old)` | `0.7651` | `0.8578` | `0.3249` | `0.1827` | `0.6392` | `0.6694` | `0.1453` |
| `Qwen per_session_oracle + v2.1` | **`0.7723`** | **`0.8617`** | **`0.3563`** | **`0.2182`** | **`0.6544`** | **`0.6314`** | **`0.1444`** |

Relative to `Qwen none`, `v2.1 oracle` also improves boundary performance:

- `Accuracy: 0.7555 -> 0.7723`
- `F1-DSAT: 0.3312 -> 0.3563`
- `Boundary kappa: 0.1824 -> 0.2182`
- `AUC: 0.6444 -> 0.6544`
- `False SAT: 0.6459 -> 0.6314`

This is important because old oracle update mainly traded off SAT/DSAT balance without giving a real win. `v2.1` improves the dissatisfied side instead of merely shifting predictions upward.

## User-Aware Metrics

### User-aware 1-5

| Method | PU-Pearson | PU-Spearman | PU-Kappa | WC-Pearson | WC-Spearman |
|---|---:|---:|---:|---:|---:|
| `Qwen none` | `0.1997` | `0.1798` | `0.1646` | `0.2154` | `0.1637` |
| `Qwen per_session_oracle (old)` | `0.1933` | `0.1737` | `0.1610` | `0.2073` | `0.1622` |
| `Qwen per_session_oracle + v2.1` | `0.1921` | `0.1764` | `0.1618` | `0.2069` | `0.1656` |

### User-aware boundary

| Method | PU-bin Acc | PU-bin F1-SAT | PU-bin F1-DSAT | PU-bin Kappa | WC-bin Pearson |
|---|---:|---:|---:|---:|---:|
| `Qwen none` | `0.7430` | `0.8244` | `0.2734` | `0.1196` | `0.1529` |
| `Qwen per_session_oracle (old)` | `0.7527` | `0.8356` | `0.2728` | `0.1303` | `0.1554` |
| `Qwen per_session_oracle + v2.1` | **`0.7563`** | `0.8353` | **`0.2835`** | **`0.1353`** | **`0.1579`** |

Interpretation:

- User-aware continuous metrics are roughly flat relative to `none`.
- But user-aware boundary metrics improve a bit, especially:
  - `PU-bin F1-DSAT: 0.2734 -> 0.2835`
  - `PU-bin Kappa: 0.1196 -> 0.1353`
  - `WC-bin Pearson: 0.1529 -> 0.1579`

So `v2.1` helps more on personalized boundary behavior than on per-user ranking correlation.

## How Much Did V2.1 Actually Change?

Shared samples across the three files: `6474`.

### Old oracle update vs `none`

- Score changed: `1183`
- Reason changed: `909`
- Better: `580`
- Worse: `602`
- Tie: `1`
- Net gain: `-22`

Most common score transitions:

- `4 -> 5`: `407`
- `3 -> 4`: `281`
- `5 -> 4`: `227`
- `4 -> 3`: `170`

### V2.1 oracle update vs `none`

- Score changed: `1584`
- Reason changed: `3740`
- Better: `841`
- Worse: `742`
- Tie: `1`
- Net gain: `+99`

Most common score transitions:

- `4 -> 5`: `400`
- `3 -> 4`: `372`
- `5 -> 4`: `364`
- `4 -> 3`: `310`

Interpretation:

1. `v2.1` is not conservative in the same way as old v2 update. It changes more turns.
2. More importantly, it changes them in a direction that is **net helpful** instead of nearly canceling out.
3. The useful part is not only pushing scores upward. It also introduces more downward corrections, but the overall mix is now better aligned with gold labels.

## Prediction Distribution

Predicted score counts:

### `Qwen none`

- `1`: `30`
- `2`: `155`
- `3`: `1075`
- `4`: `3688`
- `5`: `1526`

### `Qwen per_session_oracle (old)`

- `1`: `34`
- `2`: `154`
- `3`: `958`
- `4`: `3621`
- `5`: `1707`

### `Qwen per_session_oracle + v2.1`

- `1`: `28`
- `2`: `122`
- `3`: `1033`
- `4`: `3729`
- `5`: `1562`

Interpretation:

- Old oracle update over-pushed high scores, especially `4 -> 5`.
- `v2.1` is still willing to promote some `4` to `5`, but is much less skewed than old oracle update.
- Compared with old oracle update, `v2.1` looks more balanced and closer to the stable `none` distribution.

## Reason Consistency

Reason legality under the new score/reason rule:

- `pred_score >= 4` must have `reason_prediction = 满意`
- `pred_score <= 3` must not have `reason_prediction = 满意`

Counts:

| Method | Illegal high-score reasons | Illegal low-score reasons |
|---|---:|---:|
| `Qwen none` | `3302` | `0` |
| `Qwen per_session_oracle (old)` | `3229` | `0` |
| `Qwen per_session_oracle + v2.1` | **`0`** | **`0`** |

This semantic cleanup does not explain the score improvements by itself, but it makes the outputs much more consistent and usable.

## Task-Level MAE

`v2.1 oracle` MAE by task:

- `技能学习规划`: `0.6360`
- `旅行规划`: `0.7072`
- `礼物准备`: `0.7387`
- `菜谱规划`: `0.6700`

Compared with `none`, all four tasks improve:

- `技能学习规划`: `0.6516 -> 0.6360`
- `旅行规划`: `0.7098 -> 0.7072`
- `礼物准备`: `0.7637 -> 0.7387`
- `菜谱规划`: `0.7004 -> 0.6700`

The largest gain is on `礼物准备`, followed by `菜谱规划`.

## Conclusions

1. The old diagnosis was correct: Qwen v2 update was not a no-op, but its net effect was near zero because better and worse changes nearly canceled.
2. `memory_update_prompt_version=v2_1` fixes that problem to a meaningful degree.
3. `per_session_oracle + v2.1` is now better than `Qwen none` on:
   - global 1-5 metrics
   - boundary metrics
   - user-aware boundary metrics
4. User-aware continuous correlation gains are still limited, so the remaining bottleneck is probably not just update noise, but how updated memory is consumed during turn evaluation.

## Next Questions

The next most useful follow-up is:

1. inspect updated memory snapshots directly to see which patched fields correlate with gains
2. compare `v2.1 oracle` with `none + calibration`, to check whether update and post-hoc calibration are complementary or redundant
3. test whether a lighter non-oracle variant can improve DSAT without sacrificing ranking

## Per-Session V2.1 Update

The non-oracle `per_session + v2.1` run also improves over `Qwen none`, which is important because it shows the new update mechanism is not only an oracle artifact.

Compared files:

- `detection/outputs/personalized/Qwen_Qwen3-8B_test_per_session.jsonl`
- `detection/outputs/personalized/Qwen_Qwen3-8B_test_per_session_updv2_1.jsonl`

Comparison JSON:

- `detection/outputs/personalized/qwen_v2_1_update_compare.json`

### Global metrics

| Method | MAE | RMSE | Pearson | Spearman | QWK |
|---|---:|---:|---:|---:|---:|
| `Qwen none` | `0.7110` | `1.0122` | `0.2967` | `0.2820` | `0.2815` |
| `Qwen per_session (old)` | `0.7133` | `1.0203` | `0.2812` | `0.2619` | `0.2694` |
| `Qwen per_session + v2.1` | **`0.6997`** | **`0.9992`** | `0.2931` | `0.2763` | `0.2781` |

Relative to `Qwen none`, `per_session + v2.1` gives:

- `MAE: 0.7110 -> 0.6997`
- `RMSE: 1.0122 -> 0.9992`
- `Pearson: 0.2967 -> 0.2931`
- `Spearman: 0.2820 -> 0.2763`
- `QWK: 0.2815 -> 0.2781`

Interpretation:

- Non-oracle `v2.1` clearly improves absolute error.
- But unlike oracle `v2.1`, it does **not** improve the correlation / QWK side.
- So its gain is narrower and more calibration-like.

### Boundary metrics

| Method | Acc | F1-SAT | F1-DSAT | Kappa | AUC | False SAT | False DSAT |
|---|---:|---:|---:|---:|---:|---:|---:|
| `Qwen none` | `0.7555` | `0.8504` | `0.3312` | `0.1824` | `0.6444` | `0.6459` | `0.1617` |
| `Qwen per_session (old)` | `0.7610` | `0.8547` | `0.3271` | `0.1821` | `0.6437` | `0.6603` | `0.1520` |
| `Qwen per_session + v2.1` | **`0.7685`** | **`0.8602`** | `0.3269` | **`0.1871`** | **`0.6448`** | `0.6712` | **`0.1409`** |

Interpretation:

- `per_session + v2.1` improves boundary accuracy and SAT-side metrics.
- But it does not improve `F1-DSAT`; in fact it is slightly worse than `none`.
- It mainly reduces false DSAT by predicting SAT a bit more aggressively.

### User-aware metrics

| Method | PU-Pearson | PU-Spearman | PU-Kappa | WC-Pearson | WC-Spearman |
|---|---:|---:|---:|---:|---:|
| `Qwen none` | `0.1997` | `0.1798` | `0.1646` | `0.2154` | `0.1637` |
| `Qwen per_session + v2.1` | `0.2001` | `0.1789` | `0.1624` | `0.2139` | `0.1646` |

User-aware boundary:

| Method | PU-bin Acc | PU-bin F1-SAT | PU-bin F1-DSAT | PU-bin Kappa | WC-bin Pearson |
|---|---:|---:|---:|---:|---:|
| `Qwen none` | `0.7430` | `0.8244` | `0.2734` | `0.1196` | `0.1529` |
| `Qwen per_session + v2.1` | `0.7543` | `0.8342` | `0.2647` | `0.1195` | `0.1621` |

Interpretation:

- User-aware continuous metrics are effectively flat.
- User-aware boundary becomes slightly more SAT-oriented:
  - higher `PU-bin Accuracy`
  - lower `PU-bin F1-DSAT`

So the non-oracle update is useful, but its effect is weaker and less balanced than oracle `v2.1`.

### How much did it change?

Compared with `Qwen none`:

- score changed: `1480`
- reason changed: `3721`
- better: `766`
- worse: `714`
- net gain: `+52`

Most common score transitions:

- `4 -> 5`: `381`
- `3 -> 4`: `366`
- `5 -> 4`: `352`
- `4 -> 3`: `244`

This is still a real net-positive update, but weaker than oracle `v2.1`:

- `per_session + v2.1`: net `+52`
- `per_session_oracle + v2.1`: net `+99`

### Overall takeaway for v2.1

The new `v2.1` update mechanism works in both oracle and non-oracle settings, but the two variants help in different ways:

- `per_session_oracle + v2.1`
  - best overall update result
  - improves global, boundary, and user-aware boundary metrics together
- `per_session + v2.1`
  - still better than `none` on MAE / RMSE
  - but behaves more like a SAT-leaning calibration improvement than a true DSAT/boundary improvement

So if the question is "did v2.1 fix the old Qwen update failure?", the answer is yes.  
If the question is "is non-oracle update already as good as oracle update?", the answer is no.
