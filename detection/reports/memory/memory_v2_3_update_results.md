# Memory V2.3 Update Results

## Setup

Goal: evaluate whether `memory_update_prompt_version=v2_3` improves over the
previous non-oracle update variants.

`v2.3` was designed as a lighter alternative to `v2.2`:

- keep the raw turn-level evidence style from `v2.1`
- add a lightweight boundary summary
- add lightweight program-side gates for `three_vs_four` and `four_vs_five`
- filter generic requirements

Compared files:

- `detection/outputs/personalized/Qwen_Qwen3-8B_test_none.jsonl`
- `detection/outputs/personalized/Qwen_Qwen3-8B_test_per_session.jsonl`
- `detection/outputs/personalized/Qwen_Qwen3-8B_test_per_session_updv2_1.jsonl`
- `detection/outputs/personalized/Qwen_Qwen3-8B_test_per_session_updv2_2.jsonl`
- `detection/outputs/personalized/Qwen_Qwen3-8B_test_per_session_updv2_3.jsonl`
- `detection/outputs/personalized/Qwen_Qwen3-8B_test_per_session_oracle_updv2_1.jsonl`

Evaluation JSON:

- `detection/outputs/personalized/qwen_v2_3_update_compare.json`

All compared files contain `6474` shared test turns.

## Main Result

`per_session + v2.3` does not beat `Qwen none` or `per_session + v2.1` on
overall 1-5 prediction. It only shows a small boundary-side improvement over
`none`, mainly by reducing false SAT compared with `v2.1`.

### Global metrics

| Method | MAE | RMSE | Pearson | Spearman | QWK |
|---|---:|---:|---:|---:|---:|
| `Qwen none` | `0.7110` | `1.0122` | **`0.2967`** | `0.2820` | `0.2815` |
| `Qwen per_session old` | `0.7133` | `1.0203` | `0.2812` | `0.2619` | `0.2694` |
| `Qwen per_session + v2.1` | **`0.6997`** | **`0.9992`** | `0.2931` | `0.2763` | `0.2781` |
| `Qwen per_session + v2.2` | `0.7129` | `1.0137` | `0.2769` | `0.2607` | `0.2639` |
| `Qwen per_session + v2.3` | `0.7158` | `1.0146` | `0.2820` | `0.2655` | `0.2678` |
| `Qwen per_session_oracle + v2.1` | **`0.6934`** | **`0.9951`** | **`0.3045`** | **`0.2931`** | **`0.2889`** |

Compared with `none`, `v2.3` is worse on every global metric:

- `MAE: 0.7110 -> 0.7158`
- `RMSE: 1.0122 -> 1.0146`
- `Pearson: 0.2967 -> 0.2820`
- `Spearman: 0.2820 -> 0.2655`
- `QWK: 0.2815 -> 0.2678`

Compared with `v2.1`, the regression is clear:

- `MAE: 0.6997 -> 0.7158`
- `QWK: 0.2781 -> 0.2678`
- `Pearson: 0.2931 -> 0.2820`

So `v2.3` should not replace `v2.1` as the main non-oracle update method.

## Boundary Metrics

| Method | Acc | F1-SAT | F1-DSAT | Kappa | AUC | False SAT | False DSAT |
|---|---:|---:|---:|---:|---:|---:|---:|
| `Qwen none` | `0.7555` | `0.8504` | `0.3312` | `0.1824` | **`0.6444`** | `0.6459` | `0.1617` |
| `Qwen per_session old` | `0.7610` | `0.8547` | `0.3271` | `0.1821` | `0.6437` | `0.6603` | `0.1520` |
| `Qwen per_session + v2.1` | **`0.7685`** | **`0.8602`** | `0.3269` | `0.1871` | **`0.6448`** | `0.6712` | **`0.1409`** |
| `Qwen per_session + v2.2` | `0.7648` | `0.8575` | `0.3258` | `0.1834` | `0.6411` | `0.6676` | `0.1461` |
| `Qwen per_session + v2.3` | `0.7623` | `0.8552` | **`0.3358`** | **`0.1914`** | `0.6431` | `0.6486` | `0.1530` |
| `Qwen per_session_oracle + v2.1` | **`0.7723`** | **`0.8617`** | **`0.3563`** | **`0.2182`** | **`0.6544`** | **`0.6314`** | `0.1444` |

Boundary interpretation:

- `v2.3` improves `F1-DSAT` over `none`, `v2.1`, and `v2.2`.
- `v2.3` improves boundary kappa over non-oracle variants.
- But `v2.3` gives lower boundary accuracy and lower F1-SAT than `v2.1`.
- Its AUC is lower than `none` and `v2.1`.

This means `v2.3` is more DSAT-aware than `v2.1`, but not broadly better.

Predicted SAT/DSAT counts:

| Method | Pred SAT | Pred DSAT |
|---|---:|---:|
| `Qwen none` | `5214` | `1260` |
| `Qwen per_session + v2.1` | `5354` | `1120` |
| `Qwen per_session + v2.2` | `5322` | `1152` |
| `Qwen per_session + v2.3` | `5264` | `1210` |
| `Qwen per_session_oracle + v2.1` | `5291` | `1183` |

`v2.3` moves the non-oracle update away from the SAT-heavy behavior of `v2.1`,
but it still does not match the oracle `v2.1` boundary quality.

## User-Aware Metrics

### User-aware 1-5

| Method | PU-Pearson | PU-Spearman | PU-Kappa | WC-Pearson | WC-Spearman |
|---|---:|---:|---:|---:|---:|
| `Qwen none` | **`0.1997`** | **`0.1798`** | **`0.1646`** | **`0.2154`** | `0.1637` |
| `Qwen per_session + v2.1` | **`0.2001`** | `0.1789` | `0.1624` | `0.2139` | **`0.1646`** |
| `Qwen per_session + v2.2` | `0.1853` | `0.1623` | `0.1523` | `0.1987` | `0.1498` |
| `Qwen per_session + v2.3` | `0.1851` | `0.1638` | `0.1536` | `0.2054` | `0.1555` |
| `Qwen per_session_oracle + v2.1` | `0.1921` | `0.1764` | `0.1618` | `0.2069` | **`0.1656`** |

`v2.3` is still weak on user-aware continuous metrics. It is slightly better
than `v2.2` on within-user correlation, but it is worse than `none` and `v2.1`.

### User-aware boundary

| Method | PU-bin Acc | PU-bin F1-SAT | PU-bin F1-DSAT | PU-bin Kappa | WC-bin Pearson |
|---|---:|---:|---:|---:|---:|
| `Qwen none` | `0.7430` | `0.8244` | `0.2734` | `0.1196` | `0.1529` |
| `Qwen per_session old` | `0.7495` | `0.8316` | `0.2822` | **`0.1378`** | `0.1574` |
| `Qwen per_session + v2.1` | **`0.7543`** | **`0.8342`** | `0.2647` | `0.1195` | **`0.1621`** |
| `Qwen per_session + v2.2` | `0.7490` | `0.8313` | `0.2662` | `0.1203` | `0.1527` |
| `Qwen per_session + v2.3` | `0.7475` | `0.8286` | `0.2734` | `0.1236` | `0.1597` |
| `Qwen per_session_oracle + v2.1` | **`0.7563`** | **`0.8353`** | **`0.2835`** | `0.1353` | `0.1579` |

User-aware boundary interpretation:

- `v2.3` recovers DSAT F1 back to the `none` level.
- It does not match old per-session or oracle `v2.1` on per-user DSAT F1.
- It is worse than `v2.1` on per-user binary accuracy and F1-SAT.
- It is close to `v2.1` on within-user binary Pearson.

## Prediction Distribution

Predicted score counts:

| Method | 1 | 2 | 3 | 4 | 5 |
|---|---:|---:|---:|---:|---:|
| `Qwen none` | `30` | `155` | `1075` | `3688` | `1526` |
| `Qwen per_session old` | `31` | `156` | `1005` | `3628` | `1654` |
| `Qwen per_session + v2.1` | `30` | `126` | `964` | `3798` | `1556` |
| `Qwen per_session + v2.2` | `31` | `125` | `996` | `3710` | `1612` |
| `Qwen per_session + v2.3` | `32` | `127` | `1051` | `3709` | `1555` |
| `Qwen per_session_oracle + v2.1` | `28` | `122` | `1033` | `3729` | `1562` |

`v2.3` is less SAT-heavy than `v2.1`:

- more `3` predictions than `v2.1`
- fewer `4` predictions than `v2.1`
- almost identical number of `5` predictions

This explains why DSAT-side boundary metrics recover while global MAE worsens.
It is moving some turns down across the 3/4 boundary, but these moves are not
accurate enough for 1-5 prediction.

## Direct Change Analysis

Compared with `Qwen none`:

- score changed: `1540`
- reason changed: `3670`
- better: `751`
- worse: `789`
- net gain: `-38`

Top score transitions:

- `4 -> 5`: `399`
- `5 -> 4`: `369`
- `3 -> 4`: `336`
- `4 -> 3`: `295`
- `2 -> 3`: `62`
- `3 -> 2`: `45`

Compared with `per_session + v2.1`:

- score changed: `1278`
- reason changed: `704`
- better: `589`
- worse: `688`
- net gain: `-99`

Top score transitions:

- `5 -> 4`: `329`
- `4 -> 5`: `327`
- `4 -> 3`: `295`
- `3 -> 4`: `210`
- `2 -> 3`: `44`
- `3 -> 2`: `40`

This is the key diagnosis: `v2.3` is not just a conservative variant of
`v2.1`. It changes many scores relative to `v2.1`, especially by increasing
`4 -> 3` moves. Some of those help boundary DSAT, but the net 1-5 effect is
harmful.

## Reason Consistency

Reason legality is correct for `v2.3`:

- `pred_score >= 4` and `reason_prediction != 满意`: `0`
- `pred_score <= 3` and `reason_prediction == 满意`: `0`

The regression is therefore not a reason-label validity issue. It is a score
prediction quality issue.

## Task-Level MAE

| Task | `none` | `v2.1` | `v2.2` | `v2.3` | `oracle v2.1` |
|---|---:|---:|---:|---:|---:|
| `技能学习规划` | `0.6516` | **`0.6330`** | `0.6486` | `0.6545` | `0.6360` |
| `旅行规划` | `0.7098` | **`0.7045`** | `0.7234` | `0.7229` | `0.7072` |
| `礼物准备` | `0.7637` | `0.7599` | `0.7702` | `0.7691` | **`0.7387`** |
| `菜谱规划` | `0.7004` | **`0.6780`** | `0.6845` | `0.6946` | **`0.6700`** |

`v2.3` only beats `none` on `菜谱规划`. It is worse than `v2.1` on all four
tasks.

## Interpretation

The intended effect partially happened:

- `v2.3` is less SAT-biased than `v2.1`
- it improves global `F1-DSAT` and boundary kappa among non-oracle update variants
- it reduces false SAT relative to `v2.1`

But the cost is too high:

- global MAE becomes worse than `none`
- QWK/correlation remain below `none`
- user-aware continuous metrics remain weak
- task-level MAE regresses almost everywhere
- direct score changes have negative net gain

This suggests the lightweight boundary gate is directionally useful, but the
first-pass score movement induced by the update is not reliable enough. `v2.3`
over-corrects some SAT-side predictions into DSAT, improving the binary boundary
in aggregate while damaging exact 1-5 accuracy.

## Conclusion

`v2.3` should not be used as the next main predictor.

The best current non-oracle update remains:

- `per_session + v2.1` if prioritizing MAE/RMSE and overall stability

If prioritizing SAT/DSAT boundary only, `v2.3` contains a useful signal because
its `F1-DSAT` and boundary kappa are better than `v2.1`; however, it is not good
enough as a full 1-5 predictor.

## Recommended Next Step

Do not continue by stacking more changes onto `v2.3`.

A better route is to keep `v2.1` as the scoring base and borrow only the useful
part of `v2.3`:

1. keep v2.1 memory update prompt and merge behavior for continuous 1-5 scoring
2. add a separate lightweight boundary calibration or second-pass boundary check
3. apply it only when the first-pass prediction is near the 3/4 boundary
4. avoid changing clear `4/5` or `1/2` cases

This would preserve the MAE gain of `v2.1` while selectively recovering the
DSAT awareness that `v2.3` showed.
