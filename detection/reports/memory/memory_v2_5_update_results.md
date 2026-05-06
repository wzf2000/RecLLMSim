# Memory V2.5 Update Results

## Setup

This report evaluates the first `memory_update_prompt_version=v2_5` run.

The run is a 20-user subset:

- `detection/outputs/personalized/Qwen_Qwen3-8B_test_per_session_updv2_5_u20.jsonl`
- `1594` turns
- `20` users
- all four target tasks

For fair comparison, existing outputs were filtered to the same `sample_id`
set:

- `detection/outputs/personalized/qwen_v2_5_u20_subset_compare/none.jsonl`
- `detection/outputs/personalized/qwen_v2_5_u20_subset_compare/old_per_session.jsonl`
- `detection/outputs/personalized/qwen_v2_5_u20_subset_compare/v2_1.jsonl`
- `detection/outputs/personalized/qwen_v2_5_u20_subset_compare/v2_3.jsonl`
- `detection/outputs/personalized/qwen_v2_5_u20_subset_compare/v2_4.jsonl`
- `detection/outputs/personalized/qwen_v2_5_u20_subset_compare/v2_5.jsonl`
- `detection/outputs/personalized/qwen_v2_5_u20_subset_compare/oracle_v2_1.jsonl`

Evaluation JSON:

- `detection/outputs/personalized/qwen_v2_5_u20_update_compare.json`

## Main Result

`v2.5` fixes the worst boundary failure of `v2.4`, but it does not recover the
global 1-5 prediction quality of `v2.1`.

It should not be promoted to a full run as-is.

### Global Metrics

| Method | MAE | RMSE | Pearson | Spearman | QWK |
|---|---:|---:|---:|---:|---:|
| `none` | `0.7077` | `1.0031` | `0.2999` | `0.3012` | `0.2807` |
| `old_per_session` | `0.7064` | `1.0056` | **`0.3013`** | `0.2932` | **`0.2860`** |
| `v2.1` | **`0.6982`** | **`0.9978`** | `0.2954` | `0.2988` | `0.2755` |
| `v2.3` | `0.7221` | `1.0245` | `0.2663` | `0.2697` | `0.2511` |
| `v2.4` | `0.7045` | `1.0134` | `0.2692` | `0.2749` | `0.2526` |
| `v2.5` | `0.7158` | `1.0183` | `0.2710` | `0.2735` | `0.2543` |
| `oracle_v2.1` | `0.7001` | `1.0031` | `0.2972` | **`0.3049`** | `0.2794` |

Global interpretation:

- `v2.5` is worse than `none` on MAE/RMSE/correlation/QWK.
- `v2.5` is clearly worse than `v2.1`.
- It is only better than `v2.3` on MAE/RMSE and similar to the weak v2.3/v2.4
  family on correlation/QWK.

The scoring-style freeze and upward-average damping did not preserve the v2.1
MAE benefit.

## Boundary Metrics

| Method | Acc | F1-SAT | F1-DSAT | Kappa | AUC | False SAT | False DSAT |
|---|---:|---:|---:|---:|---:|---:|---:|
| `none` | `0.7629` | `0.8550` | `0.3505` | `0.2055` | **`0.6588`** | `0.6495` | `0.1450` |
| `old_per_session` | `0.7748` | `0.8629` | **`0.3691`** | **`0.2321`** | **`0.6660`** | **`0.6392`** | `0.1328` |
| `v2.1` | **`0.7804`** | **`0.8675`** | `0.3590` | `0.2272` | `0.6525` | `0.6632` | **`0.1205`** |
| `v2.3` | `0.7641` | `0.8563` | `0.3427` | `0.1990` | `0.6492` | `0.6632` | `0.1404` |
| `v2.4` | `0.7660` | `0.8596` | `0.2976` | `0.1587` | `0.6322` | `0.7285` | `0.1236` |
| `v2.5` | `0.7710` | `0.8605` | `0.3608` | `0.2214` | `0.6475` | `0.6460` | `0.1358` |
| `oracle_v2.1` | `0.7660` | `0.8567` | `0.3624` | `0.2191` | `0.6560` | `0.6357` | `0.1443` |

Boundary interpretation:

- `v2.5` successfully repairs the severe `v2.4` DSAT collapse.
- `v2.5` has better `F1-DSAT` than `none`, `v2.1`, `v2.3`, and `v2.4`.
- `v2.5` has lower false SAT than `v2.1` and much lower false SAT than `v2.4`.
- But `v2.5` still has worse boundary accuracy and F1-SAT than `v2.1`.
- AUC remains lower than `none`, `old_per_session`, `v2.1`, and `oracle_v2.1`.

So v2.5 is a boundary-balance improvement over v2.4, but it is not a full
predictor improvement.

## Prediction Distribution

| Method | 1 | 2 | 3 | 4 | 5 |
|---|---:|---:|---:|---:|---:|
| `none` | `1` | `23` | `267` | `924` | `379` |
| `old_per_session` | `2` | `23` | `253` | `884` | `432` |
| `v2.1` | `2` | `22` | `231` | `956` | `383` |
| `v2.3` | `1` | `21` | `259` | `899` | `414` |
| `v2.4` | `1` | `22` | `217` | `930` | `424` |
| `v2.5` | `2` | `16` | `262` | `912` | `402` |
| `oracle_v2.1` | `1` | `14` | `279` | `889` | `411` |

This is the most positive signal for v2.5:

- v2.5 no longer has v2.4's extreme SAT-heavy distribution.
- v2.5 recovers the number of `3` predictions close to `none`.
- v2.5 has a distribution closer to `oracle_v2.1` than v2.4 does.

But distribution repair did not translate into better exact 1-5 accuracy.

## Direct Change Analysis

Compared with `none`:

- score changed: `381`
- reason changed: `931`
- better: `184`
- worse: `197`
- net gain: `-13`

Top transitions:

- `4 -> 5`: `116`
- `5 -> 4`: `93`
- `3 -> 4`: `82`
- `4 -> 3`: `72`
- `2 -> 3`: `11`
- `3 -> 2`: `5`

Compared with `v2.1`:

- score changed: `308`
- reason changed: `175`
- better: `139`
- worse: `168`
- tie: `1`
- net gain: `-29`

Top transitions:

- `4 -> 5`: `91`
- `4 -> 3`: `76`
- `5 -> 4`: `72`
- `3 -> 4`: `49`
- `2 -> 3`: `9`
- `3 -> 2`: `5`

Compared with `v2.4`:

- score changed: `319`
- better: `151`
- worse: `168`
- net gain: `-17`

This confirms the core issue: v2.5 improves boundary balance but its exact-score
moves are still net harmful.

## User-Aware Metrics

### User-aware 1-5

| Method | PU-Pearson | PU-Spearman | PU-Kappa | WC-Pearson | WC-Spearman |
|---|---:|---:|---:|---:|---:|
| `none` | `0.2007` | `0.1853` | `0.1736` | **`0.2154`** | `0.1678` |
| `old_per_session` | **`0.2041`** | `0.1787` | **`0.1798`** | `0.2136` | `0.1641` |
| `v2.1` | `0.1832` | `0.1677` | `0.1567` | `0.1986` | `0.1613` |
| `v2.3` | `0.1757` | `0.1634` | `0.1557` | `0.1856` | `0.1600` |
| `v2.4` | `0.1741` | `0.1645` | `0.1512` | `0.1793` | `0.1549` |
| `v2.5` | `0.1677` | `0.1564` | `0.1462` | `0.1754` | `0.1475` |
| `oracle_v2.1` | `0.1954` | **`0.1905`** | `0.1712` | `0.2035` | **`0.1848`** |

`v2.5` is the weakest variant on user-aware continuous metrics in this subset.

### User-aware boundary

| Method | PU-bin Acc | PU-bin F1-SAT | PU-bin F1-DSAT | PU-bin Kappa | WC-bin Pearson |
|---|---:|---:|---:|---:|---:|
| `none` | `0.7507` | `0.8363` | `0.2998` | `0.1520` | `0.1777` |
| `old_per_session` | `0.7629` | `0.8446` | **`0.3244`** | **`0.1876`** | **`0.1848`** |
| `v2.1` | **`0.7670`** | **`0.8460`** | `0.2917` | `0.1507` | `0.1669` |
| `v2.3` | `0.7493` | `0.8355` | `0.2778` | `0.1289` | `0.1608` |
| `v2.4` | `0.7520` | `0.8401` | `0.2372` | `0.0988` | `0.1375` |
| `v2.5` | `0.7575` | `0.8398` | `0.2991` | `0.1509` | `0.1498` |
| `oracle_v2.1` | `0.7500` | `0.8334` | `0.3013` | `0.1516` | `0.1697` |

v2.5 recovers user-aware boundary F1-DSAT to roughly the none/oracle level, but
does not beat old per-session and does not improve within-user binary Pearson.

## Task-Level MAE

| Task | `none` | `v2.1` | `v2.4` | `v2.5` | `oracle_v2.1` |
|---|---:|---:|---:|---:|---:|
| `技能学习规划` | `0.6896` | `0.6925` | `0.7134` | `0.7164` | **`0.6836`** |
| `旅行规划` | `0.6631` | **`0.6371`** | `0.6458` | `0.6717` | `0.6739` |
| `礼物准备` | `0.7891` | **`0.7457`** | **`0.7457`** | `0.7630` | `0.7478` |
| `菜谱规划` | **`0.6756`** | `0.7232` | `0.7202` | `0.7113` | `0.6875` |

v2.5 only beats none on `礼物准备`. It is worse than v2.1 on all four tasks.

## Reason Consistency

Reason legality is correct:

- `pred_score >= 4` and `reason_prediction != 满意`: `0`
- `pred_score <= 3` and `reason_prediction == 满意`: `0`

## Interpretation

The v2.5 hypothesis was partially correct:

- freezing non-oracle `scoring_style`
- damping upward average-score movement
- filtering generic requirements

did reduce the extreme SAT-heavy behavior seen in v2.4.

However, the resulting score movements are not accurate enough. v2.5 improves
boundary balance but damages exact 1-5 prediction and user-aware continuous
metrics. This suggests that controlling memory-level priors alone is not enough.

The remaining issue is likely in how the turn-evaluation prompt consumes updated
memory. Once memory changes, the evaluator may still perform broad score
reshuffling across `3/4/5`, even if the prior is damped.

## Conclusion

Do not run full `v2.5` as-is.

Current ranking for this subset:

- best MAE: `v2.1`
- best boundary balance: `old_per_session` / `v2.5` depending on metric
- best overall practical choice remains `v2.1`

v2.5 is useful diagnostically because it confirms that SAT drift can be reduced,
but it does not solve the exact-score degradation problem.

## Recommended Next Step

Stop adding complexity to memory update for now. The more promising route is:

1. keep memory update at `v2.1`
2. add an explicit post-prediction boundary calibration layer for near-3/4 cases
3. treat calibration as separate from memory update
4. evaluate raw v2.1 and calibrated v2.1 side by side

This keeps the strongest MAE base while allowing targeted SAT/DSAT adjustment
without perturbing all 1-5 predictions through memory updates.
