# Memory V2.4 Update Results

## Setup

This report evaluates the first `memory_update_prompt_version=v2_4` run.

The run is a 20-user subset, not a full test-set run:

- `detection/outputs/personalized/Qwen_Qwen3-8B_test_per_session_updv2_4_u20.jsonl`
- `1594` turns
- `20` users
- all four target tasks

For fair comparison, existing full-run outputs were filtered to the same
`sample_id` set:

- `detection/outputs/personalized/qwen_v2_4_u20_subset_compare/none.jsonl`
- `detection/outputs/personalized/qwen_v2_4_u20_subset_compare/old_per_session.jsonl`
- `detection/outputs/personalized/qwen_v2_4_u20_subset_compare/v2_1.jsonl`
- `detection/outputs/personalized/qwen_v2_4_u20_subset_compare/v2_2.jsonl`
- `detection/outputs/personalized/qwen_v2_4_u20_subset_compare/v2_3.jsonl`
- `detection/outputs/personalized/qwen_v2_4_u20_subset_compare/v2_4.jsonl`
- `detection/outputs/personalized/qwen_v2_4_u20_subset_compare/oracle_v2_1.jsonl`

Evaluation JSON:

- `detection/outputs/personalized/qwen_v2_4_u20_update_compare.json`

## Main Result

`v2.4` does not achieve the intended effect on this subset.

It slightly improves MAE over `none`, but it is clearly worse than `v2.1`, and
it makes SAT/DSAT boundary behavior worse by becoming more SAT-heavy.

### Global Metrics

| Method | MAE | RMSE | Pearson | Spearman | QWK |
|---|---:|---:|---:|---:|---:|
| `none` | `0.7077` | `1.0031` | `0.2999` | **`0.3012`** | `0.2807` |
| `old_per_session` | `0.7064` | `1.0056` | **`0.3013`** | `0.2932` | **`0.2860`** |
| `v2.1` | **`0.6982`** | **`0.9978`** | `0.2954` | `0.2988` | `0.2755` |
| `v2.2` | `0.7108` | `1.0239` | `0.2640` | `0.2688` | `0.2490` |
| `v2.3` | `0.7221` | `1.0245` | `0.2663` | `0.2697` | `0.2511` |
| `v2.4` | `0.7045` | `1.0134` | `0.2692` | `0.2749` | `0.2526` |
| `oracle_v2.1` | `0.7001` | `1.0031` | `0.2972` | **`0.3049`** | `0.2794` |

Interpretation:

- `v2.4` beats `none` on MAE only: `0.7077 -> 0.7045`.
- It is worse than `v2.1` on MAE/RMSE/correlation/QWK.
- Its QWK and correlations are close to the weaker `v2.2/v2.3` family, not to
  `v2.1`.

This means v2.4 did not preserve the v2.1 global-quality advantage.

## Boundary Metrics

| Method | Acc | F1-SAT | F1-DSAT | Kappa | AUC | False SAT | False DSAT |
|---|---:|---:|---:|---:|---:|---:|---:|
| `none` | `0.7629` | `0.8550` | `0.3505` | `0.2055` | `0.6588` | `0.6495` | `0.1450` |
| `old_per_session` | `0.7748` | `0.8629` | **`0.3691`** | **`0.2321`** | **`0.6660`** | `0.6392` | `0.1328` |
| `v2.1` | **`0.7804`** | **`0.8675`** | `0.3590` | `0.2272` | `0.6525` | `0.6632` | **`0.1205`** |
| `v2.2` | `0.7729` | `0.8626` | `0.3466` | `0.2096` | `0.6464` | `0.6701` | `0.1282` |
| `v2.3` | `0.7641` | `0.8563` | `0.3427` | `0.1990` | `0.6492` | `0.6632` | `0.1404` |
| `v2.4` | `0.7660` | `0.8596` | `0.2976` | `0.1587` | `0.6322` | `0.7285` | `0.1236` |
| `oracle_v2.1` | `0.7660` | `0.8567` | `0.3624` | `0.2191` | `0.6560` | **`0.6357`** | `0.1443` |

This is the main negative result:

- `v2.4` has the worst `F1-DSAT` among all compared variants.
- `v2.4` has the worst boundary kappa.
- `v2.4` has the worst AUC.
- `v2.4` has the highest false SAT rate: `0.7285`.

So the lightweight boundary protection did not improve DSAT. It had the
opposite effect.

Predicted SAT/DSAT counts:

| Method | Pred SAT | Pred DSAT |
|---|---:|---:|
| `none` | `1303` | `291` |
| `old_per_session` | `1316` | `278` |
| `v2.1` | `1339` | `255` |
| `v2.2` | `1331` | `263` |
| `v2.3` | `1313` | `281` |
| `v2.4` | `1354` | `240` |
| `oracle_v2.1` | `1300` | `294` |

`v2.4` predicts the most SAT cases and the fewest DSAT cases. This explains the
DSAT failure.

## User-Aware Metrics

### User-aware 1-5

| Method | PU-Pearson | PU-Spearman | PU-Kappa | WC-Pearson | WC-Spearman |
|---|---:|---:|---:|---:|---:|
| `none` | `0.2007` | `0.1853` | `0.1736` | **`0.2154`** | `0.1678` |
| `old_per_session` | **`0.2041`** | `0.1787` | **`0.1798`** | `0.2136` | `0.1641` |
| `v2.1` | `0.1832` | `0.1677` | `0.1567` | `0.1986` | `0.1613` |
| `v2.2` | `0.1736` | `0.1603` | `0.1486` | `0.1772` | `0.1459` |
| `v2.3` | `0.1757` | `0.1634` | `0.1557` | `0.1856` | `0.1600` |
| `v2.4` | `0.1741` | `0.1645` | `0.1512` | `0.1793` | `0.1549` |
| `oracle_v2.1` | `0.1954` | **`0.1905`** | `0.1712` | `0.2035` | **`0.1848`** |

`v2.4` remains weak on user-aware continuous metrics and is clearly below
`none`, `old_per_session`, and `v2.1`.

### User-aware boundary

| Method | PU-bin Acc | PU-bin F1-SAT | PU-bin F1-DSAT | PU-bin Kappa | WC-bin Pearson |
|---|---:|---:|---:|---:|---:|
| `none` | `0.7507` | `0.8363` | `0.2998` | `0.1520` | `0.1777` |
| `old_per_session` | `0.7629` | `0.8446` | **`0.3244`** | **`0.1876`** | **`0.1848`** |
| `v2.1` | **`0.7670`** | **`0.8460`** | `0.2917` | `0.1507` | `0.1669` |
| `v2.2` | `0.7588` | `0.8431` | `0.2839` | `0.1429` | `0.1526` |
| `v2.3` | `0.7493` | `0.8355` | `0.2778` | `0.1289` | `0.1608` |
| `v2.4` | `0.7520` | `0.8401` | `0.2372` | `0.0988` | `0.1375` |
| `oracle_v2.1` | `0.7500` | `0.8334` | `0.3013` | `0.1516` | `0.1697` |

`v2.4` is also the weakest on user-aware DSAT and kappa. This confirms the
global boundary result.

## Prediction Distribution

| Method | 1 | 2 | 3 | 4 | 5 |
|---|---:|---:|---:|---:|---:|
| `none` | `1` | `23` | `267` | `924` | `379` |
| `old_per_session` | `2` | `23` | `253` | `884` | `432` |
| `v2.1` | `2` | `22` | `231` | `956` | `383` |
| `v2.2` | `3` | `19` | `241` | `911` | `420` |
| `v2.3` | `1` | `21` | `259` | `899` | `414` |
| `v2.4` | `1` | `22` | `217` | `930` | `424` |
| `oracle_v2.1` | `1` | `14` | `279` | `889` | `411` |

`v2.4` has the fewest `3` predictions and the most SAT predictions. It is more
SAT-heavy than v2.1, despite the design goal of boundary protection.

## Direct Change Analysis

Compared with `none`:

- score changed: `357`
- reason changed: `930`
- better: `181`
- worse: `176`
- net gain: `+5`

Top transitions:

- `4 -> 5`: `122`
- `3 -> 4`: `93`
- `5 -> 4`: `77`
- `4 -> 3`: `42`
- `2 -> 3`: `12`
- `3 -> 2`: `11`

Compared with `v2.1`:

- score changed: `316`
- reason changed: `154`
- better: `152`
- worse: `163`
- tie: `1`
- net gain: `-11`

Top transitions:

- `4 -> 5`: `101`
- `3 -> 4`: `70`
- `5 -> 4`: `61`
- `4 -> 3`: `57`
- `2 -> 3`: `12`
- `3 -> 2`: `12`

Compared with `v2.3`:

- score changed: `309`
- better: `168`
- worse: `141`
- net gain: `+27`

This means v2.4 is better than v2.3 on this subset, but still worse than v2.1
and poor on DSAT.

## Task-Level MAE

| Task | `none` | `v2.1` | `v2.3` | `v2.4` | `oracle_v2.1` |
|---|---:|---:|---:|---:|---:|
| `技能学习规划` | `0.6896` | `0.6925` | `0.7224` | `0.7134` | **`0.6836`** |
| `旅行规划` | `0.6631` | **`0.6371`** | `0.6847` | `0.6458` | `0.6739` |
| `礼物准备` | `0.7891` | **`0.7457`** | `0.7587` | **`0.7457`** | `0.7478` |
| `菜谱规划` | **`0.6756`** | `0.7232` | `0.7232` | `0.7202` | `0.6875` |

v2.4 helps `旅行规划` and `礼物准备` relative to `none`, but hurts
`技能学习规划` and `菜谱规划`. It is not a robust task-level improvement.

## Interpretation

The v2.4 design failed in an informative way.

The intent was to keep v2.1's MAE behavior while avoiding boundary over-update.
In practice, requiring adjacent evidence for boundary text updates did not make
the final predictions more DSAT-aware. Instead, the model became even more
SAT-heavy:

- fewer `3` predictions
- more `4/5` predictions
- higher false SAT
- much lower F1-DSAT

This suggests the main source of SAT drift is probably not only verbal boundary
field updates. It may come from:

- updated `scoring_style`
- added/changed task observations
- updated average score/distribution being interpreted as a higher prior
- turn-evaluation prompt consuming memory in a SAT-leaning way

## Conclusion

Do not run full `v2.4` yet unless a larger subset is needed for confirmation.

On this 20-user subset:

- v2.4 is better than v2.3 on MAE
- v2.4 is worse than v2.1 on global metrics
- v2.4 is clearly worse on DSAT/boundary metrics
- v2.4 does not solve the issue it was designed for

The best non-oracle update remains `per_session + v2.1`.

## Recommended Next Step

If continuing from v2.1, avoid modifying only the memory-update boundary text.
The next better experiment should constrain SAT drift more directly:

1. keep v2.1 update prompt
2. freeze or heavily gate `scoring_style`
3. optionally avoid updating `avg_satisfaction_score` upward from predicted
   labels in non-oracle mode
4. evaluate whether this preserves MAE while reducing false SAT

Another option is to leave memory update at v2.1 and handle 3/4 calibration as a
separate post-processing or second-pass decision layer, rather than trying to
encode it into memory updates.
