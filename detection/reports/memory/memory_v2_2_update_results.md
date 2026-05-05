# Memory V2.2 Update Results

## Setup

Goal: test whether `memory_update_prompt_version=v2_2` can improve over `v2.1` by using:

- structured evidence bundle input
- richer patch confidence fields
- harder program-side gates

Compared files:

- `detection/outputs/personalized/Qwen_Qwen3-8B_test_none.jsonl`
- `detection/outputs/personalized/Qwen_Qwen3-8B_test_per_session.jsonl`
- `detection/outputs/personalized/Qwen_Qwen3-8B_test_per_session_updv2_1.jsonl`
- `detection/outputs/personalized/Qwen_Qwen3-8B_test_per_session_updv2_2.jsonl`
- `detection/outputs/personalized/Qwen_Qwen3-8B_test_per_session_oracle_updv2_1.jsonl`

Comparison JSON:

- `detection/outputs/personalized/qwen_v2_2_update_compare.json`

## Main Result

`per_session + v2.2` is a regression relative to `per_session + v2.1`, and it also fails to beat `Qwen none`.

### Global metrics

| Method | MAE | RMSE | Pearson | Spearman | QWK |
|---|---:|---:|---:|---:|---:|
| `Qwen none` | `0.7110` | `1.0122` | `0.2967` | `0.2820` | `0.2815` |
| `Qwen per_session (old)` | `0.7133` | `1.0203` | `0.2812` | `0.2619` | `0.2694` |
| `Qwen per_session + v2.1` | **`0.6997`** | **`0.9992`** | `0.2931` | `0.2763` | `0.2781` |
| `Qwen per_session + v2.2` | `0.7129` | `1.0137` | `0.2769` | `0.2607` | `0.2639` |
| `Qwen per_session_oracle + v2.1` | `0.6934` | `0.9951` | **`0.3045`** | **`0.2931`** | **`0.2889`** |

Compared with `Qwen none`, `per_session + v2.2` is worse on every main global metric:

- `MAE: 0.7110 -> 0.7129`
- `RMSE: 1.0122 -> 1.0137`
- `Pearson: 0.2967 -> 0.2769`
- `Spearman: 0.2820 -> 0.2607`
- `QWK: 0.2815 -> 0.2639`

Compared with `per_session + v2.1`, the regression is even clearer:

- `MAE: 0.6997 -> 0.7129`
- `Pearson: 0.2931 -> 0.2769`
- `QWK: 0.2781 -> 0.2639`

## Boundary Metrics

| Method | Acc | F1-SAT | F1-DSAT | Kappa | AUC | False SAT | False DSAT |
|---|---:|---:|---:|---:|---:|---:|---:|
| `Qwen none` | `0.7555` | `0.8504` | `0.3312` | `0.1824` | `0.6444` | `0.6459` | `0.1617` |
| `Qwen per_session (old)` | `0.7610` | `0.8547` | `0.3271` | `0.1821` | `0.6437` | `0.6603` | `0.1520` |
| `Qwen per_session + v2.1` | **`0.7685`** | **`0.8602`** | `0.3269` | **`0.1871`** | **`0.6448`** | `0.6712` | **`0.1409`** |
| `Qwen per_session + v2.2` | `0.7648` | `0.8575` | `0.3258` | `0.1834` | `0.6411` | `0.6676` | `0.1461` |
| `Qwen per_session_oracle + v2.1` | **`0.7723`** | **`0.8617`** | **`0.3563`** | **`0.2182`** | **`0.6544`** | **`0.6314`** | `0.1444` |

Interpretation:

- `v2.2` does not improve DSAT.
- It is slightly worse than `v2.1` on all main boundary metrics.
- It is also slightly worse than `none` on:
  - `F1-DSAT`
  - `AUC`

So the extra evidence/gating logic did not translate into better SAT/DSAT decisions.

## User-Aware Metrics

### User-aware 1-5

| Method | PU-Pearson | PU-Spearman | PU-Kappa | WC-Pearson | WC-Spearman |
|---|---:|---:|---:|---:|---:|
| `Qwen none` | `0.1997` | `0.1798` | `0.1646` | `0.2154` | `0.1637` |
| `Qwen per_session + v2.1` | `0.2001` | `0.1789` | `0.1624` | `0.2139` | `0.1646` |
| `Qwen per_session + v2.2` | `0.1853` | `0.1623` | `0.1523` | `0.1987` | `0.1498` |

### User-aware boundary

| Method | PU-bin Acc | PU-bin F1-SAT | PU-bin F1-DSAT | PU-bin Kappa | WC-bin Pearson |
|---|---:|---:|---:|---:|---:|
| `Qwen none` | `0.7430` | `0.8244` | `0.2734` | `0.1196` | `0.1529` |
| `Qwen per_session + v2.1` | **`0.7543`** | **`0.8342`** | `0.2647` | `0.1195` | **`0.1621`** |
| `Qwen per_session + v2.2` | `0.7490` | `0.8313` | `0.2662` | `0.1203` | `0.1527` |

Interpretation:

- `v2.2` loses the user-aware gains that `v2.1` had on the SAT side.
- Compared with `none`, user-aware continuous metrics all drop.
- Compared with `v2.1`, user-aware binary metrics also regress overall.

## How Much Did V2.2 Actually Change?

Relative to `Qwen none`:

- score changed: `1429`
- reason changed: `3733`
- better: `702`
- worse: `727`
- net gain: `-25`

For comparison:

- old `per_session`: net `-17`
- `per_session + v2.1`: net `+52`

This is the most important direct diagnosis:

`v2.2` is not a no-op, but its changes are again slightly net harmful, much closer to the old failing update pattern than to `v2.1`.

Most common score transitions for `v2.2`:

- `4 -> 5`: `420`
- `5 -> 4`: `335`
- `3 -> 4`: `319`
- `4 -> 3`: `224`

Compared with `v2.1`, `v2.2` shows:

- more `4 -> 5`
- fewer `3 -> 4`
- fewer useful net corrections overall

This suggests the new structure/gating did not sharpen boundary learning; instead it appears to have weakened some of the helpful updates while still allowing enough noisy movement to hurt net accuracy.

## Prediction Distribution

Predicted score counts for `per_session + v2.2`:

- `1`: `31`
- `2`: `125`
- `3`: `996`
- `4`: `3710`
- `5`: `1612`

Compared with `none`:

- fewer `3`
- more `5`

Compared with `v2.1`:

- slightly more `3`
- slightly fewer `4`
- more `5`

The overall shape is still SAT-leaning and does not explain the regression by DSAT recovery. Instead it looks more like noisier internal reshuffling of `3/4/5`.

## Reason Consistency

Reason legality remains correct:

- `pred_score >= 4` and `reason != 满意`: `0`
- `pred_score <= 3` and `reason == 满意`: `0`

So the regression is not caused by reason-label inconsistency. It is a genuine prediction-quality regression.

## Task-Level MAE

Per-task MAE:

| Task | `none` | `per_v2.1` | `per_v2.2` |
|---|---:|---:|---:|
| `技能学习规划` | `0.6516` | **`0.6330`** | `0.6486` |
| `旅行规划` | **`0.7098`** | **`0.7045`** | `0.7234` |
| `礼物准备` | `0.7637` | **`0.7599`** | `0.7702` |
| `菜谱规划` | `0.7004` | **`0.6780`** | `0.6845` |

`v2.2` only remains slightly better than `none` on:

- `技能学习规划`
- `菜谱规划`

But it is worse than `none` on:

- `旅行规划`
- `礼物准备`

And it is worse than `v2.1` on all four tasks.

## Why V2.2 Likely Failed

Based on the observed metrics and change patterns, the most plausible interpretation is:

1. **The structured evidence bundle lost useful nuance**
   - `v2.1` still let the model see enough raw local context to form useful patch updates.
   - `v2.2` compresses this into a more abstract summary, which may remove the very clues that were helping `v2.1`.

2. **The harder gates were not selective in the right way**
   - They did not reduce update volume enough to prevent harmful changes.
   - But they may have filtered out some of the more helpful boundary refinements.

3. **Non-oracle uncertainty modeling did not convert into better boundary behavior**
   - If it had worked, we would expect better `F1-DSAT` / `false_sat_rate`.
   - Instead, those metrics are flat or worse.

In short:

`v2.2` became more principled, but less effective.

## Conclusions

1. `v2.1` remains the best non-oracle update version.
2. `v2.2` is a regression:
   - worse than `v2.1`
   - slightly worse than `none`
3. The current evidence strongly suggests that:
   - field-level patching was the key good idea in `v2.1`
   - the extra evidence abstraction / harder gating in `v2.2` over-corrected and removed useful signal

## Recommended Next Step

Do **not** continue directly from `v2.2`.

If update is to be improved further, the better path is likely:

1. start again from `v2.1`
2. add only one small change at a time
3. prioritize preserving raw local evidence, instead of replacing it with heavily compressed structured summaries
