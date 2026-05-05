# URS Qwen3-8B None Calibration Results

## Setup

Base prediction file:

- `detection/outputs/urs/Qwen_Qwen3-8B_test_none_rerun.jsonl`

Calibrated files:

- Mean shift: `detection/outputs/urs/Qwen_Qwen3-8B_test_none_rerun_calMS.jsonl`
- CDF: `detection/outputs/urs/Qwen_Qwen3-8B_test_none_rerun_calCDF.jsonl`

Comparison JSON:

- `detection/outputs/urs/Qwen_Qwen3-8B_test_none_rerun_calibration_comparison.json`

Calibration command reused the existing post-hoc pipeline with:

- `memory_cache_dir=outputs/urs/memory_cache`

## Calibration Coverage

Calibration barely applied on URS:

- Total blocks: `339`
- Actually calibrated blocks: `20`
- Fallback identity blocks: `319`

Fallback reasons:

- `thin_history`: `283`
- `small_block`: `36`

Changed predictions:

- Mean shift: `28 / 584`
- CDF: `34 / 584`

This is the key structural reason calibration has limited effect on URS: most URS blocks do not have enough usable history under the current calibration assumptions.

## Main Comparison

### Global Metrics

| Method | MAE | RMSE | Pearson | Spearman | QWK |
|---|---:|---:|---:|---:|---:|
| Raw | `0.8887` | `1.2086` | `0.2830` | `0.2781` | `0.2498` |
| Mean shift | `0.8973` | `1.2135` | `0.2707` | `0.2704` | `0.2440` |
| CDF | `0.9110` | `1.2345` | `0.2536` | `0.2551` | `0.2291` |

### Boundary Metrics

| Method | Acc | F1-SAT | F1-DSAT | Kappa | AUC | False SAT | False DSAT |
|---|---:|---:|---:|---:|---:|---:|---:|
| Raw | `0.5702` | `0.5958` | `0.5411` | `0.1821` | `0.6304` | `0.2780` | `0.5119` |
| Mean shift | `0.5771` | `0.6110` | `0.5366` | `0.1841` | `0.6308` | `0.3024` | `0.4881` |
| CDF | `0.5685` | `0.6025` | `0.5281` | `0.1684` | `0.6142` | `0.3122` | `0.4960` |

### User-Aware Metrics

| Method | PU-Pearson | PU-Spearman | PU-Kappa | WC-Pearson | WC-Spearman |
|---|---:|---:|---:|---:|---:|
| Raw | `-0.2851` | `-0.2989` | `-0.0556` | `-0.0974` | `-0.0989` |
| Mean shift | `-0.3158` | `-0.3245` | `-0.0706` | `-0.1232` | `-0.1227` |
| CDF | `-0.3450` | `-0.3504` | `-0.0923` | `-0.1379` | `-0.1386` |

### User-Aware Boundary Metrics

| Method | PU-bin Acc | PU-bin F1-SAT | PU-bin F1-DSAT | PU-bin Kappa | WC-bin Pearson |
|---|---:|---:|---:|---:|---:|
| Raw | `0.4527` | `0.2786` | `0.4138` | `-0.1046` | `-0.1344` |
| Mean shift | `0.4627` | `0.3040` | `0.4047` | `-0.1021` | `-0.1365` |
| CDF | `0.4502` | `0.2928` | `0.3962` | `-0.1266` | `-0.1758` |

## Interpretation

### 1. Calibration does not improve the main global objective on URS

Both calibration methods make the global metrics worse:

- higher MAE / RMSE
- lower Pearson / Spearman
- lower QWK

This is the opposite of what was observed on the personalized benchmark.

### 2. Mean shift gives a tiny boundary tradeoff, but not a real win

Mean shift slightly improves:

- boundary accuracy: `0.5702 -> 0.5771`
- boundary kappa: `0.1821 -> 0.1841`
- false DSAT: `0.5119 -> 0.4881`

But it also worsens:

- F1-DSAT: `0.5411 -> 0.5366`
- false SAT: `0.2780 -> 0.3024`
- all major global metrics

So mean shift mostly shifts the SAT/DSAT balance a little, rather than producing a genuine across-the-board improvement.

### 3. CDF is clearly not suitable here

CDF degrades almost everything:

- global metrics
- boundary metrics
- user-aware metrics

The likely reason is that the history distribution available per URS block is too sparse, so rank remapping is driven by weak or unstable empirical CDFs.

### 4. The real blocker is low calibration coverage

Only `20 / 339` blocks were calibrated at all. Most blocks fell back to identity because:

- URS history is too thin for the current threshold
- many blocks are too small

So the current post-hoc calibration design is structurally mismatched to URS.

## Conclusions

1. Post-hoc calibration is useful on the personalized benchmark, but **does not transfer cleanly to URS** under the current setup.
2. For URS `Qwen3-8B none`, **raw predictions are still the best default among raw / mean_shift / cdf**.
3. If calibration is revisited on URS, it should probably not reuse the personalized block-level assumptions unchanged.

## Next Directions

More promising next steps for URS are:

1. Compare `no_memory` vs `none` first, to verify whether memory itself is helping.
2. If calibration is retried, use a URS-specific variant:
   - lower `min_history_turns`
   - consider session-level user mean calibration instead of block-level CDF
   - possibly calibrate at user level instead of `(user, task, model)` block level
3. Treat the current URS issue primarily as a distribution-bias problem, not just a post-hoc scaling problem.
