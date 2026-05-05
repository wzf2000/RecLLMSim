# URS Qwen3-8B None Results

## Setup

- Dataset: `URS`
- Model: `Qwen/Qwen3-8B`
- Memory update mode: `none`
- Output file: `detection/outputs/urs/Qwen_Qwen3-8B_test_none_rerun.jsonl`
- Eval file: `detection/outputs/urs/Qwen_Qwen3-8B_test_none_rerun_eval.json`

This run was produced after fixing the URS pipeline to use session-level memory building and updating instead of the personalized turn-level logic. See `detection/reports/pipeline/urs_session_level_memory_fix.md`.

## Data Coverage

- Total samples: `584`
- Total users: `116`

Predicted score distribution:

- `1`: `7`
- `2`: `105`
- `3`: `230`
- `4`: `209`
- `5`: `33`

Gold score distribution:

- `1`: `19`
- `2`: `47`
- `3`: `139`
- `4`: `250`
- `5`: `129`

Binary SAT/DSAT distribution:

- Gold SAT (`>=4`): `379`
- Gold DSAT (`<=3`): `205`
- Pred SAT (`>=4`): `242`
- Pred DSAT (`<=3`): `342`

The model clearly predicts too many dissatisfied sessions on URS.

## Global Metrics

- MAE: `0.8887`
- RMSE: `1.2086`
- Pearson: `0.2830`
- Spearman: `0.2781`
- QWK: `0.2498`

Interpretation:

- Correlation is moderate but not strong.
- QWK is non-trivial, so the model is not random.
- Error remains fairly high, especially compared with the personalized main benchmark.

## Boundary Metrics

- Accuracy: `0.5702`
- F1-macro: `0.5685`
- F1-SAT: `0.5958`
- F1-DSAT: `0.5411`
- Boundary kappa: `0.1821`
- AUC: `0.6304`
- False SAT rate: `0.2780`
- False DSAT rate: `0.5119`

Interpretation:

- The model is relatively willing to call sessions dissatisfied.
- This yields a fairly low false-SAT rate, meaning many true dissatisfied sessions are caught.
- But the tradeoff is severe overprediction of dissatisfaction, reflected in the very high false-DSAT rate.

In other words, the current URS `Qwen3-8B none` run is DSAT-biased rather than SAT-biased.

## User-Aware Metrics

### User-aware 1-5

- PU-Pearson: `-0.2851`
- PU-Spearman: `-0.2989`
- PU-Kappa: `-0.0556`
- WC-Pearson: `-0.0974`
- WC-Spearman: `-0.0989`

### User-aware boundary

- PU-bin Accuracy: `0.4527`
- PU-bin F1-SAT: `0.2786`
- PU-bin F1-DSAT: `0.4138`
- PU-bin Kappa: `-0.1046`
- WC-bin Pearson: `-0.1344`

Interpretation:

- The user-aware metrics are poor and even negative on several correlation measures.
- This suggests that on URS, the current system does not preserve user-specific relative preference structure well.
- The model is doing something globally non-random, but the user-level ordering/calibration is weak.

## Task-Level Observations

Selected task-wise global metrics:

- `text`: MAE `0.7209`, QWK `0.3241`
- `creative`: MAE `0.8438`, QWK `0.2932`
- `professional`: MAE `0.8547`, QWK `0.3214`
- `advice`: MAE `0.9167`, QWK `0.1843`
- `retrieval`: MAE `0.9758`, QWK `0.2882`
- `leisure`: MAE `1.0000`, QWK `-0.0314`

Initial takeaways:

- `text` is currently the easiest task by MAE.
- `retrieval` has decent correlation but still large absolute error.
- `leisure` is the weakest task and appears especially unstable.
- `other` has only `8` samples and should not be over-interpreted.

## Main Conclusions

1. The rerun succeeded after the URS session-level memory fix, so the previous failures were caused by pipeline logic rather than bad data.
2. `Qwen3-8B none` on URS shows moderate global signal but large absolute error.
3. The current prediction pattern is strongly DSAT-biased:
   - too many predicted `2/3`
   - too few predicted `4/5`
4. User-aware performance is currently weak, so URS appears harder than the personalized benchmark for preserving user-specific preference structure.
5. If URS is going to be a serious secondary benchmark, the next comparisons should prioritize:
   - `no_memory` vs `none`
   - `none` vs calibrated variants
   - whether a task-level or global post-hoc calibration can reduce the current DSAT bias
