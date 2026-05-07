# History Prior Delta V2 Full Results

## Run

Result file:

```text
detection/outputs/personalized/history_prior_delta_v2_none_n3.jsonl
```

Generated post-hoc calibration files:

```text
detection/outputs/personalized/history_prior_delta_v2_none_n3_calMS.jsonl
detection/outputs/personalized/history_prior_delta_v2_none_n3_calCDF.jsonl
```

Evaluation output:

```text
detection/outputs/personalized/history_prior_delta_v2_full_compare.json
```

Configuration inferred from records:

- model: `Qwen/Qwen3-8B`
- users: 90
- samples: 6474
- memory update: `none`
- memory version: `v2`
- anchors: `n_anchors=3`
- turn prompt: `history_prior_delta_v2`

The result file has the same 6474 rows as the full history baseline files, so
the structured parse route fixed the earlier raw-text parse failure pattern for
this full run.

## Main Comparison

| method | MAE | RMSE | Pearson | Spearman | QWK | Boundary Acc | F1-DSAT | False SAT | False DSAT |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen v2 none | 0.7110 | 1.0122 | 0.2967 | 0.2820 | 0.2815 | 0.7555 | 0.3312 | 0.6459 | 0.1617 |
| Qwen v2 none + CDF | 0.6355 | 1.0191 | 0.3601 | 0.3716 | 0.3595 | 0.7930 | 0.3655 | 0.6513 | 0.1153 |
| user history mean | 0.5661 | 0.8870 | 0.3560 | 0.3917 | 0.3105 | 0.8217 | 0.2255 | 0.8482 | 0.0401 |
| user history median | **0.5542** | 0.9150 | 0.3526 | **0.3972** | 0.3075 | **0.8256** | 0.1643 | 0.8997 | **0.0248** |
| nearest history k3 | 0.6285 | 0.9798 | 0.3076 | 0.3406 | 0.2974 | 0.7916 | 0.2963 | 0.7435 | 0.0980 |
| history prior delta v2 | 0.5658 | 0.8904 | 0.3598 | 0.3835 | 0.3166 | 0.8221 | 0.2055 | 0.8654 | 0.0361 |
| history prior delta v2 + mean shift | 0.5749 | 0.8983 | 0.3555 | 0.3826 | 0.3198 | 0.8140 | 0.2370 | 0.8311 | 0.0529 |
| history prior delta v2 + CDF | 0.6540 | 1.0428 | 0.3293 | 0.3450 | **0.3287** | 0.7919 | **0.3607** | 0.6567 | 0.1155 |

## Internal Diagnostics

Prediction distribution:

```text
gold_score: 1=123, 2=251, 3=733, 4=2449, 5=2918
pred_score: 2=7, 3=336, 4=4038, 5=2093
raw_classification: 1=7, 2=11, 3=387, 4=4811, 5=1258
```

Judge field distribution:

```text
delta_label: below=512, around=4875, above=1087
delta_score: -2=64, -1=451, 0=4872, 1=1060, 2=27
delta_confidence: medium=5791, high=683
boundary_score: 3=401, 4=6073
boundary_confidence: low=2, medium=5750, high=722
strong_failure_evidence: true=71
strong_excellence_evidence: true=754
```

Only 398 / 6474 samples (6.15%) changed from `round(history_prior_score)` after
soft reconstruction. This explains the main behavior: v2 is effectively a
history-prior estimator with a small number of LLM-guided corrections.

Confusion matrix, rows are gold 1-5 and columns are predicted 1-5:

```text
[
  [0, 6, 21, 81, 15],
  [0, 1, 50, 181, 19],
  [0, 0, 71, 571, 91],
  [0, 0, 142, 1834, 473],
  [0, 0, 52, 1371, 1495],
]
```

The model almost never predicts scores 1 or 2. Most true DSAT samples are still
mapped to 4 or 5, which causes `False SAT=0.8654`.

## Offline Reconstruction Variants

These variants were computed from the same v2 diagnostic fields without extra
LLM calls:

| variant | MAE | Pearson | Spearman | QWK | Boundary Acc | F1-DSAT | pred DSAT | False SAT |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| current final | 0.5658 | 0.3598 | 0.3835 | 0.3166 | 0.8221 | 0.2055 | 343 | 0.8654 |
| prior round only | 0.5734 | 0.3293 | 0.3702 | 0.2844 | 0.8205 | 0.1953 | 337 | 0.8726 |
| raw classification | 0.5968 | 0.3554 | 0.3468 | 0.2978 | 0.8261 | 0.2553 | 405 | 0.8257 |
| prior + all delta | 0.5913 | 0.3679 | 0.3666 | **0.3499** | 0.8170 | 0.2891 | 560 | 0.7823 |
| prior + sign(delta) | 0.5908 | 0.3582 | 0.3645 | 0.3368 | 0.8162 | 0.2770 | 539 | 0.7940 |
| prior + high delta only | 0.5672 | 0.3431 | 0.3770 | 0.2996 | 0.8228 | 0.1928 | 314 | 0.8762 |
| prior + all boundary constraints | **0.5656** | **0.3773** | **0.3906** | 0.3293 | **0.8261** | 0.2533 | 401 | 0.8275 |
| boundary score only | 0.7014 | 0.2583 | 0.2063 | 0.1174 | 0.8261 | 0.2533 | 401 | 0.8275 |

The LLM diagnostic fields do contain some useful ordering signal: applying all
boundary constraints or all delta information improves QWK / Pearson / DSAT
relative to pure prior. However, the current confidence-gated reconstruction is
too conservative to expose much of that signal.

## Interpretation

The full run changes the conclusion from the v1 subset:

- V2 fixes exact-score MAE by anchoring strongly to user history. It nearly
  matches `user_history_mean` (`0.5658` vs `0.5661`) and is close to
  `user_history_median` (`0.5542`).
- The improvement over pure prior is tiny on MAE: per-sample comparison against
  `user_history_mean` gives 488 wins, 488 losses, and 5498 ties.
- Boundary detection regresses compared with Qwen v2 and calibrated Qwen v2.
  `F1-DSAT=0.2055` is below Qwen v2 (`0.3312`), Qwen v2 + CDF (`0.3655`), and
  nearest-history k3 (`0.2963`).
- CDF calibration recovers DSAT (`F1-DSAT=0.3607`) and QWK (`0.3287`), but it
  worsens exact-score MAE to `0.6540`. For v2, CDF is a boundary/ranking
  post-processing option, not an exact-score improvement.

## Next Direction

The method is now too prior-dominated. The next improvement should not further
increase confidence gating. It should expose residual/boundary evidence in a
controlled way:

- keep history prior as the exact-score anchor for MAE
- use `boundary_score=3` or raw `classification<=3` as a soft DSAT discovery
  signal, not only high-confidence boundary
- consider a two-output setup: one exact score optimized for MAE and one SAT/DSAT
  score optimized for boundary metrics
- learn or tune a small post-hoc rule over `prior_score`, `delta_score`,
  `boundary_score`, `raw_classification`, and confidence fields on a validation
  split, instead of hand-setting confidence thresholds

