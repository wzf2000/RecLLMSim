# History Prior Delta Judge Subset20 Results

## Run

Result file:

```text
detection/outputs/personalized/history_prior_delta_none_n3_limit20.jsonl
```

Evaluation output:

```text
detection/outputs/personalized/history_prior_delta_u20_compare.json
```

Configuration inferred from records:

- model: `Qwen/Qwen3-8B`
- users: 20
- samples: 1594
- memory update: `none`
- memory version: `v2`
- anchors: `n_anchors=3`
- turn prompt: `history_prior_delta`

## Main Metrics

Aligned comparison on the same 1594 sample ids:

| method | MAE | RMSE | Pearson | Spearman | QWK | Boundary Acc | F1-DSAT | DSAT Recall | False SAT | AUC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| history_prior_delta | 0.7215 | 1.0430 | 0.3131 | 0.2900 | 0.3084 | 0.7760 | 0.3521 | 0.3333 | 0.6667 | 0.6720 |
| Qwen v2.5 update | 0.7158 | 1.0183 | 0.2710 | 0.2735 | 0.2543 | 0.7710 | 0.3608 | 0.3540 | 0.6460 | 0.6475 |
| Qwen v2.4 update | 0.7045 | 1.0134 | 0.2692 | 0.2749 | 0.2526 | 0.7660 | 0.2976 | 0.2715 | 0.7285 | 0.6322 |
| boundary v4 | 0.7848 | 1.0251 | 0.2173 | 0.2075 | 0.1391 | 0.7422 | 0.3445 | 0.3711 | 0.6289 | 0.5981 |
| user_history_mean | 0.6198 | 0.9511 | 0.2965 | 0.3370 | 0.2467 | 0.8074 | 0.1303 | 0.0790 | 0.9210 | 0.6548 |
| user_history_median | 0.5772 | 0.9701 | 0.3365 | 0.3988 | 0.2548 | 0.8174 | 0.0000 | 0.0000 | 1.0000 | 0.6710 |
| nearest_history_turn_k3 | 0.6826 | 1.0478 | 0.2675 | 0.3176 | 0.2580 | 0.7647 | 0.2884 | 0.2612 | 0.7388 | 0.6517 |

## Diagnostics

Prediction distributions:

```text
gold: 1=38, 2=72, 3=181, 4=611, 5=692
pred: 2=68, 3=192, 4=695, 5=639
delta_score: -2=50, -1=503, 0=474, 1=551, 2=16
boundary_score: 3=260, 4=1334
```

Residual signal:

```text
history_prior_score mean = 4.1851
gold - prior mean = -0.0264, std = 0.8878
delta_score mean = -0.0125, std = 0.9092
corr(delta_score, gold - prior): Pearson = 0.2206, Spearman = 0.1972
```

Reconstruction variants:

| variant | MAE | Boundary Acc | F1-DSAT | DSAT Recall |
|---|---:|---:|---:|---:|
| raw LLM classification | 0.6819 | 0.7742 | 0.3571 | 0.3436 |
| prior round only | 0.5916 | 0.8124 | 0.2027 | 0.1306 |
| prior + delta only | 0.7654 | 0.7346 | 0.3878 | 0.4605 |
| current prior + delta + both boundary bounds | 0.7215 | 0.7760 | 0.3521 | 0.3333 |

## Interpretation

The new formulation is directionally useful, but not yet a clear improvement.

Positive signals:

- It improves ranking / agreement over Qwen v2.4/v2.5: Pearson, Spearman, QWK, and boundary AUC are all higher than v2.4/v2.5.
- It beats `nearest_history_turn_k3` on QWK, boundary F1-DSAT, DSAT recall, and AUC, so it is not just copying retrieval anchors.
- It detects more dissatisfied turns than history mean / median, whose DSAT recall is very low or zero.

Problems:

- Full-score MAE is worse than simple history priors. `user_history_median` has MAE 0.5772 and `user_history_mean` has MAE 0.6198, while this method is 0.7215.
- The predicted residual is weakly correlated with the true residual: Pearson 0.2206. This means the residual judge has signal, but it is too noisy to replace the history prior as a score estimator.
- The code-side reconstruction rule hurts exact score MAE. Raw LLM `classification` has MAE 0.6819, better than reconstructed 0.7215.
- The hard boundary is not reliable enough: among `boundary_score=3` samples, only 37.3% are actually DSAT, so forcing SAT turns down to <=3 introduces many false DSAT errors.
- False SAT remains high at 0.6667, so the method still misses two thirds of true dissatisfied turns.

## Conclusion

This run supports the overall direction but not the current scoring rule.

The method adds useful semantic/ranking signal beyond history-only and nearest-history baselines, but the residual delta is too noisy and the hard boundary constraint is too aggressive. The best next step is not another memory-update tweak. It should focus on calibrating the residual-to-score reconstruction:

- use history prior as the default exact score estimator
- use residual only when confidence/evidence is strong
- make boundary constraints soft, or only cap/floor when the boundary decision is high confidence
- add a diagnostic/eval table for `raw_classification`, `prior_only`, `prior+delta`, and `prior+delta+boundary`

## Post-hoc Calibration Check

Before changing the method implementation, two existing calibration methods were applied to the current result file:

```bash
input=outputs/personalized/history_prior_delta_none_n3_limit20.jsonl \
method=mean_shift \
output=outputs/personalized/history_prior_delta_none_n3_limit20_calMS.jsonl \
bash scripts/calibrate.sh

input=outputs/personalized/history_prior_delta_none_n3_limit20.jsonl \
method=cdf \
output=outputs/personalized/history_prior_delta_none_n3_limit20_calCDF.jsonl \
bash scripts/calibrate.sh
```

Both methods calibrated all 80 blocks with no fallback:

```text
n_blocks=80, n_calibrated=80, n_fallback_identity=0
```

Aligned comparison on the same 1594 sample ids:

| method | MAE | RMSE | Pearson | Spearman | QWK | Boundary Acc | F1-DSAT | DSAT Recall | False SAT | AUC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| rebuilt raw | 0.7215 | 1.0430 | 0.3131 | 0.2900 | 0.3084 | 0.7760 | 0.3521 | 0.3333 | 0.6667 | 0.6720 |
| rebuilt + mean_shift | 0.6731 | 0.9997 | 0.3706 | 0.3896 | 0.3650 | 0.7798 | 0.4160 | 0.4296 | 0.5704 | 0.7068 |
| rebuilt + CDF | 0.6531 | 1.0379 | 0.3735 | 0.4018 | 0.3725 | 0.7986 | 0.4298 | 0.4158 | 0.5842 | 0.7135 |
| Qwen v2.5 update | 0.7158 | 1.0183 | 0.2710 | 0.2735 | 0.2543 | 0.7710 | 0.3608 | 0.3540 | 0.6460 | 0.6475 |
| Qwen v2.4 update | 0.7045 | 1.0134 | 0.2692 | 0.2749 | 0.2526 | 0.7660 | 0.2976 | 0.2715 | 0.7285 | 0.6322 |
| user_history_mean | 0.6198 | 0.9511 | 0.2965 | 0.3370 | 0.2467 | 0.8074 | 0.1303 | 0.0790 | 0.9210 | 0.6548 |
| user_history_median | 0.5772 | 0.9701 | 0.3365 | 0.3988 | 0.2548 | 0.8174 | 0.0000 | 0.0000 | 1.0000 | 0.6710 |

Additional post-hoc variants:

| variant | MAE | RMSE | Pearson | Spearman | QWK | Boundary Acc | F1-DSAT | AUC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| raw LLM classification | 0.6819 | 0.9723 | 0.3149 | 0.3132 | 0.2873 | 0.7742 | 0.3571 | 0.6527 |
| raw classification + mean_shift | 0.6481 | 0.9554 | 0.3391 | 0.3539 | 0.3131 | 0.7886 | 0.3629 | 0.6669 |
| raw classification + CDF | 0.6631 | 1.0493 | 0.3596 | 0.3889 | 0.3586 | 0.7911 | 0.4085 | 0.7030 |
| prior round only | 0.5916 | 0.9199 | 0.3549 | 0.3936 | 0.3007 | 0.8124 | 0.2027 | 0.6639 |
| prior + delta, no boundary | 0.7654 | 1.0796 | 0.3148 | 0.2958 | 0.3131 | 0.7346 | 0.3878 | 0.6780 |
| prior + delta, no boundary + CDF | 0.6606 | 1.0421 | 0.3684 | 0.3919 | 0.3674 | 0.7861 | 0.3943 | 0.7031 |

### Calibration Interpretation

Calibration is strongly positive for the current version.

- CDF reduces MAE from 0.7215 to 0.6531.
- CDF raises Pearson from 0.3131 to 0.3735 and Spearman from 0.2900 to 0.4018.
- CDF raises QWK from 0.3084 to 0.3725.
- CDF raises boundary F1-DSAT from 0.3521 to 0.4298 and AUC from 0.6720 to 0.7135.

This means the model's within-block ordering is useful, and a large part of the error is calibration / scale mismatch. However, even after CDF, exact-score MAE is still worse than history-only median/mean on this subset. The calibrated method's advantage is in ranking and DSAT discovery, not in raw MAE.

The best post-hoc choices depend on the target:

- If optimizing full-score MAE only: `prior_round` or history-only remains stronger.
- If optimizing balanced semantic signal and boundary detection: rebuilt + CDF is strongest among tested variants.
- If optimizing RMSE / conservative full score: raw classification + mean_shift has the best RMSE among model variants, but weaker boundary F1-DSAT than rebuilt + CDF.

This supports a hybrid direction: keep history prior as the exact-score anchor, but use the residual/boundary judge as a ranking or DSAT-discovery signal, followed by CDF-style calibration.
