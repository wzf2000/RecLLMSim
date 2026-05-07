# History Prior Delta V3 Subset20 Results

## Run

Result file:

```text
detection/outputs/personalized/history_prior_delta_v3_none_n3_limit20.jsonl
```

Post-hoc calibration files:

```text
detection/outputs/personalized/history_prior_delta_v3_none_n3_limit20_calMS.jsonl
detection/outputs/personalized/history_prior_delta_v3_none_n3_limit20_calCDF.jsonl
```

Configuration inferred from records:

- model: `Qwen/Qwen3-8B`
- users: 20
- samples: 1594
- memory update: `none`
- memory version: `v2`
- anchors: `n_anchors=3`
- turn prompt: `history_prior_delta_v3`

No `history_prior_delta_v3` parse failure files were found under
`detection/outputs/personalized/parse_failures/`.

## Aligned Comparison

All rows below are aligned to the same 1594 `sample_id`s.

| method | MAE | RMSE | Pearson | Spearman | QWK | Boundary Acc | F1-DSAT | DSAT Recall | False SAT | False DSAT |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| history prior delta v1 | 0.7215 | 1.0430 | 0.3131 | 0.2900 | 0.3084 | 0.7760 | 0.3521 | 0.3333 | 0.6667 | 0.1251 |
| v1 + CDF | 0.6531 | 1.0379 | 0.3735 | 0.4018 | 0.3725 | 0.7986 | **0.4298** | **0.4158** | **0.5842** | 0.1159 |
| history prior delta v2 | **0.5916** | **0.9281** | 0.3608 | 0.3924 | 0.3081 | **0.8143** | 0.2000 | 0.1271 | 0.8729 | **0.0322** |
| v2 + CDF | 0.6945 | 1.0955 | 0.3017 | 0.3366 | 0.3009 | 0.7848 | 0.3908 | 0.3780 | 0.6220 | 0.1243 |
| history prior delta v3 | 0.6311 | 0.9298 | **0.3999** | 0.4179 | **0.3777** | 0.7723 | 0.3816 | 0.3849 | 0.6151 | 0.1412 |
| v3 + mean shift | 0.6211 | 0.9325 | 0.3964 | **0.4232** | 0.3753 | 0.7817 | 0.3603 | 0.3368 | 0.6632 | 0.1190 |
| v3 + CDF | 0.6694 | 1.0523 | 0.3560 | 0.3828 | 0.3550 | 0.7873 | 0.3979 | 0.3849 | 0.6151 | 0.1228 |
| Qwen v2.5 update | 0.7158 | 1.0183 | 0.2710 | 0.2735 | 0.2543 | 0.7710 | 0.3608 | 0.3540 | 0.6460 | 0.1358 |
| Qwen v2.4 update | 0.7045 | 1.0134 | 0.2692 | 0.2749 | 0.2526 | 0.7660 | 0.2976 | 0.2715 | 0.7285 | 0.1236 |
| boundary v4 | 0.7848 | 1.0251 | 0.2173 | 0.2075 | 0.1391 | 0.7422 | 0.3445 | 0.3711 | 0.6289 | 0.1750 |
| user history mean | 0.6198 | 0.9511 | 0.2965 | 0.3370 | 0.2467 | 0.8074 | 0.1303 | 0.0790 | 0.9210 | 0.0299 |
| user history median | 0.5772 | 0.9701 | 0.3365 | 0.3988 | 0.2548 | 0.8174 | 0.0000 | 0.0000 | 1.0000 | 0.0000 |
| nearest history k3 | 0.6826 | 1.0478 | 0.2675 | 0.3176 | 0.2580 | 0.7647 | 0.2884 | 0.2612 | 0.7388 | 0.1228 |

## V3 Diagnostics

Prediction distribution:

```text
gold_score: 1=38, 2=72, 3=181, 4=611, 5=692
pred_score: 3=296, 4=827, 5=471
raw_classification: 1=1, 2=10, 3=240, 4=1151, 5=192
```

Judge field distribution:

```text
delta_label: below=326, around=1132, above=136
delta_score: -2=71, -1=256, 0=1131, 1=126, 2=10
delta_confidence: medium=1528, high=66
boundary_score: 3=232, 4=1362
boundary_confidence: medium=1526, high=68
passes_satisfaction_boundary: false=172, true=1422
strong_failure_evidence: true=122
strong_excellence_evidence: true=60
dsat_signal_votes: 0=1267, 1=75, 2=21, 3=231
```

V3 changed 246 / 1594 samples (15.43%) from `round(history_prior_score)`. This
is much less prior-dominated than v2 on the same subset, where only 79 predicted
DSAT cases were produced.

Confusion matrix, rows are gold 1-5 and columns are predicted 1-5:

```text
[
  [0, 0, 21, 15, 2],
  [0, 0, 32, 38, 2],
  [0, 0, 59, 110, 12],
  [0, 0, 132, 358, 121],
  [0, 0, 52, 306, 334],
]
```

V3 still never predicts 1 or 2; the vote rule only creates 3-level DSAT.

## DSAT Vote Quality

| signal | marked | DSAT precision | DSAT recall |
|---|---:|---:|---:|
| `dsat_signal_votes >= 2` | 252 | 0.373 | 0.323 |
| `dsat_signal_votes >= 1` | 327 | 0.306 | 0.344 |
| `boundary_score == 3` | 232 | 0.384 | 0.306 |
| `raw_classification <= 3` | 251 | 0.375 | 0.323 |
| `delta_score < 0` | 327 | 0.306 | 0.344 |
| `passes_satisfaction_boundary == false` | 172 | 0.372 | 0.220 |
| `strong_failure_evidence == true` | 122 | 0.336 | 0.141 |

Vote buckets:

| votes | n | gold DSAT rate | MAE | F1-DSAT |
|---:|---:|---:|---:|---:|
| 0 | 1267 | 0.151 | 0.5541 | 0.1466 |
| 1 | 75 | 0.080 | 0.3867 | 0.2222 |
| 2 | 21 | 0.238 | 0.9524 | 0.3846 |
| 3 | 231 | 0.385 | 1.1039 | 0.5563 |

The vote has usable recall but weak precision. The bucket with 3 votes is the
main DSAT-discovery source; 2-vote cases are few and noisy.

## Interpretation

V3 is a better balanced method than both v1 and v2 on this subset:

- Compared with v1, MAE improves from `0.7215` to `0.6311`, while F1-DSAT also
  improves from `0.3521` to `0.3816`.
- Compared with v2, MAE worsens from `0.5916` to `0.6311`, but DSAT recall rises
  from `0.1271` to `0.3849` and F1-DSAT rises from `0.2000` to `0.3816`.
- Compared with Qwen v2.5 update, v3 improves MAE, Pearson, Spearman, QWK, and
  F1-DSAT on the same subset.
- Compared with history-only, v3 is much better on ranking and DSAT discovery,
  but worse than `user_history_median` on pure MAE.

Post-hoc calibration is mixed:

- mean shift slightly improves MAE (`0.6311` to `0.6211`) and Spearman
  (`0.4179` to `0.4232`), but lowers F1-DSAT.
- CDF raises boundary accuracy and F1-DSAT modestly (`0.3979`), but damages MAE
  (`0.6694`) and QWK (`0.3550`).

## Next Step

V3 should not be rolled back. The vote rule successfully restores boundary
signal while keeping much better MAE than v1. The next refinement should focus
on precision:

- Treat 3-vote DSAT as strong enough for `score<=3`.
- Treat 2-vote DSAT more carefully; it has only `0.238` DSAT rate on this run.
- Consider adding a stricter 2-vote rule, for example require either
  `boundary_score==3 and raw_classification<=3`, or require
  `strong_failure_evidence=true` for 2-vote cases.
- Keep mean-shift as an optional post-processing for MAE-focused reporting, but
  do not use CDF as the default exact-score output.

