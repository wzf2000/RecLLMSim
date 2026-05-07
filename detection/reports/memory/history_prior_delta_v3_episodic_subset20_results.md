# History Prior Delta V3 Episodic Subset-20 Results

Date: 2026-05-07

## Setup

Evaluated file:

- `outputs/personalized/qwen3_test_v2_none_hpd_v3_episodic_anchor4_u20.jsonl`

Comparison files:

- `outputs/personalized/history_prior_delta_v3_none_n3_limit20.jsonl`
- `outputs/personalized/history_prior_delta_v3_1_none_n3_limit20.jsonl`

Evaluation output:

- `outputs/personalized/hpd_v3_episodic_u20_personalized_compare.json`

All three files contain the same fixed 20-user subset:

- `n = 1594`
- `users = 20`
- gold score distribution: `1:38, 2:72, 3:181, 4:611, 5:692`
- gold DSAT ratio (`score<=3`): `18.26%`

## Global Metrics

| method | MAE | RMSE | Pearson | Spearman | QWK |
|---|---:|---:|---:|---:|---:|
| HPD v3 | 0.6311 | 0.9298 | 0.3999 | 0.4179 | 0.3777 |
| HPD v3.1 | 0.6110 | 0.9250 | 0.3842 | 0.4064 | 0.3544 |
| HPD v3 episodic | **0.6048** | **0.9114** | **0.4016** | **0.4211** | 0.3690 |

Episodic retrieval gives the best MAE/RMSE/Pearson/Spearman on this subset. QWK improves over v3.1 but remains below original v3.

## 3/4 Boundary Metrics

| method | pred DSAT | Acc | F1-macro | F1-DSAT | DSAT recall | False SAT |
|---|---:|---:|---:|---:|---:|---:|
| HPD v3 | 18.57% | 0.7723 | **0.6210** | **0.3816** | **0.3849** | **0.6151** |
| HPD v3.1 | 12.99% | **0.7942** | 0.6097 | 0.3414 | 0.2921 | 0.7079 |
| HPD v3 episodic | 13.30% | 0.7936 | 0.6117 | 0.3459 | 0.2990 | 0.7010 |

The episodic version is close to v3.1 on boundary metrics, with a very small DSAT improvement:

- F1-DSAT: `0.3414 -> 0.3459`
- DSAT recall: `0.2921 -> 0.2990`
- False SAT: `0.7079 -> 0.7010`

However, it is still much more SAT-conservative than original v3. The retrieved DSAT-side evidence did not restore v3-level DSAT discovery.

## User-Aware Metrics

| method | PU Pearson | PU Spearman | PU QWK | PU MAE | WC Pearson | WC MAE |
|---|---:|---:|---:|---:|---:|---:|
| HPD v3 | 0.1655 | 0.1464 | **0.1124** | 0.6311 | **0.1850** | 0.6665 |
| HPD v3.1 | 0.1534 | 0.1348 | 0.0897 | 0.6110 | 0.1517 | 0.6608 |
| HPD v3 episodic | **0.1836** | **0.1676** | 0.1085 | **0.6048** | 0.1807 | **0.6503** |

Episodic retrieval improves most user-aware full-score metrics versus v3.1, and even exceeds v3 on per-user Pearson/Spearman. This suggests the episodic evidence is useful for relative score calibration, even though it does not substantially improve DSAT recall.

User-aware binary metrics remain weak:

- PU F1-DSAT: `0.2369`
- PU binary AUC: `0.5247`
- WC binary Pearson: `0.1253`

## Prediction Distribution

| method | pred 3 | pred 4 | pred 5 | pred DSAT |
|---|---:|---:|---:|---:|
| HPD v3 | 296 | 827 | 471 | 18.57% |
| HPD v3.1 | 207 | 893 | 494 | 12.99% |
| HPD v3 episodic | 212 | 919 | 463 | 13.30% |

Compared with v3.1, episodic retrieval slightly increases DSAT predictions (`207 -> 212`) and shifts some `5` predictions down to `4` (`494 -> 463`, `893 -> 919`). This explains why MAE improves while DSAT F1 barely moves.

## Anchor Diagnostics

All 1594 turns retrieved 4 anchors.

Anchor evidence distribution:

- total DSAT-side anchors: `2623`
- total SAT-side anchors: `3753`
- turns with at least one DSAT-side anchor: `1406 / 1594 = 88.2%`

By task, turns with any DSAT-side anchor:

- `技能学习规划`: `308 / 335 = 91.9%`
- `旅行规划`: `421 / 463 = 90.9%`
- `礼物准备`: `369 / 460 = 80.2%`
- `菜谱规划`: `308 / 336 = 91.7%`

Even though most turns saw DSAT-side evidence, the final predicted DSAT ratio is only `13.30%`, below the true `18.26%`. Thus the bottleneck is not retrieval coverage alone; it is the model's use of retrieved DSAT evidence under the current prompt/reconstruction rule.

## Changed Predictions

Compared with HPD v3:

- changed samples: `163 / 1594 = 10.2%`
- error improved / worsened / tied among changed samples: `102 / 58 / 3`
- boundary improved / worsened / tied among changed samples: `80 / 46 / 37`

Compared with HPD v3.1:

- changed samples: `139 / 1594 = 8.7%`
- error improved / worsened / tied among changed samples: `74 / 60 / 5`
- boundary improved / worsened / tied among changed samples: `46 / 47 / 46`

This confirms that episodic retrieval makes non-trivial but localized changes. It improves exact-score error more often than it hurts, but its boundary changes are almost balanced relative to v3.1.

## Interpretation

The episodic version is useful as a score-calibration improvement, but not yet a strong satisfaction-boundary improvement.

Likely reasons:

1. The prompt still tells the model not to mechanically copy anchors, and HPD v3 reconstruction still requires multiple DSAT signals. Retrieved DSAT examples may influence analysis without flipping enough structured signals.
2. Boundary-paired retrieval often includes both DSAT and SAT evidence. This prevents over-penalization, but also makes the judge conservative when evidence is mixed.
3. TF-IDF anchors are surface-similar, not necessarily failure-mode-similar. A DSAT-side anchor may not demonstrate the same failure as the target reply.
4. The model shifts many `5` predictions to `4`, which helps MAE but does not address the main DSAT recall gap.

## Recommendation

Do not run this version full-scale yet as the main candidate. It is promising enough to keep, but the current implementation is not clearly better than v3/v3.1 on the boundary objective.

Recommended next variant:

- keep `history_prior_delta_v3_episodic`
- only inject episodic anchors when first-pass HPD is uncertain or internally inconsistent
- make retrieved anchors failure-mode oriented rather than raw TF-IDF only
- add an explicit structured field such as `episodic_boundary_vote` or `closest_evidence_side`, so anchor comparison has a direct path into reconstruction

The most direct next experiment is a two-pass version:

1. First pass: current HPD v3.1 or episodic-free HPD v3 route.
2. If boundary confidence is not high, retrieve paired anchors and ask only for `closest_evidence_side`, `evidence_match_confidence`, and revised DSAT signals.

This should preserve the full-score gains while giving retrieval a cleaner mechanism to affect the `3/4` decision.

