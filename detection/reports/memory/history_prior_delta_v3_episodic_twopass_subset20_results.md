# History Prior Delta V3 Episodic Two-Pass Subset-20 Results

Date: 2026-05-09

## Setup

Evaluated file:

- `outputs/personalized/qwen3_test_v2_none_hpd_v3_episodic_twopass_anchor4_u20.jsonl`

Comparison files:

- `outputs/personalized/history_prior_delta_v3_none_n3_limit20.jsonl`
- `outputs/personalized/history_prior_delta_v3_1_none_n3_limit20.jsonl`
- `outputs/personalized/qwen3_test_v2_none_hpd_v3_episodic_anchor4_u20.jsonl`

Evaluation output:

- `outputs/personalized/hpd_v3_episodic_twopass_u20_personalized_compare.json`
- `outputs/personalized/hpd_v3_episodic_twopass_u20_user_aware.json`
- `outputs/personalized/hpd_v3_episodic_twopass_u20_binary_sat.json`

All files use the same fixed 20-user subset:

- `n = 1594`
- `users = 20`
- gold score distribution: `1:38, 2:72, 3:181, 4:611, 5:692`
- gold DSAT ratio (`score<=3`): `18.26%`

## Global Metrics

| method | MAE | RMSE | Pearson | Spearman | QWK |
|---|---:|---:|---:|---:|---:|
| HPD v3 | 0.6311 | 0.9298 | 0.3999 | 0.4179 | **0.3777** |
| HPD v3.1 | 0.6110 | 0.9250 | 0.3842 | 0.4064 | 0.3544 |
| HPD v3 episodic | **0.6048** | **0.9114** | **0.4016** | **0.4211** | 0.3690 |
| HPD v3 episodic two-pass | 0.6368 | 0.9455 | 0.3611 | 0.3695 | 0.3353 |

The two-pass version is worse than all three comparison methods on full-score metrics.

## 3/4 Boundary Metrics

| method | pred DSAT | Acc | F1-macro | F1-DSAT | DSAT recall | False SAT |
|---|---:|---:|---:|---:|---:|---:|
| HPD v3 | 18.57% | 0.7723 | **0.6210** | **0.3816** | **0.3849** | **0.6151** |
| HPD v3.1 | 12.99% | **0.7942** | 0.6097 | 0.3414 | 0.2921 | 0.7079 |
| HPD v3 episodic | 13.30% | 0.7936 | 0.6117 | 0.3459 | 0.2990 | 0.7010 |
| HPD v3 episodic two-pass | 16.37% | 0.7754 | 0.6078 | 0.3514 | 0.3333 | 0.6667 |

Two-pass improves DSAT recall over v3.1 and old episodic, but still remains below HPD v3. The extra DSAT recall comes with clear losses in accuracy, MAE, Pearson/Spearman, and QWK.

## User-Aware Metrics

| method | PU Pearson | PU Spearman | PU QWK | PU MAE | WC Pearson | WC MAE |
|---|---:|---:|---:|---:|---:|---:|
| HPD v3 | 0.1655 | 0.1464 | 0.1124 | 0.6311 | **0.1850** | 0.6665 |
| HPD v3.1 | 0.1534 | 0.1348 | 0.0897 | 0.6110 | 0.1517 | 0.6608 |
| HPD v3 episodic | **0.1836** | **0.1676** | 0.1085 | **0.6048** | 0.1807 | **0.6503** |
| HPD v3 episodic two-pass | 0.1707 | 0.1411 | **0.1217** | 0.6368 | 0.1635 | 0.6668 |

Two-pass has the best per-user QWK, but most other user-aware full-score metrics are worse than the old episodic route.

For user-aware binary metrics, two-pass improves over old episodic:

- PU F1-DSAT: `0.2369 -> 0.3057`
- PU binary Kappa: `0.1130 -> 0.1703`
- PU binary AUC: `0.5247 -> 0.5666`

This confirms that two-pass pushes the system toward better DSAT discovery, but at substantial full-score cost.

## Prediction Distribution

| method | pred 3 | pred 4 | pred 5 | pred DSAT |
|---|---:|---:|---:|---:|
| HPD v3 | 296 | 827 | 471 | 18.57% |
| HPD v3.1 | 207 | 893 | 494 | 12.99% |
| HPD v3 episodic | 212 | 919 | 463 | 13.30% |
| HPD v3 episodic two-pass | 261 | 898 | 435 | 16.37% |

Two-pass moves predictions toward DSAT and away from score 5. This helps DSAT recall but hurts exact-score calibration.

## Two-Pass Diagnostics

The intended design was selective refinement, but the implementation was effectively almost always triggered:

- `episodic_refine_triggered=True`: `1562 / 1594 = 98.0%`
- `episodic_refine_applied=True`: `1554 / 1594 = 97.5%`

Second-pass evidence side:

- `sat`: 1324
- `dsat`: 238
- not triggered / missing: 32

Confidence:

- `high`: 741
- `medium`: 821

Side-confidence pairs:

- `sat + high`: 695
- `sat + medium`: 629
- `dsat + medium`: 192
- `dsat + high`: 46

The main problem is that the trigger rule is too broad: almost every sample goes through the second pass, so the method is no longer selective.

## Error Diagnostics

Refinement groups:

| group | n | MAE | gold DSAT | pred DSAT |
|---|---:|---:|---:|---:|
| not triggered | 32 | 1.0312 | 25.00% | 46.88% |
| triggered not applied | 8 | 1.0000 | 50.00% | 100.00% |
| applied | 1554 | 0.6268 | 17.95% | 15.32% |
| applied DSAT | 238 | 1.1387 | 37.39% | 100.00% |
| applied SAT | 1316 | 0.5342 | 14.44% | 0.00% |

The `applied DSAT` group is the major failure mode. It contains many true 4/5 samples that were pulled down to 3 by medium-confidence DSAT evidence.

Pairwise changes:

Compared with HPD v3:

- changed samples: 250
- error improved / worsened / tied: `123 / 123 / 4`
- boundary improved / worsened / tied: `107 / 102 / 41`

Compared with HPD v3.1:

- changed samples: 262
- error improved / worsened / tied: `113 / 143 / 6`
- boundary improved / worsened / tied: `88 / 118 / 56`

Compared with old HPD v3 episodic:

- changed samples: 213
- error improved / worsened / tied: `83 / 127 / 3`
- boundary improved / worsened / tied: `77 / 106 / 30`

Thus the two-pass changes are not net-positive. Relative to the previous episodic version, it makes more boundary mistakes than it fixes.

## Task-Level Observations

| task | n | triggered | applied | MAE | pred DSAT | gold DSAT |
|---|---:|---:|---:|---:|---:|---:|
| 技能学习规划 | 335 | 323 | 322 | 0.6030 | 8.36% | 18.81% |
| 旅行规划 | 463 | 457 | 453 | 0.6048 | 16.20% | 17.93% |
| 礼物准备 | 460 | 453 | 451 | 0.6783 | 23.26% | 19.57% |
| 菜谱规划 | 336 | 329 | 328 | 0.6577 | 15.18% | 16.37% |

礼物准备 is over-predicted as DSAT, while 技能学习规划 is still strongly under-predicted as DSAT. This suggests retrieval/refinement behavior is task-sensitive and not just a global threshold issue.

## Interpretation

The two-pass design did achieve its narrow goal of increasing DSAT predictions:

- pred DSAT rose from old episodic `13.30%` to `16.37%`
- DSAT recall rose from `0.2990` to `0.3333`
- false SAT dropped from `0.7010` to `0.6667`

However, it over-triggered and over-applied. The second pass became a near-universal reranker rather than a targeted uncertainty mechanism. Medium-confidence DSAT evidence was too strong and caused many high-score samples to collapse to 3.

## Recommendation

Do not use the current two-pass version as the main route.

The best current option remains:

- `history_prior_delta_v3_episodic` if prioritizing full-score and user-aware metrics
- `history_prior_delta_v3` if prioritizing DSAT recall / boundary F1

If continuing this direction, the next revision should:

1. Tighten the trigger condition substantially:
   - trigger only when first-pass DSAT vote count is 1 or 2, or boundary/confidence fields explicitly conflict
   - do not trigger merely because `boundary_confidence != high`
2. Require stronger DSAT application:
   - medium DSAT should not force `<=3`
   - DSAT-side refinement should apply only when `closest_evidence_side=dsat`, confidence is high, and first pass already has at least 2 DSAT signals
3. Use SAT-side evidence mainly to rescue false DSAT, not to rewrite most normal SAT cases.

