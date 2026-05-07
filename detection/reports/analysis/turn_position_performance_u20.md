# Turn Position Performance Analysis on U20

## Setup

This report analyzes how prediction performance changes with dialogue turn
position. The comparison is aligned to the 1594 `sample_id`s in:

```text
detection/outputs/personalized/history_prior_delta_v3_none_n3_limit20.jsonl
```

Computed JSON summary:

```text
detection/outputs/personalized/turn_position_compare_u20.json
```

Compared routes:

- history-only: `user_history_median`, `user_history_mean`
- retrieval baseline: `nearest_history_turn_k3`
- Qwen update route: `Qwen_Qwen3-8B_test_per_session_updv2_5_u20`
- prior-delta route:
  - `history_prior_delta` + CDF
  - `history_prior_delta_v2`
  - `history_prior_delta_v3`
  - `history_prior_delta_v3` + mean shift
  - `history_prior_delta_v3` + CDF

## Gold Distribution by Turn

| turn bucket | n | gold mean | gold DSAT rate |
|---:|---:|---:|---:|
| 0 | 346 | 3.942 | 0.275 |
| 1 | 344 | 4.172 | 0.177 |
| 2 | 328 | 4.183 | 0.180 |
| 3 | 284 | 4.310 | 0.116 |
| 4 | 165 | 4.309 | 0.127 |
| 5+ | 127 | 4.118 | 0.173 |

The key observation is that turn 0 is not an obviously "uninformative" bucket in
the labels. It has the highest DSAT rate and the lowest mean score. Removing the
first one or two turns therefore changes the task distribution, not just noise.

## Exact-Score MAE by Turn

| method | all | turn 0 | turn 1 | turn 2 | turn 3 | turn 4 | turn 5+ |
|---|---:|---:|---:|---:|---:|---:|---:|
| history median | 0.577 | 0.682 | **0.503** | 0.573 | **0.514** | **0.503** | 0.740 |
| history mean | 0.620 | 0.720 | 0.547 | 0.616 | 0.563 | 0.545 | 0.780 |
| nearest k3 | 0.683 | 0.717 | 0.634 | 0.634 | 0.637 | 0.624 | 1.024 |
| Qwen v2.5 update | 0.716 | 0.818 | 0.680 | 0.655 | 0.602 | 0.776 | 0.866 |
| hpd v1 + CDF | 0.653 | 0.864 | 0.602 | 0.591 | 0.521 | 0.576 | 0.772 |
| hpd v2 | **0.592** | 0.705 | 0.488 | 0.601 | 0.518 | 0.552 | 0.756 |
| hpd v3 | 0.631 | 0.720 | 0.584 | 0.564 | 0.567 | 0.606 | 0.866 |
| hpd v3 + mean shift | 0.621 | 0.702 | 0.581 | 0.570 | 0.546 | 0.594 | 0.843 |
| hpd v3 + CDF | 0.669 | 0.934 | 0.645 | **0.558** | 0.525 | 0.576 | 0.748 |

MAE generally improves when excluding early turns, especially turn 0. However,
turn 5+ is also difficult for many methods, likely due to smaller sample size
and different session dynamics.

## Boundary F1-DSAT by Turn

| method | all | turn 0 | turn 1 | turn 2 | turn 3 | turn 4 | turn 5+ |
|---|---:|---:|---:|---:|---:|---:|---:|
| history median | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| history mean | 0.130 | 0.143 | 0.179 | 0.135 | 0.095 | 0.080 | 0.000 |
| nearest k3 | 0.288 | 0.359 | 0.280 | 0.333 | 0.215 | 0.150 | 0.196 |
| Qwen v2.5 update | 0.361 | 0.477 | 0.234 | 0.421 | 0.255 | 0.240 | 0.238 |
| hpd v1 + CDF | **0.430** | 0.541 | **0.441** | 0.396 | 0.208 | 0.286 | 0.216 |
| hpd v2 | 0.200 | 0.193 | 0.263 | 0.156 | 0.255 | 0.207 | 0.074 |
| hpd v3 | 0.382 | 0.443 | 0.293 | **0.477** | **0.299** | **0.333** | **0.321** |
| hpd v3 + mean shift | 0.360 | 0.439 | 0.286 | 0.423 | 0.295 | 0.298 | 0.255 |
| hpd v3 + CDF | 0.398 | **0.543** | 0.339 | 0.306 | 0.105 | 0.242 | 0.194 |

Boundary behavior differs from MAE. The strongest boundary methods often get
much of their F1-DSAT from turn 0, where the DSAT base rate is high. V3 is more
stable across later turns than CDF variants: it keeps useful F1-DSAT at turns
2, 3, 4, and 5+.

## Early vs Later Turns

| method | bucket | n | DSAT rate | MAE | QWK | F1-DSAT | pred DSAT rate | False SAT |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| history median | all | 1594 | 0.183 | **0.577** | 0.255 | 0.000 | 0.000 | 1.000 |
| history median | turn >= 2 | 904 | 0.149 | **0.565** | 0.200 | 0.000 | 0.000 | 1.000 |
| hpd v2 | all | 1594 | 0.183 | 0.592 | 0.308 | 0.200 | 0.050 | 0.873 |
| hpd v2 | turn >= 2 | 904 | 0.149 | 0.587 | 0.268 | 0.178 | 0.050 | 0.881 |
| hpd v3 | all | 1594 | 0.183 | 0.631 | **0.378** | 0.382 | 0.186 | 0.615 |
| hpd v3 | turn >= 2 | 904 | 0.149 | 0.615 | **0.369** | **0.378** | 0.167 | **0.600** |
| hpd v1 + CDF | all | 1594 | 0.183 | 0.653 | 0.373 | **0.430** | 0.171 | **0.584** |
| hpd v1 + CDF | turn >= 2 | 904 | 0.149 | 0.592 | 0.321 | 0.306 | 0.090 | 0.756 |
| Qwen v2.5 update | all | 1594 | 0.183 | 0.716 | 0.254 | 0.361 | 0.176 | 0.646 |
| Qwen v2.5 update | turn >= 2 | 904 | 0.149 | 0.690 | 0.229 | 0.314 | 0.118 | 0.719 |

Filtering out turn 0/1 improves MAE for most methods, but it also lowers the
DSAT base rate and makes boundary metrics less comparable. For example, hpd v1
+ CDF has high all-sample F1-DSAT partly because it is very aggressive at turn
0. V3 retains a better later-turn boundary profile.

## Interpretation

1. Early turns are not just meaningless setup turns in this subset. Turn 0 has
   the highest DSAT rate, so it likely includes many immediately inadequate
   first responses or clarifying replies judged as unsatisfactory.

2. The method ranking depends on the target metric:

   - Best pure MAE: `history_median` / `hpd_v2`.
   - Best balanced ranking: `hpd_v3` has the best all-sample QWK and remains
     strong on turn >= 2.
   - Best DSAT discovery: `hpd_v1 + CDF` overall, but its advantage is heavily
     concentrated in turn 0/1.
   - Best later-turn DSAT balance: `hpd_v3`, especially on turn 2 and turn 5+.

3. A fairer reporting protocol should include both:

   - all turns, because early-turn dissatisfaction is real in the labels
   - content-heavy turns, e.g. `turn_idx >= 2`, because the user hypothesis
     about later turns is plausible and changes conclusions

4. If the paper or report wants to argue that the method captures real
   satisfaction rather than generic first-turn failures, `hpd_v3` is currently
   the best candidate: it does not rely solely on turn-0 DSAT like CDF variants,
   and it keeps useful QWK / F1-DSAT on later turns.

