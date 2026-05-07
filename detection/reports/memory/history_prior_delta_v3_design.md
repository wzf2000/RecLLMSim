# History Prior Delta V3 Design

## Motivation

Full-run `history_prior_delta_v2` results show a clear tradeoff:

- exact-score MAE is strong (`0.5658`), almost matching `user_history_mean`
  (`0.5661`)
- DSAT discovery is weak (`F1-DSAT=0.2055`, `False SAT=0.8654`)
- only 398 / 6474 samples changed from `round(history_prior_score)`, so v2 is
  mostly a history-prior estimator

The useful v2 diagnostic fields were being suppressed by the high-confidence
gate. Offline variants suggested that `boundary_score=3`,
`history_prior_delta_raw_score<=3`, and negative `delta_score` carry DSAT signal,
but each is too noisy alone.

## Implementation

New prompt version:

```text
turn_eval_prompt_version=history_prior_delta_v3
```

Code changes:

- reuses the `HistoryPriorDeltaV2Prediction` schema
- keeps SDK/schema structured parsing, same as v2
- adds code-side helper `_history_prior_delta_v3_dsat_votes`
- adds code-side reconstruction `_reconstruct_history_prior_delta_v3_score`
- writes two diagnostic fields:
  - `dsat_signal_votes`
  - `pred_boundary_score`

Prompt changes are intentionally small. V3 tells the model that
`boundary_score=3`, `classification<=3`, and `delta_score<0` are downstream
DSAT signals, and asks it not to let a high user history mean hide current-turn
failure evidence.

## Reconstruction Rule

```text
score = round(history_prior_score)

dsat_votes =
  (boundary_score == 3)
  + (history_prior_delta_raw_score <= 3)
  + (delta_score < 0)

if dsat_votes >= 2:
    score = min(score, 3)
elif delta_confidence == "high":
    score += sign(delta_score)
elif delta_confidence == "medium" and abs(delta_score) == 2:
    score += sign(delta_score)

if boundary_confidence == "high" and boundary_score == 4:
    score = max(score, 4)

score = clip(score, 1, 5)
```

This keeps the v2 prior anchor for exact-score MAE, but allows medium-confidence
DSAT evidence to move the output when multiple signals agree.

## Expected Tradeoff

Compared with v2, expected behavior:

- higher predicted DSAT count
- higher F1-DSAT and lower False SAT
- slightly worse MAE than v2 if the DSAT vote is noisy
- better QWK / boundary metrics if the combined vote exposes the useful v2
  diagnostic signal seen in offline variants

Compared with v1, expected behavior:

- lower MAE because the prior remains the exact-score anchor
- lower DSAT recall than v1 if v1's single-signal boundary behavior was more
  aggressive
- cleaner diagnostics because v3 records the DSAT vote count explicitly

## Suggested First Run

```bash
cd detection
turn_eval_prompt_version=history_prior_delta_v3 \
memory_update_mode=none \
n_anchors=3 \
limit_users=20 \
output_jsonl=outputs/personalized/history_prior_delta_v3_none_n3_limit20.jsonl \
bash scripts/collect_personalized_vllm.sh
```

Then compare with v2 on the same subset and run CDF only as a boundary/ranking
post-processing check:

```bash
input=outputs/personalized/history_prior_delta_v3_none_n3_limit20.jsonl \
method=cdf \
output=outputs/personalized/history_prior_delta_v3_none_n3_limit20_calCDF.jsonl \
bash scripts/calibrate.sh
```

