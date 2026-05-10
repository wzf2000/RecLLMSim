# Static Replay Judge Selection Analysis

## Question

Which satisfaction predictor should be used as the judge for Static Replay
Evaluation, and is the current performance sufficient for a fair benchmark?

## Current Candidate Predictors

Relevant 20-user subset results:

| judge candidate | MAE | Pearson | Spearman | QWK | SAT Acc | F1-DSAT | key behavior |
|---|---:|---:|---:|---:|---:|---:|---|
| Qwen3-8B v2 none | 0.7077 | 0.2999 | 0.3012 | 0.2807 | 0.7629 | 0.3505 | balanced but weaker ordinal signal |
| Qwen3.6-35B-A3B v2 none | 0.7842 | 0.3411 | 0.3335 | 0.3083 | 0.7120 | 0.4168 | stricter, better DSAT recall, poor raw calibration |
| Qwen3.6-35B-A3B v2 none + mean-shift | 0.6631 | 0.3828 | 0.4028 | 0.3758 | 0.7754 | 0.4207 | better scale and DSAT |
| Qwen3.6-35B-A3B v2 none + CDF | 0.6405 | 0.3977 | 0.4238 | 0.3966 | 0.7923 | 0.4121 | best ordinal/ranking signal among tested direct judges |
| Qwen3-8B HPD v3 episodic anchor4 | 0.6048 | 0.4016 | 0.4211 | 0.3690 | 0.7936 | 0.3459 | best MAE, weaker DSAT detection |

## Recommended Judge Track

For the first serious Static Replay benchmark, use a two-layer judge definition:

1. **Primary judge model:** `Qwen/Qwen3.6-35B-A3B`
2. **Memory builder model:** `Qwen/Qwen3-8B` as the first low-cost/default
   setting, with `Qwen/Qwen3.6-35B-A3B` as an ablation
3. **Primary predictor method:** `memory_version=v2`, `memory_update_mode=none`,
   `turn_eval_prompt_version=v2`
4. **Primary reported score:** user-history CDF calibrated score
5. **Auditable secondary score:** raw Qwen3.6 v2 score

Rationale:

- Static replay benchmark aggregation depends on average score scale, so raw
  Qwen3.6 is too strict to use alone.
- Qwen3.6 raw has better relative signal than Qwen3-8B raw, and calibration
  converts that signal into stronger QWK/Spearman while fixing much of the
  scale mismatch.
- `v2 none` is already supported by the static replay scorer and keeps judge
  state frozen across candidate models.
- Memory construction and turn prediction can now use different models via
  `memory_model` and `judge_model`. This should be used to isolate whether gains
  come from better memory summaries or better turn-level judging.
- CDF calibration uses only the user's historical labeled distribution, not the
  candidate model identity, so it remains model-independent if applied uniformly
  to all candidate responses.

The benchmark should report both:

- `judge_raw_score`: raw predictor output
- `judge_calibrated_score`: calibrated primary score used for ranking

This keeps the benchmark auditable and makes it possible to analyze whether a
candidate model wins only under calibration.

## Secondary Validation Track

Run a secondary judge using `Qwen3-8B HPD v3 episodic anchor4` or its Qwen3.6
counterpart once implemented and tested.

This track is valuable because HPD v3 episodic has the best current MAE and
strong rank metrics, but it is a more complex predictor flow and is not yet the
simplest static replay judge. It should be reported as a robustness/sensitivity
analysis rather than the first primary judge.

## Fairness Assessment

Current performance is enough for **exploratory and relative benchmark analysis**
if the following constraints are explicit:

- compare candidate models only under the same frozen judge;
- report bootstrap confidence intervals and avoid over-interpreting small score
  gaps;
- report raw and calibrated judge scores;
- include diagnostic distribution and SAT/DSAT rates;
- avoid claiming human-equivalent absolute satisfaction measurement.

Current performance is **not yet enough for a final high-stakes or fully fair
benchmark claim** by itself:

- predictor MAE is still around `0.64` even after Qwen3.6 CDF calibration;
- false SAT / false DSAT tradeoffs vary materially across predictors;
- judge backbone bias is possible, especially if Qwen judges Qwen-family
  candidate models;
- generated candidate responses do not have direct human labels, so gold-score
  diagnostics only evaluate against the original assistant response, not the
  candidate response.

The benchmark should therefore be framed as:

> automatic user-specific satisfaction proxy benchmark, validated against
> turn-level human satisfaction labels, with calibrated judge scores and
> sensitivity checks.

## Next Steps Before Using as Main Benchmark

1. Run Qwen3.6 full test or at least a larger fixed subset with `v2 none`.
2. Run Qwen3.6 HPD v3 episodic anchor4 on the same subset to test whether the
   stronger backbone improves the best current method.
3. Run the model-split ablation:
   - memory=`Qwen3-8B`, judge=`Qwen3.6-35B-A3B`
   - memory=`Qwen3.6-35B-A3B`, judge=`Qwen3.6-35B-A3B`
   - memory=`Qwen3.6-35B-A3B`, judge=`Qwen3-8B`
4. Add calibration as an explicit post-processing step in static replay scoring
   or evaluation.
5. Evaluate at least one non-Qwen judge on a subset to estimate judge-family
   sensitivity.
6. For paper reporting, include a small human validation sample for static
   replay outputs if budget allows.
