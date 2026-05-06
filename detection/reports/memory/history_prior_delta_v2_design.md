# History Prior Delta V2 Design

## Motivation

`history_prior_delta` showed useful ranking and boundary signal after calibration, but its direct reconstruction rule was too aggressive:

- `delta_score` had only weak correlation with the true residual.
- Hard 3/4 boundary constraints introduced many false DSAT errors.
- Code-side reconstruction had worse MAE than the model's raw diagnostic classification.

V2 keeps the same decomposition idea but changes the contract:

```text
exact score anchor = round(history_prior_score)
semantic judge = residual direction + confidence + boundary confidence
```

The LLM no longer gets to mechanically move the final score through every residual decision. It must additionally report whether the evidence is strong enough.

## New Prompt Version

```bash
turn_eval_prompt_version=history_prior_delta_v2
```

Added in:

- `detection/lib/memory.py`
- `detection/trace/collect_personalized.py`
- `detection/scripts/collect_personalized.sh`
- `detection/scripts/collect_personalized_vllm.sh`

## Output Fields

V2 extends the existing `history_prior_delta` schema with:

- `delta_confidence`: `low | medium | high`
- `boundary_confidence`: `low | medium | high`
- `strong_failure_evidence`: boolean
- `strong_excellence_evidence`: boolean

The JSONL records keep all diagnostic fields:

- `history_prior_score`
- `delta_label`
- `delta_score`
- `delta_confidence`
- `passes_satisfaction_boundary`
- `boundary_score`
- `boundary_confidence`
- `strong_failure_evidence`
- `strong_excellence_evidence`
- `history_prior_delta_raw_score`

## Reconstruction Rule

V1:

```text
score = round(history_prior_score + delta_score)
if boundary_score == 3: score <= 3
if boundary_score == 4: score >= 4
```

V2:

```text
score = round(history_prior_score)

if delta_confidence == high:
    score += sign(delta_score)
elif delta_confidence == medium and abs(delta_score) == 2:
    score += sign(delta_score)

if boundary_confidence == high:
    if boundary_score == 3:
        score = min(score, 3)
    if boundary_score == 4:
        score = max(score, 4)

score = clip(score, 1, 5)
```

This makes the history prior the default exact-score estimator, while keeping the LLM's residual and boundary judgments as conditional adjustment signals.

## Parse Stability Note

The first implementation reused the boundary prompt raw-text parse route. In
Qwen runs this produced frequent failures where the model emitted `<think>`
reasoning text but no JSON object, so `_structured_parse_from_raw_text` could not
recover a structured prediction.

`history_prior_delta_v2` now uses the SDK/schema structured parse route instead
of the raw boundary route, while `history_prior_delta` V1 remains unchanged for
comparability. The prompt also explicitly forbids `<think>`, reasoning drafts,
Markdown, explanations, and any text outside the JSON object.

## Intended Evaluation

Recommended first run:

```bash
cd detection
turn_eval_prompt_version=history_prior_delta_v2 \
memory_update_mode=none \
n_anchors=3 \
limit_users=20 \
output_jsonl=outputs/personalized/history_prior_delta_v2_none_n3_limit20.jsonl \
bash scripts/collect_personalized_vllm.sh
```

Then run the same calibration checks:

```bash
input=outputs/personalized/history_prior_delta_v2_none_n3_limit20.jsonl \
method=mean_shift \
output=outputs/personalized/history_prior_delta_v2_none_n3_limit20_calMS.jsonl \
bash scripts/calibrate.sh

input=outputs/personalized/history_prior_delta_v2_none_n3_limit20.jsonl \
method=cdf \
output=outputs/personalized/history_prior_delta_v2_none_n3_limit20_calCDF.jsonl \
bash scripts/calibrate.sh
```

Expected success criteria:

- lower raw MAE than V1 by avoiding noisy over-adjustment
- preserve or improve CDF-calibrated QWK / AUC
- keep F1-DSAT near the V1+CDF range without excessive false DSAT
- narrow the gap to `user_history_mean` / `user_history_median` on full-score MAE
