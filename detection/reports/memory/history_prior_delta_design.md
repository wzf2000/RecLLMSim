# History Prior Delta Judge Design

## Motivation

Recent history-only baselines show that full 1-5 satisfaction scores contain a strong user-level prior. A simple user history mean / median is already competitive on global MAE, while semantic LLM judges are more useful on turn-level residual variation and the 3/4 satisfied boundary.

This version makes that decomposition explicit:

```text
final_score = user_history_prior + turn_quality_delta
```

The LLM is no longer asked to freely emit a 1-5 score first. It must first judge whether the current turn is below, around, or above the user's historical scoring prior, and separately judge whether the turn passes the 3/4 satisfaction boundary.

## Implementation

Added a new turn evaluation prompt version:

```bash
turn_eval_prompt_version=history_prior_delta
```

Code paths:

- `detection/lib/memory.py`
  - Adds the `history_prior_delta` prompt branch in `build_turn_eval_prompt`.
  - The prompt exposes `memory.avg_satisfaction_score` as `history_prior_score`.
  - It asks for `delta_label`, `delta_score`, `passes_satisfaction_boundary`, and `boundary_score`.
- `detection/trace/collect_personalized.py`
  - Adds `HistoryPriorDeltaPrediction`.
  - Parses the extra residual / boundary fields.
  - Reconstructs the final `pred_score` in code from `history_prior_score + delta_score`.
  - Applies a hard 3/4 boundary constraint:
    - if `boundary_score=3`, final score is capped at 3
    - if `boundary_score=4`, final score is floored at 4
  - Stores diagnostic fields in each JSONL record:
    - `history_prior_score`
    - `delta_label`
    - `delta_score`
    - `passes_satisfaction_boundary`
    - `boundary_score`
    - `history_prior_delta_raw_score`
- `detection/scripts/collect_personalized.sh`
- `detection/scripts/collect_personalized_vllm.sh`
  - Document the new prompt version in configurable comments.

## Reconstruction Rule

The LLM output includes a diagnostic `classification`, but the recorded `pred_score` is rebuilt by code:

```text
reconstructed = round(history_prior_score + delta_score), clipped to [1, 5]

if boundary_score == 3:
    pred_score = min(3, reconstructed)
else:
    pred_score = max(4, reconstructed)
```

This makes the method faithful to the intended residual formulation and prevents the model from bypassing the history-prior step by directly choosing a final score.

## Suggested Run

From `detection/`:

```bash
turn_eval_prompt_version=history_prior_delta \
memory_update_mode=none \
n_anchors=3 \
bash scripts/collect_personalized_vllm.sh
```

For API models:

```bash
turn_eval_prompt_version=history_prior_delta \
memory_update_mode=none \
n_anchors=3 \
bash scripts/collect_personalized.sh
```

`memory_update_mode=none` is the recommended first run so the history prior remains fixed across target sessions. After the fixed-memory behavior is understood, `per_session` can be tested separately.

## Evaluation Focus

This variant should be evaluated against:

- full 1-5 metrics, especially relative to `user_history_mean` / `user_history_median`
- 3/4 boundary metrics, especially relative to `nearest_history_turn` / `nearest_history_turn_k3`
- residual diagnostics:
  - whether `delta_score` correlates with `gold_score - history_prior_score`
  - whether `boundary_score` improves DSAT recall without excessive false DSAT

The key question is not only whether MAE improves, but whether the LLM adds turn-specific signal beyond the user history prior.
