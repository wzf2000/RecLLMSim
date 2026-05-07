# History Prior Delta V3 Episodic Design

Date: 2026-05-07

## Goal

This version tests whether the current summary-memory predictor can benefit from concrete historical evidence without changing the output format or evaluation pipeline.

The implementation keeps the existing `UserMemory` summary as the main user prior, and adds a lightweight episodic retrieval path over historical labeled turns. The output JSONL remains compatible with existing personalized evaluation scripts.

## What Changed

New prompt version:

- `history_prior_delta_v3_episodic`

This version reuses the existing `HistoryPriorDeltaV2Prediction` schema and the same v3 reconstruction logic:

- default score is anchored by `history_prior_score`
- `boundary_score=3`, `classification<=3`, and `delta_score<0` are DSAT signals
- two or more DSAT signals constrain the final score to `<=3`
- otherwise, high-confidence residual evidence can move the prior by one step

The difference is that prediction can now insert boundary-paired episodic anchors retrieved from the user's historical source sessions.

## Episodic Retrieval

The existing `AnchorRetriever` was extended with `retrieve_boundary_paired`.

For each target turn, when `turn_eval_prompt_version=history_prior_delta_v3_episodic`, the retriever tries to return both:

- DSAT-side evidence: similar historical turns with score `<=3`
- SAT-side evidence: similar historical turns with score `>=4`

The retrieval backend is still the existing character n-gram TF-IDF index, so this is intentionally a lightweight first step rather than a full vector database or graph-memory system.

Each retrieved anchor is labeled with:

- `DSAT-side evidence (score<=3)`
- `SAT-side evidence (score>=4)`

The prompt instructs the model to compare the target reply against both sides of the `3/4` boundary, while still using summary memory as the user-level scoring prior.

## Output Fields

The core output fields are unchanged:

- `sample_id`
- `user`
- `target_task`
- `turn_idx`
- `gold_score`
- `pred_score`
- `gold_reason`
- `reason_prediction`
- `analysis`

Additional diagnostic fields are added when anchors are used:

- `n_anchors_retrieved`
- `anchor_scores`
- `anchor_tasks`
- `anchor_evidence_roles`

These fields are optional and should not affect existing evaluation scripts.

## Intended Comparison

Recommended first comparison:

- baseline: `turn_eval_prompt_version=history_prior_delta_v3`, `memory_update_mode=none`
- episodic: `turn_eval_prompt_version=history_prior_delta_v3_episodic`, `memory_update_mode=none`, `n_anchors=4`

Use the same fixed user subset first, then scale up if parsing and latency are acceptable.

Suggested command:

```bash
model=Qwen/Qwen3-8B \
memory_update_mode=none \
memory_version=v2 \
turn_eval_prompt_version=history_prior_delta_v3_episodic \
n_anchors=4 \
limit_users=20 \
output_jsonl=outputs/personalized/qwen3_test_v2_none_hpd_v3_episodic_anchor4_u20.jsonl \
bash scripts/collect_personalized.sh
```

Then evaluate with the existing personalized, user-aware, and boundary scripts.

## Expected Behavior

This version should mainly affect boundary cases:

- if summary memory is SAT-heavy but retrieved DSAT-side examples show similar concrete failures, DSAT recall may improve
- if the model is tempted to over-penalize a merely imperfect reply, SAT-side examples should provide counter-evidence
- full-score MAE/QWK may not improve immediately, because the new retrieval is optimized for `3/4` boundary evidence rather than exact `1-5` calibration

## Risks

- TF-IDF retrieval may retrieve surface-similar but semantically misleading examples.
- Adding anchors increases prompt length and latency.
- If the user's history has very few `<=3` examples, boundary-paired retrieval may be imbalanced.
- The current implementation is not yet a persistent vector/graph memory corpus; it is a lightweight episodic retrieval upgrade over the existing transient anchor path.

