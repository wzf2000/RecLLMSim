# History Prior Delta V3 Episodic Two-Pass Design

Date: 2026-05-09

## Motivation

The first `history_prior_delta_v3_episodic` subset run showed that boundary-paired anchors improved full-score calibration but did not substantially improve `3/4` DSAT discovery. The likely reason was that retrieved anchors were injected into the main scoring prompt, so their effect on the final boundary decision was indirect.

This version makes episodic evidence affect the boundary through an explicit second-pass review.

## New Version

New prompt version:

- `history_prior_delta_v3_episodic_twopass`

The version is compatible with the existing JSONL output and evaluation pipeline.

## Prediction Flow

### First Pass

The first pass uses the existing `history_prior_delta_v3_1` prompt without episodic anchors.

The model predicts:

- `classification`
- `history_prior_score`
- `delta_label`
- `delta_score`
- `delta_confidence`
- `boundary_score`
- `boundary_confidence`
- `strong_failure_evidence`
- `strong_excellence_evidence`

The program reconstructs the initial score with the existing v3.1 rule:

- default to rounded history prior
- force `<=3` only when all three DSAT signals agree:
  - `boundary_score == 3`
  - `classification <= 3`
  - `delta_score < 0`

### Trigger Rule

The episodic second pass is triggered only when anchors are available and the first pass is uncertain or internally inconsistent.

Current trigger conditions:

- `boundary_confidence != high`
- or `delta_confidence != high` with non-zero residual evidence
- or DSAT vote count is 1 or 2
- or `boundary_score`, `classification`, and `delta_score` disagree in direction

This is intended to avoid paying a second LLM call for easy cases while still targeting boundary-sensitive cases.

### Episodic Refinement

When triggered, the system retrieves boundary-paired anchors:

- DSAT-side evidence: similar historical turns with score `<=3`
- SAT-side evidence: similar historical turns with score `>=4`

The second-pass prompt asks the model to output:

- `classification`: 3 or 4 only
- `closest_evidence_side`: `dsat`, `sat`, or `mixed`
- `evidence_match_confidence`: `low`, `medium`, or `high`
- `reason`
- `analysis`

The prompt explicitly says not to mechanically copy anchor scores. The key judgment is whether the target reply shares the same concrete failure/success mode as the retrieved side.

### Program-Side Application

The second pass is applied conservatively:

- `closest_evidence_side=dsat`, `confidence=high`: force final score `<=3`
- `closest_evidence_side=dsat`, `confidence=medium`: force `<=3` only if first pass already had at least one DSAT signal
- `closest_evidence_side=sat`, `confidence=high`: force final score `>=4`
- `closest_evidence_side=sat`, `confidence=medium`: force `>=4` only if first pass boundary score was already 4
- `mixed` or low confidence: keep first-pass score

## Diagnostic Output Fields

The output keeps all existing HPD fields and adds:

- `analysis_episodic_refine`
- `episodic_refine_triggered`
- `episodic_refine_applied`
- `episodic_refine_initial_score`
- `episodic_refine_initial_reason`
- `episodic_refine_first_pass_dsat_votes`
- `episodic_closest_evidence_side`
- `episodic_evidence_match_confidence`
- `episodic_refine_boundary_score`
- `episodic_refine_reason`

Anchor diagnostics are still included when anchors are retrieved:

- `n_anchors_retrieved`
- `anchor_scores`
- `anchor_tasks`
- `anchor_evidence_roles`

## Recommended Smoke Test

```bash
cd detection
model=Qwen/Qwen3-8B \
memory_update_mode=none \
memory_version=v2 \
turn_eval_prompt_version=history_prior_delta_v3_episodic_twopass \
n_anchors=4 \
limit_users=5 \
output_jsonl=outputs/personalized/qwen3_test_v2_none_hpd_v3_episodic_twopass_anchor4_u5.jsonl \
bash scripts/collect_personalized.sh
```

For vLLM:

```bash
cd detection
model=Qwen/Qwen3-8B \
vllm_base_url=http://localhost:8000/v1 \
memory_update_mode=none \
memory_version=v2 \
turn_eval_prompt_version=history_prior_delta_v3_episodic_twopass \
n_anchors=4 \
limit_users=5 \
output_jsonl=outputs/personalized/qwen3_test_v2_none_hpd_v3_episodic_twopass_anchor4_u5.jsonl \
bash scripts/collect_personalized_vllm.sh
```

After the smoke test, compare against:

- `history_prior_delta_v3_none_n3_limit20.jsonl`
- `history_prior_delta_v3_1_none_n3_limit20.jsonl`
- `qwen3_test_v2_none_hpd_v3_episodic_anchor4_u20.jsonl`

Key diagnostics to inspect before scaling:

- parse failure rate for `history_prior_delta_v3_episodic_refine`
- `episodic_refine_triggered` ratio
- `episodic_refine_applied` ratio
- distribution of `episodic_closest_evidence_side`
- DSAT recall / false-SAT tradeoff

