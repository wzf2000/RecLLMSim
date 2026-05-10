# Personalized SPUR Baseline

Date: 2026-05-10

This note records the implementation of a SPUR baseline adapted to the current
personalized cross-task satisfaction prediction setting.

## Existing Limitation

The existing SPUR implementation under `detection/eval/spur/cli.py` uses the old
global user-group split over all turn records. It is useful as a generic
SAT/DSAT baseline, but its test rows are not aligned with the current
personalized test sample ids used by Qwen memory-agent experiments.

## New Implementation

Added:

- `detection/eval/spur/personalized.py`
- `detection/scripts/run_personalized_spur.sh`

The new script:

1. Builds train rows from `build_personalized_samples(split="train")`.
2. Builds test rows from `build_personalized_samples(split="test")`.
3. Extracts SAT/DSAT rubric candidates from train users.
4. Summarizes global SAT/DSAT rubrics.
5. Applies rubrics to the personalized test rows.
6. Exports `eval/personalized.py`-compatible JSONL.

Default output maps:

- SPUR `SAT` -> `pred_score=4`
- SPUR `DSAT` -> `pred_score=3`

This means the baseline should be interpreted mainly as a 3/4 boundary baseline,
not a full 1-5 predictor.

## Supported Variants

The default and recommended first run is:

- `variant=direct`: LLM directly applies the learned SPUR rubrics.

Optional variants are implemented for later ablation:

- `variant=rubric_lr`: logistic regression over rubric-match features.
- `variant=embedding_lr`: logistic regression over embedding features.
- `variant=combined`: logistic regression over rubric-match + embedding features.

Only `direct` is needed for the first baseline result.

## Example Command

```bash
cd detection

conda activate chat

model=gpt-4o-mini \
variant=direct \
k_rubrics=10 \
max_extract_per_label=150 \
max_workers=8 \
output_dir=outputs/spur_personalized/gpt4o_mini_direct \
output_jsonl=outputs/personalized/spur_direct_gpt4o_mini_personalized_test.jsonl \
metrics_json=outputs/personalized/spur_direct_gpt4o_mini_personalized_test_metrics.json \
bash scripts/run_personalized_spur.sh
```

Then evaluate with:

```bash
result_file=outputs/personalized/spur_direct_gpt4o_mini_personalized_test.jsonl \
bash scripts/eval_personalized.sh
```

## Verified Data Alignment

Using the default split:

- train blocks: 85
- train users: 22
- train rows: 1413
- test blocks: 356
- test users: 90
- test rows: 6474

The emitted JSONL contains the original personalized `sample_id`, `user`,
`target_task`, `gold_score`, and `pred_score` fields.

