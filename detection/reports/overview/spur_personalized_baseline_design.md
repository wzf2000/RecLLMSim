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

For rebuttal analysis, the runner also supports `score_mapping=trinary_24`, which keeps the same binary SPUR decisions but maps SPUR `DSAT` to `pred_score=2`.
This allows a trinary `1--2=DSAT`, `3=Neutral`, `4--5=SAT` sensitivity check without re-running rubric induction or LLM scoring.

The runner also supports a full 3-level SPUR setting with `label_schema=trinary`.
In this mode, train labels are constructed as `score 1--2 -> DSAT`, `score 3 -> NEUTRAL`, and `score 4--5 -> SAT`.
Phase 1 extracts three groups of rubric candidates, Phase 2 summarizes three rubric sets, and Phase 3 predicts one of `DSAT`, `NEUTRAL`, and `SAT`.
The compatible JSONL maps these predictions to scores `2`, `3`, and `4`, respectively.

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

Full 3-level SPUR command for the rebuttal run:

```bash
cd /data/wangzhefan/RecLLMSim
conda activate chat
model='Qwen/Qwen3-8B' \
base_url='http://localhost:8001/v1' \
api_key='EMPTY' \
variant='direct' \
label_schema='trinary' \
score_mapping='trinary_24' \
max_extract_per_label=150 \
max_workers=4 \
output_dir='outputs/spur_personalized/qwen3_8b_trinary_direct' \
output_jsonl='outputs/personalized/spur_trinary_direct_qwen3_8b_personalized_test.jsonl' \
metrics_json='outputs/personalized/spur_trinary_direct_qwen3_8b_personalized_test_metrics.json' \
bash detection/scripts/run_personalized_spur.sh
```

Evaluate the full 3-level run with:

```bash
result_file='outputs/personalized/spur_trinary_direct_qwen3_8b_personalized_test.jsonl' \
output_json='outputs/personalized/spur_trinary_direct_qwen3_8b_personalized_test_eval.json' \
bash detection/scripts/eval_personalized.sh
```

Evaluate the example binary run with:

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
