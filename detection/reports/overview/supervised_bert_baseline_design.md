# Supervised BERT Personalized Baseline

Date: 2026-05-10

This note records the implementation of a supervised BERT baseline aligned with
the current personalized satisfaction prediction setting.

## Goal

The baseline trains on the project-level personalized train split users and
evaluates on the same personalized test sample ids used by Qwen memory-agent
experiments.

This differs from the older supervised predictor pipeline, which used a separate
user-group split over all satisfaction records and did not emit JSONL aligned
with `eval/personalized.py`.

## Implementation

Added:

- `detection/predictor/personalized_bert_baseline.py`
- `detection/scripts/train_personalized_bert_baseline.sh`

Default configuration:

- backbone: `bert-base-chinese`
- head: ordinal 1-5 satisfaction head
- train split: `build_personalized_samples(split="train", train_ratio=0.2, seed=42)`
- test split: `build_personalized_samples(split="test", train_ratio=0.2, seed=42)`
- train users: 22
- test users: 90
- train examples: 1413 turn-level assistant responses
- test examples: 6474 turn-level assistant responses

Input text contains:

- user profile
- source-task user history score summary
- target task name
- task context
- dialogue prefix up to and including the target assistant reply

The output JSONL is compatible with existing evaluation:

- `gold_score`
- `pred_score`
- `user`
- `target_task`
- `sample_id`

## Example Command

```bash
cd detection

conda activate chat

epochs=5 \
batch_size=16 \
eval_batch_size=32 \
output_jsonl=outputs/personalized/bert_supervised_ordinal_personalized_test.jsonl \
metrics_json=outputs/personalized/bert_supervised_ordinal_personalized_test_metrics.json \
bash scripts/train_personalized_bert_baseline.sh
```

Then evaluate with:

```bash
result_file=outputs/personalized/bert_supervised_ordinal_personalized_test.jsonl \
bash scripts/eval_personalized.sh
```

## Environment Note

The current `chat2` environment lacks `sklearn`, which is imported by
`lib.personalized_data`. The `chat` environment has the required dependency and
successfully runs the script help/data-building checks.

