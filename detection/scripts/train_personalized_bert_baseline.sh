#!/usr/bin/env bash
# Supervised BERT baseline on the personalized cross-task split.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

model_name="${model_name:-bert-base-chinese}"
head="${head:-ordinal}"
epochs="${epochs:-5}"
batch_size="${batch_size:-16}"
eval_batch_size="${eval_batch_size:-32}"
max_length="${max_length:-512}"
max_dialogue_turns="${max_dialogue_turns:-12}"
lr="${lr:-2e-5}"
ordinal_threshold="${ordinal_threshold:-0.5}"
dsat_weight="${dsat_weight:-1.0}"
selection_metric="${selection_metric:-mae}"
train_ratio="${train_ratio:-0.2}"
split_seed="${split_seed:-42}"
valid_ratio="${valid_ratio:-0.2}"
checkpoint_dir="${checkpoint_dir:-ckpts/supervised_bert_personalized}"
output_jsonl="${output_jsonl:-}"
metrics_json="${metrics_json:-}"
eval_checkpoint="${eval_checkpoint:-}"

args=(
    --model_name "$model_name"
    --head "$head"
    --epochs "$epochs"
    --batch_size "$batch_size"
    --eval_batch_size "$eval_batch_size"
    --max_length "$max_length"
    --max_dialogue_turns "$max_dialogue_turns"
    --lr "$lr"
    --ordinal_threshold "$ordinal_threshold"
    --dsat_weight "$dsat_weight"
    --selection_metric "$selection_metric"
    --train_ratio "$train_ratio"
    --split_seed "$split_seed"
    --valid_ratio "$valid_ratio"
    --checkpoint_dir "$checkpoint_dir"
)

[ -n "$output_jsonl" ] && args+=(--output_jsonl "$output_jsonl")
[ -n "$metrics_json" ] && args+=(--metrics_json "$metrics_json")
[ -n "$eval_checkpoint" ] && args+=(--eval_checkpoint "$eval_checkpoint")
[ "${no_profile:-0}" = "1" ] && args+=(--no_profile)
[ "${no_history_summary:-0}" = "1" ] && args+=(--no_history_summary)
if [ -n "${target_tasks:-}" ]; then
    # shellcheck disable=SC2206
    args+=(--target_tasks $target_tasks)
fi

echo "=========================================="
echo " Supervised Personalized BERT Baseline"
echo "  model_name      = $model_name"
echo "  head            = $head"
echo "  epochs          = $epochs"
echo "  batch_size      = $batch_size"
echo "  max_length      = $max_length"
echo "  ordinal_thresh  = $ordinal_threshold"
echo "  dsat_weight     = $dsat_weight"
echo "  selection_metric= $selection_metric"
echo "  checkpoint_dir  = $checkpoint_dir"
echo "  output_jsonl    = ${output_jsonl:-auto}"
echo "=========================================="

python predictor/personalized_bert_baseline.py "${args[@]}"
