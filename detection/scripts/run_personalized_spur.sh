#!/usr/bin/env bash
# Run SPUR on the current personalized cross-task split.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

model="${model:-gpt-4o-mini}"
base_url="${base_url:-}"
api_key="${api_key:-}"
variant="${variant:-direct}"
k_rubrics="${k_rubrics:-10}"
max_extract_per_label="${max_extract_per_label:-150}"
max_workers="${max_workers:-8}"
output_dir="${output_dir:-outputs/spur_personalized}"
output_jsonl="${output_jsonl:-}"
metrics_json="${metrics_json:-}"
split_seed="${split_seed:-42}"
train_ratio="${train_ratio:-0.2}"
history_window_size="${history_window_size:-5}"
limit_test="${limit_test:-0}"
limit_train="${limit_train:-0}"
seed="${seed:-42}"
embedding_model="${embedding_model:-text-embedding-ada-002}"
embedding_batch_size="${embedding_batch_size:-64}"

args=(
    --model "$model"
    --base_url "$base_url"
    --api_key "$api_key"
    --variant "$variant"
    --k_rubrics "$k_rubrics"
    --max_extract_per_label "$max_extract_per_label"
    --max_workers "$max_workers"
    --output_dir "$output_dir"
    --split_seed "$split_seed"
    --train_ratio "$train_ratio"
    --history_window_size "$history_window_size"
    --seed "$seed"
    --embedding_model "$embedding_model"
    --embedding_batch_size "$embedding_batch_size"
)

[ -n "$output_jsonl" ] && args+=(--output_jsonl "$output_jsonl")
[ -n "$metrics_json" ] && args+=(--metrics_json "$metrics_json")
[ "$limit_test" -gt 0 ] && args+=(--limit_test "$limit_test")
[ "$limit_train" -gt 0 ] && args+=(--limit_train "$limit_train")
[ "${skip_phase1:-0}" = "1" ] && args+=(--skip_phase1)
[ "${skip_phase2:-0}" = "1" ] && args+=(--skip_phase2)
[ "${only_eval:-0}" = "1" ] && args+=(--only_eval)
if [ -n "${target_tasks:-}" ]; then
    # shellcheck disable=SC2206
    args+=(--target_tasks $target_tasks)
fi

echo "=========================================="
echo " Personalized SPUR"
echo "  model                 = $model"
echo "  base_url              = ${base_url:-default client}"
echo "  variant               = $variant"
echo "  k_rubrics             = $k_rubrics"
echo "  max_extract_per_label = $max_extract_per_label"
echo "  output_dir            = $output_dir"
echo "=========================================="

python -m eval.spur.personalized "${args[@]}"
