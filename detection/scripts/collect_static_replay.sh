#!/usr/bin/env bash
# Static replay candidate response collection.
#
# From detection/:
#   model=Qwen/Qwen3-8B \
#   base_url=http://localhost:8000/v1 \
#   api_key=EMPTY \
#   bash scripts/collect_static_replay.sh

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

model="${model:?missing model=candidate/model}"
base_url="${base_url:-}"
api_key="${api_key:-}"
split="${split:-test}"
train_ratio="${train_ratio:-0.2}"
split_seed="${split_seed:-42}"
min_history_sessions="${min_history_sessions:-1}"
limit="${limit:-0}"
limit_users="${limit_users:-0}"
user_offset="${user_offset:-0}"
max_workers="${max_workers:-4}"
temperature="${temperature:-0.7}"
max_tokens="${max_tokens:-1024}"
timeout="${timeout:-120}"
output_jsonl="${output_jsonl:-}"
replay_context_mode="${replay_context_mode:-raw}"  # raw / task / profile / dialogue_memory_tfidf / dialogue_memory_diverse
dialogue_memory_top_k="${dialogue_memory_top_k:-4}"
dialogue_memory_max_chars_per_item="${dialogue_memory_max_chars_per_item:-700}"
dialogue_memory_local_history_size="${dialogue_memory_local_history_size:-4}"
selection_mode="${selection_mode:-full}"  # full / hard / filter
hard_max_per_block="${hard_max_per_block:-3}"
hard_max_per_session="${hard_max_per_session:-1}"
hard_global_budget="${hard_global_budget:-300}"
hard_positive_controls_per_block="${hard_positive_controls_per_block:-1}"
hard_min_turn_idx="${hard_min_turn_idx:-1}"
hard_score_quota="${hard_score_quota:-1:25,2:50,4:75}"
hard_min_per_user="${hard_min_per_user:-1}"

args=(
    --model "$model"
    --split "$split"
    --train_ratio "$train_ratio"
    --split_seed "$split_seed"
    --min_history_sessions "$min_history_sessions"
    --max_workers "$max_workers"
    --temperature "$temperature"
    --max_tokens "$max_tokens"
    --timeout "$timeout"
    --replay_context_mode "$replay_context_mode"
    --dialogue_memory_top_k "$dialogue_memory_top_k"
    --dialogue_memory_max_chars_per_item "$dialogue_memory_max_chars_per_item"
    --dialogue_memory_local_history_size "$dialogue_memory_local_history_size"
    --selection_mode "$selection_mode"
    --hard_max_per_block "$hard_max_per_block"
    --hard_max_per_session "$hard_max_per_session"
    --hard_global_budget "$hard_global_budget"
    --hard_positive_controls_per_block "$hard_positive_controls_per_block"
    --hard_min_turn_idx "$hard_min_turn_idx"
    --hard_score_quota "$hard_score_quota"
    --hard_min_per_user "$hard_min_per_user"
)

[ -n "$base_url" ] && args+=(--base_url "$base_url")
[ -n "$api_key" ] && args+=(--api_key "$api_key")
[ -n "$output_jsonl" ] && args+=(--output_jsonl "$output_jsonl")
[ "$limit" -gt 0 ] && args+=(--limit "$limit")
[ "$limit_users" -gt 0 ] && args+=(--limit_users "$limit_users" --user_offset "$user_offset")
if [ -n "${target_tasks:-}" ]; then
    # shellcheck disable=SC2206
    args+=(--target_tasks $target_tasks)
fi

echo "=========================================="
echo " Static Replay Collection"
echo "  candidate model = $model"
echo "  base_url        = ${base_url:-default API}"
echo "  split           = $split"
echo "  context_mode    = $replay_context_mode"
echo "  memory_top_k    = $dialogue_memory_top_k"
echo "  selection_mode  = $selection_mode"
echo "  hard_budget     = $hard_global_budget"
echo "  hard_min_turn   = $hard_min_turn_idx"
echo "  hard_quota      = $hard_score_quota"
echo "  hard_min_user   = $hard_min_per_user"
echo "  limit_users     = $limit_users (offset=$user_offset)"
echo "  max_workers     = $max_workers"
echo "=========================================="

python trace/collect_static_replay.py "${args[@]}"
