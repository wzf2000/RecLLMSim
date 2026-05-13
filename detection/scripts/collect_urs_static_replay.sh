#!/usr/bin/env bash
# URS session-level static replay response collection.

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
languages="${languages:-zh en}"
target_intents="${target_intents:-}"
min_history_sessions="${min_history_sessions:-1}"
limit="${limit:-0}"
limit_users="${limit_users:-0}"
user_offset="${user_offset:-0}"
max_workers="${max_workers:-4}"
temperature="${temperature:-0.7}"
max_tokens="${max_tokens:-4096}"
timeout="${timeout:-120}"
output_jsonl="${output_jsonl:-}"
selection_mode="${selection_mode:-hard}"
hard_global_budget="${hard_global_budget:-300}"
hard_max_per_user="${hard_max_per_user:-4}"
hard_score_quota="${hard_score_quota:-1:25,2:50,3:75,4:75}"
replay_context_mode="${replay_context_mode:-raw}"
replay_granularity="${replay_granularity:-first_user}"
dialogue_memory_top_k="${dialogue_memory_top_k:-4}"
dialogue_memory_max_chars_per_item="${dialogue_memory_max_chars_per_item:-700}"
dialogue_memory_local_history_size="${dialogue_memory_local_history_size:-4}"

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
    --selection_mode "$selection_mode"
    --hard_global_budget "$hard_global_budget"
    --hard_max_per_user "$hard_max_per_user"
    --hard_score_quota "$hard_score_quota"
    --replay_context_mode "$replay_context_mode"
    --replay_granularity "$replay_granularity"
    --dialogue_memory_top_k "$dialogue_memory_top_k"
    --dialogue_memory_max_chars_per_item "$dialogue_memory_max_chars_per_item"
    --dialogue_memory_local_history_size "$dialogue_memory_local_history_size"
    --languages $languages
)

[ -n "$base_url" ] && args+=(--base_url "$base_url")
[ -n "$api_key" ] && args+=(--api_key "$api_key")
[ -n "$output_jsonl" ] && args+=(--output_jsonl "$output_jsonl")
[ "$limit" -gt 0 ] && args+=(--limit "$limit")
[ "$limit_users" -gt 0 ] && args+=(--limit_users "$limit_users" --user_offset "$user_offset")
if [ -n "$target_intents" ]; then
    # shellcheck disable=SC2206
    args+=(--target_intents $target_intents)
fi

echo "=========================================="
echo " URS Static Replay Collection"
echo "  candidate model = $model"
echo "  base_url        = ${base_url:-default API}"
echo "  split           = $split"
echo "  languages       = $languages"
echo "  context_mode    = $replay_context_mode"
echo "  granularity     = $replay_granularity"
echo "  selection_mode  = $selection_mode"
echo "  hard_budget     = $hard_global_budget"
echo "  max_workers     = $max_workers"
echo "=========================================="

python trace/collect_urs_static_replay.py "${args[@]}"
