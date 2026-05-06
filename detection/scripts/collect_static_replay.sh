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
replay_context_mode="${replay_context_mode:-raw}"  # raw / task / profile

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
echo "  limit_users     = $limit_users (offset=$user_offset)"
echo "  max_workers     = $max_workers"
echo "=========================================="

python trace/collect_static_replay.py "${args[@]}"
