#!/usr/bin/env bash
# Score URS static replay responses with a URS session-level satisfaction judge.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

input_jsonl="${input_jsonl:?missing input_jsonl=outputs/urs_static_replay/...responses.jsonl}"
output_jsonl="${output_jsonl:-}"
judge_model="${judge_model:?missing judge_model=...}"
memory_model="${memory_model:-}"
judge_base_url="${judge_base_url:-}"
judge_api_key="${judge_api_key:-}"
memory_base_url="${memory_base_url:-}"
memory_api_key="${memory_api_key:-}"
judge_config="${judge_config:-}"
urs_prompt_version="${urs_prompt_version:-v2}"
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
memory_cache_dir="${memory_cache_dir:-outputs/urs/memory_cache}"

args=(
    --input_jsonl "$input_jsonl"
    --judge_model "$judge_model"
    --urs_prompt_version "$urs_prompt_version"
    --split "$split"
    --train_ratio "$train_ratio"
    --split_seed "$split_seed"
    --min_history_sessions "$min_history_sessions"
    --max_workers "$max_workers"
    --memory_cache_dir "$memory_cache_dir"
    --languages $languages
)

[ -n "$output_jsonl" ] && args+=(--output_jsonl "$output_jsonl")
[ -n "$memory_model" ] && args+=(--memory_model "$memory_model")
[ -n "$judge_base_url" ] && args+=(--judge_base_url "$judge_base_url")
[ -n "$judge_api_key" ] && args+=(--judge_api_key "$judge_api_key")
[ -n "$memory_base_url" ] && args+=(--memory_base_url "$memory_base_url")
[ -n "$memory_api_key" ] && args+=(--memory_api_key "$memory_api_key")
[ -n "$judge_config" ] && args+=(--judge_config "$judge_config")
[ "$limit" -gt 0 ] && args+=(--limit "$limit")
[ "$limit_users" -gt 0 ] && args+=(--limit_users "$limit_users" --user_offset "$user_offset")
if [ "${no_memory:-0}" = "1" ]; then
    args+=(--no_memory)
fi
if [ -n "$target_intents" ]; then
    # shellcheck disable=SC2206
    args+=(--target_intents $target_intents)
fi

echo "=========================================="
echo " URS Static Replay Scoring"
echo "  input       = $input_jsonl"
echo "  judge_model = $judge_model"
echo "  memory_model= ${memory_model:-$judge_model}"
echo "  prompt_ver  = $urs_prompt_version"
echo "  languages   = $languages"
echo "  max_workers = $max_workers"
echo "=========================================="

python trace/score_urs_static_replay.py "${args[@]}"
