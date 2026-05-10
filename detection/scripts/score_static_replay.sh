#!/usr/bin/env bash
# Score static replay candidate responses with a satisfaction judge.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

input_jsonl="${input_jsonl:?missing input_jsonl=outputs/static_replay/...jsonl}"
judge_model="${judge_model:?missing judge_model=...}"
memory_model="${memory_model:-}"
judge_base_url="${judge_base_url:-}"
judge_api_key="${judge_api_key:-}"
memory_base_url="${memory_base_url:-}"
memory_api_key="${memory_api_key:-}"
judge_config="${judge_config:-}"
split="${split:-test}"
train_ratio="${train_ratio:-0.2}"
split_seed="${split_seed:-42}"
min_history_sessions="${min_history_sessions:-1}"
limit="${limit:-0}"
limit_users="${limit_users:-0}"
user_offset="${user_offset:-0}"
max_workers="${max_workers:-4}"
history_window_size="${history_window_size:-5}"
memory_cache_dir="${memory_cache_dir:-outputs/personalized/memory_cache}"
memory_version="${memory_version:-v2}"
turn_eval_prompt_version="${turn_eval_prompt_version:-v2}"
no_memory="${no_memory:-0}"
output_jsonl="${output_jsonl:-}"

args=(
    --input_jsonl "$input_jsonl"
    --judge_model "$judge_model"
    --split "$split"
    --train_ratio "$train_ratio"
    --split_seed "$split_seed"
    --min_history_sessions "$min_history_sessions"
    --max_workers "$max_workers"
    --history_window_size "$history_window_size"
    --memory_cache_dir "$memory_cache_dir"
    --memory_version "$memory_version"
    --turn_eval_prompt_version "$turn_eval_prompt_version"
)

[ -n "$memory_model" ] && args+=(--memory_model "$memory_model")

[ -n "$judge_base_url" ] && args+=(--judge_base_url "$judge_base_url")
[ -n "$judge_api_key" ] && args+=(--judge_api_key "$judge_api_key")
[ -n "$memory_base_url" ] && args+=(--memory_base_url "$memory_base_url")
[ -n "$memory_api_key" ] && args+=(--memory_api_key "$memory_api_key")
[ -n "$judge_config" ] && args+=(--judge_config "$judge_config")
[ -n "$output_jsonl" ] && args+=(--output_jsonl "$output_jsonl")
[ "$limit" -gt 0 ] && args+=(--limit "$limit")
[ "$limit_users" -gt 0 ] && args+=(--limit_users "$limit_users" --user_offset "$user_offset")
[ "$no_memory" = "1" ] && args+=(--no_memory)
if [ -n "${target_tasks:-}" ]; then
    # shellcheck disable=SC2206
    args+=(--target_tasks $target_tasks)
fi

echo "=========================================="
echo " Static Replay Scoring"
echo "  input       = $input_jsonl"
echo "  judge model = $judge_model"
echo "  memory model= ${memory_model:-$judge_model}"
echo "  judge URL   = ${judge_base_url:-default API}"
if [ -n "$memory_base_url" ]; then
    memory_url_label="$memory_base_url"
elif [ "${memory_model:-$judge_model}" = "$judge_model" ]; then
    memory_url_label="${judge_base_url:-default API}"
else
    memory_url_label="default API"
fi
echo "  memory URL  = $memory_url_label"
echo "  memory      = ${memory_version}/${turn_eval_prompt_version}"
echo "  no_memory   = $no_memory"
echo "=========================================="

python trace/score_static_replay.py "${args[@]}"
