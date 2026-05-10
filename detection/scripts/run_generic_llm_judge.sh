#!/usr/bin/env bash
# Run generic LLM-as-judge baselines on the personalized split.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

model="${model:?missing model=...}"
base_url="${base_url:-}"
api_key="${api_key:-}"
variant="${variant:-zero_shot}"
split="${split:-test}"
train_ratio="${train_ratio:-0.2}"
seed="${seed:-42}"
min_history_sessions="${min_history_sessions:-1}"
limit_users="${limit_users:-0}"
limit_turns="${limit_turns:-0}"
few_shot_n="${few_shot_n:-10}"
max_workers="${max_workers:-4}"
max_context_chars="${max_context_chars:-6000}"
temperature="${temperature:-0.2}"
timeout="${timeout:-90}"
output_jsonl="${output_jsonl:-}"
metrics_json="${metrics_json:-}"

args=(
    --model "$model"
    --base_url "$base_url"
    --api_key "$api_key"
    --variant "$variant"
    --split "$split"
    --train_ratio "$train_ratio"
    --seed "$seed"
    --min_history_sessions "$min_history_sessions"
    --limit_users "$limit_users"
    --limit_turns "$limit_turns"
    --few_shot_n "$few_shot_n"
    --max_workers "$max_workers"
    --max_context_chars "$max_context_chars"
    --temperature "$temperature"
    --timeout "$timeout"
)

[ -n "$output_jsonl" ] && args+=(--output_jsonl "$output_jsonl")
[ -n "$metrics_json" ] && args+=(--metrics_json "$metrics_json")
if [ -n "${target_tasks:-}" ]; then
    # shellcheck disable=SC2206
    args+=(--target_tasks $target_tasks)
fi

echo "=========================================="
echo " Generic LLM-as-judge baseline"
echo "  model       = $model"
echo "  base_url    = ${base_url:-default API client}"
echo "  variant     = $variant"
echo "  split       = $split"
echo "  limit_users = $limit_users"
echo "  limit_turns = $limit_turns"
echo "=========================================="

python eval/generic_llm_judge.py "${args[@]}"
