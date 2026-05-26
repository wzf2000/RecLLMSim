#!/usr/bin/env bash
# ============================================================
# collect_urs_episodic_rag.sh — URS episodic-retrieval memory predictor
# Run from detection/: bash scripts/collect_urs_episodic_rag.sh
# ============================================================
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

model="${model:-gpt-4o}"
split="${split:-test}"
train_ratio="${train_ratio:-0.2}"
dev_ratio="${dev_ratio:-0.5}"
split_seed="${split_seed:-42}"
urs_prompt_version="${urs_prompt_version:-urs_episodic_task_guarded}"
retrieval_strategy="${retrieval_strategy:-boundary_paired}"
top_k="${top_k:-4}"
max_dialogue_chars="${max_dialogue_chars:-900}"
max_workers="${max_workers:-4}"
min_history_sessions="${min_history_sessions:-1}"
limit="${limit:-0}"
limit_users="${limit_users:-0}"
user_offset="${user_offset:-0}"
output_jsonl="${output_jsonl:-}"
languages="${languages:-zh en}"
target_intents="${target_intents:-}"

extra_args=""
if [ -n "$target_intents" ]; then
    extra_args="$extra_args --target_intents $target_intents"
fi

vllm_base_url="${vllm_base_url:-}"
vllm_api_key="${vllm_api_key:-EMPTY}"
if [ -n "$vllm_base_url" ]; then
    extra_args="$extra_args --vllm_base_url $vllm_base_url --vllm_api_key $vllm_api_key"
fi

args=(
    --model "$model"
    --split "$split"
    --train_ratio "$train_ratio"
    --dev_ratio "$dev_ratio"
    --split_seed "$split_seed"
    --urs_prompt_version "$urs_prompt_version"
    --retrieval_strategy "$retrieval_strategy"
    --top_k "$top_k"
    --max_dialogue_chars "$max_dialogue_chars"
    --max_workers "$max_workers"
    --min_history_sessions "$min_history_sessions"
    --languages $languages
)
[ -n "$output_jsonl" ] && args+=(--output_jsonl "$output_jsonl")
[ "$limit" -gt 0 ] && args+=(--limit "$limit")
[ "$limit_users" -gt 0 ] && args+=(--limit_users "$limit_users" --user_offset "$user_offset")

echo "=========================================="
echo " URS Episodic Retrieval Memory Prediction"
echo "  model              = $model"
echo "  split              = $split  (train_ratio=$train_ratio)"
echo "  languages          = $languages"
echo "  prompt_version     = $urs_prompt_version"
echo "  retrieval_strategy = $retrieval_strategy"
echo "  top_k              = $top_k"
echo "  limit_users        = $limit_users  (offset=$user_offset)"
echo "  max_workers        = $max_workers"
echo "=========================================="

python trace/collect_urs_episodic_rag.py "${args[@]}" $extra_args

echo "Done."
