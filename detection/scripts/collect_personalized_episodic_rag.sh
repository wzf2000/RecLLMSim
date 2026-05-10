#!/usr/bin/env bash
# Raw episodic-memory RAG personalized satisfaction prediction.
#
# Usage from detection/:
#   model=gpt-4o-mini bash scripts/collect_personalized_episodic_rag.sh
#
# Local vLLM/OpenAI-compatible endpoint:
#   model=Qwen/Qwen3-8B vllm_base_url=http://localhost:8000/v1 \
#     bash scripts/collect_personalized_episodic_rag.sh

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

model="${model:-gpt-4o-mini}"
split="${split:-test}"
train_ratio="${train_ratio:-0.2}"
split_seed="${split_seed:-42}"
min_history_sessions="${min_history_sessions:-1}"
limit="${limit:-0}"
limit_users="${limit_users:-0}"
user_offset="${user_offset:-0}"
output_jsonl="${output_jsonl:-}"
max_workers="${max_workers:-4}"
history_window_size="${history_window_size:-5}"
retrieval_strategy="${retrieval_strategy:-boundary_paired}"
top_k="${top_k:-6}"
turn_eval_prompt_version="${turn_eval_prompt_version:-episodic_rag}"
vllm_base_url="${vllm_base_url:-}"
vllm_api_key="${vllm_api_key:-EMPTY}"

args=(
    --model "$model"
    --split "$split"
    --train_ratio "$train_ratio"
    --split_seed "$split_seed"
    --min_history_sessions "$min_history_sessions"
    --max_workers "$max_workers"
    --history_window_size "$history_window_size"
    --retrieval_strategy "$retrieval_strategy"
    --top_k "$top_k"
    --turn_eval_prompt_version "$turn_eval_prompt_version"
)

[ -n "$output_jsonl" ] && args+=(--output_jsonl "$output_jsonl")
[ "$limit" -gt 0 ] && args+=(--limit "$limit")
[ "$limit_users" -gt 0 ] && args+=(--limit_users "$limit_users" --user_offset "$user_offset")
[ -n "$vllm_base_url" ] && args+=(--vllm_base_url "$vllm_base_url" --vllm_api_key "$vllm_api_key")

if [ -n "${target_tasks:-}" ]; then
    args+=(--target_tasks $target_tasks)
fi

if [ "$#" -gt 0 ]; then
    args+=("$@")
fi

echo "=========================================="
echo " Episodic-RAG Personalized Prediction"
echo "  model        = $model"
echo "  split        = $split"
echo "  strategy     = $retrieval_strategy"
echo "  top_k        = $top_k"
echo "  prompt       = $turn_eval_prompt_version"
echo "  limit_users  = $limit_users (offset=$user_offset)"
echo "  max_workers  = $max_workers"
echo "  vllm_url     = ${vllm_base_url:-<openai/default>}"
echo "=========================================="

python trace/collect_personalized_episodic_rag.py "${args[@]}"
