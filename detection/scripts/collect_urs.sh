#!/usr/bin/env bash
# ============================================================
# collect_urs.sh — URS session-level 满意度感知 Agent 推理
# 从 detection/ 目录运行：bash scripts/collect_urs.sh
# ============================================================
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

# ── 可配置参数（通过环境变量覆盖）────────────────────────────
model="${model:-gpt-4o}"
split="${split:-test}"
train_ratio="${train_ratio:-0.2}"
split_seed="${split_seed:-42}"
memory_update_mode="${memory_update_mode:-per_session}"
urs_prompt_version="${urs_prompt_version:-v2}"
max_workers="${max_workers:-8}"
memory_cache_dir="${memory_cache_dir:-outputs/urs/memory_cache}"
min_history_sessions="${min_history_sessions:-1}"
limit="${limit:-0}"
limit_users="${limit_users:-0}"
user_offset="${user_offset:-0}"
output_jsonl="${output_jsonl:-}"
languages="${languages:-zh en}"           # 空格分隔，如 "zh" / "en" / "zh en"
target_intents="${target_intents:-}"      # 空则全部 7 类 intent

extra_args=""
if [ "${save_memory_snapshots:-0}" = "1" ]; then
    extra_args="$extra_args --save_memory_snapshots"
fi
if [ "${no_memory:-0}" = "1" ]; then
    memory_update_mode="none"
    extra_args="$extra_args --no_memory"
fi
if [ -n "$target_intents" ]; then
    extra_args="$extra_args --target_intents $target_intents"
fi

# ── vLLM 切换 ────────────────────────────────────────────────
vllm_base_url="${vllm_base_url:-}"
vllm_api_key="${vllm_api_key:-EMPTY}"
if [ -n "$vllm_base_url" ]; then
    extra_args="$extra_args --vllm_base_url $vllm_base_url --vllm_api_key $vllm_api_key"
fi

# ── 组装参数 ──────────────────────────────────────────────────
args=(
    --model "$model"
    --split "$split"
    --train_ratio "$train_ratio"
    --split_seed "$split_seed"
    --memory_update_mode "$memory_update_mode"
    --urs_prompt_version "$urs_prompt_version"
    --max_workers "$max_workers"
    --memory_cache_dir "$memory_cache_dir"
    --min_history_sessions "$min_history_sessions"
    --languages $languages
)
[ -n "$output_jsonl" ] && args+=(--output_jsonl "$output_jsonl")
[ "$limit" -gt 0 ] && args+=(--limit "$limit")
[ "$limit_users" -gt 0 ] && args+=(--limit_users "$limit_users" --user_offset "$user_offset")

# ── 运行 ──────────────────────────────────────────────────────
echo "=========================================="
echo " URS session-level 满意度感知 Agent 推理"
echo "  model              = $model"
echo "  split              = $split  (train_ratio=$train_ratio)"
echo "  languages          = $languages"
echo "  memory_update_mode = $memory_update_mode"
echo "  urs_prompt_version = $urs_prompt_version"
echo "  limit_users        = $limit_users  (offset=$user_offset)"
echo "  max_workers        = $max_workers"
echo "=========================================="

python trace/collect_urs.py "${args[@]}" $extra_args

echo "Done."
