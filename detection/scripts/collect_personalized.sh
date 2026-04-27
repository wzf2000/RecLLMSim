#!/usr/bin/env bash
# ============================================================
# collect_personalized.sh — 个性化满意度感知 Agent 推理
# 从 detection/ 目录运行：bash scripts/collect_personalized.sh
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
history_window_size="${history_window_size:-5}"
max_workers="${max_workers:-8}"
memory_cache_dir="${memory_cache_dir:-outputs/personalized/memory_cache}"
memory_version="${memory_version:-v2}"  # v2 / v3
min_history_sessions="${min_history_sessions:-1}"
limit="${limit:-0}"                    # <=0 = 不限制，>0 = 调试用
limit_users="${limit_users:-0}"        # <=0 = 不限制，>0 = 按用户数量截取
user_offset="${user_offset:-0}"        # 用户子集起始偏移
output_jsonl="${output_jsonl:-}"        # 留空则自动命名
n_anchors="${n_anchors:-0}"            # >0 时每轮插入 k 个 few-shot anchor turns
turn_eval_prompt_version="${turn_eval_prompt_version:-v2}"  # v2 / v3 / v3_1 / qwen_short / boundary_34 / boundary_34_refute / boundary_34_refute_v2 / boundary_34_selective_refute / boundary_34_selective_refute_v2 / boundary_34_selective_refute_v2_fullscale / boundary_34_selective_refute_v3 / boundary_34_selective_refute_v4

# 可选 flag
extra_args=""
if [ "${save_memory_snapshots:-0}" = "1" ]; then
    extra_args="$extra_args --save_memory_snapshots"
fi
if [ "${no_memory:-0}" = "1" ]; then
    # 无记忆 baseline 模式：memory_update_mode=none 且跳过 memory building
    memory_update_mode="none"
    extra_args="$extra_args --no_memory"
fi
if [ -n "${target_tasks:-}" ]; then
    extra_args="$extra_args --target_tasks $target_tasks"
fi

# ── 组装参数 ──────────────────────────────────────────────────
args=(
    --model "$model"
    --split "$split"
    --train_ratio "$train_ratio"
    --split_seed "$split_seed"
    --memory_update_mode "$memory_update_mode"
    --history_window_size "$history_window_size"
    --max_workers "$max_workers"
    --memory_cache_dir "$memory_cache_dir"
    --memory_version "$memory_version"
    --min_history_sessions "$min_history_sessions"
    --n_anchors "$n_anchors"
    --turn_eval_prompt_version "$turn_eval_prompt_version"
)
[ -n "$output_jsonl" ] && args+=(--output_jsonl "$output_jsonl")
[ "$limit" -gt 0 ] && args+=(--limit "$limit")
[ "$limit_users" -gt 0 ] && args+=(--limit_users "$limit_users" --user_offset "$user_offset")

# ── 运行 ──────────────────────────────────────────────────────
echo "=========================================="
echo " 个性化满意度感知 Agent 推理"
echo "  model              = $model"
echo "  split              = $split  (train_ratio=$train_ratio)"
echo "  memory_update_mode = $memory_update_mode"
echo "  memory_version     = $memory_version"
echo "  history_window     = $history_window_size turns"
echo "  n_anchors          = $n_anchors"
echo "  turn_eval_prompt   = $turn_eval_prompt_version"
echo "  limit_users        = $limit_users  (offset=$user_offset)"
echo "  max_workers        = $max_workers"
echo "=========================================="

python trace/collect_personalized.py "${args[@]}" $extra_args

echo "Done."
