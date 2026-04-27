#!/usr/bin/env bash
# ============================================================
# collect_uss.sh — USS cross-dataset 满意度感知 Agent 推理
# 从 detection/ 目录运行：bash scripts/collect_uss.sh
# ============================================================
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

# ── 可配置参数（环境变量覆盖）─────────────────────────────────
model="${model:-gpt-4o}"
mode="${mode:-warmup}"                    # warmup (R1) / population (R2)
subsets="${subsets:-CCPE SGD MWOZ ReDial JDDC}"
test_split="${test_split:-test}"
train_split="${train_split:-train}"
memory_update_mode="${memory_update_mode:-none}"
history_window_size="${history_window_size:-5}"
max_workers="${max_workers:-8}"
memory_cache_dir="${memory_cache_dir:-outputs/uss/memory_cache}"
output_jsonl="${output_jsonl:-}"
turn_eval_prompt_version="${turn_eval_prompt_version:-v2}"
n_anchors="${n_anchors:-0}"

# R1
warmup_turns="${warmup_turns:-5}"
min_warmup_turns="${min_warmup_turns:-3}"
min_target_turns="${min_target_turns:-1}"

# R2
n_population_dialogues="${n_population_dialogues:-8}"
population_seed="${population_seed:-42}"

# 调试 / 子集
limit="${limit:-0}"
limit_blocks="${limit_blocks:-0}"

extra_args=""
if [ "${save_memory_snapshots:-0}" = "1" ]; then
    extra_args="$extra_args --save_memory_snapshots"
fi
if [ "${no_memory:-0}" = "1" ]; then
    memory_update_mode="none"
    extra_args="$extra_args --no_memory"
fi

vllm_base_url="${vllm_base_url:-}"
vllm_api_key="${vllm_api_key:-EMPTY}"
if [ -n "$vllm_base_url" ]; then
    extra_args="$extra_args --vllm_base_url $vllm_base_url --vllm_api_key $vllm_api_key"
fi

args=(
    --mode "$mode"
    --model "$model"
    --subsets $subsets
    --test_split "$test_split"
    --train_split "$train_split"
    --memory_update_mode "$memory_update_mode"
    --history_window_size "$history_window_size"
    --max_workers "$max_workers"
    --memory_cache_dir "$memory_cache_dir"
    --turn_eval_prompt_version "$turn_eval_prompt_version"
    --n_anchors "$n_anchors"
    --warmup_turns "$warmup_turns"
    --min_warmup_turns "$min_warmup_turns"
    --min_target_turns "$min_target_turns"
    --n_population_dialogues "$n_population_dialogues"
    --population_seed "$population_seed"
)
[ -n "$output_jsonl" ] && args+=(--output_jsonl "$output_jsonl")
[ "$limit" -gt 0 ] && args+=(--limit "$limit")
[ "$limit_blocks" -gt 0 ] && args+=(--limit_blocks "$limit_blocks")

echo "=========================================="
echo " USS cross-dataset 满意度感知 Agent 推理"
echo "  mode               = $mode"
echo "  model              = $model"
echo "  subsets            = $subsets"
echo "  memory_update_mode = $memory_update_mode"
if [ "$mode" = "warmup" ]; then
    echo "  warmup_turns       = $warmup_turns"
else
    echo "  n_population       = $n_population_dialogues  (seed=$population_seed)"
fi
echo "  prompt_version     = $turn_eval_prompt_version"
echo "  max_workers        = $max_workers"
echo "=========================================="

python trace/collect_uss.py "${args[@]}" $extra_args

echo "Done."
