#!/usr/bin/env bash
# 对已有 JSONL 预测做 post-hoc 校准（按用户历史分数分布）。
#
# 用法（从 detection/ 目录）：
#   input=outputs/personalized/gpt-4o-mini_test_none_v2.jsonl \
#   method=cdf \
#   bash scripts/calibrate.sh
#
# 可选环境变量：
#   input               待校准的 JSONL（必填）
#   output              输出 JSONL（留空 → 自动加 _calCDF / _calMS 后缀）
#   method              identity / mean_shift / cdf（默认 cdf）
#   memory_cache_dir    默认 outputs/personalized/memory_cache
#   min_history_turns   用户历史最少 turn 数，低于则降级为 identity（默认 5）

set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

input="${input:?missing input=path/to/result.jsonl}"
method="${method:-cdf}"
output="${output:-}"
memory_cache_dir="${memory_cache_dir:-outputs/personalized/memory_cache}"
min_history_turns="${min_history_turns:-5}"

args=(
    --input_jsonl "$input"
    --method "$method"
    --memory_cache_dir "$memory_cache_dir"
    --min_history_turns "$min_history_turns"
)
[ -n "$output" ] && args+=(--output_jsonl "$output")

python eval/calibrate.py "${args[@]}"
