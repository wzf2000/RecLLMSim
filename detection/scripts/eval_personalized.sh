#!/usr/bin/env bash
# ============================================================
# eval_personalized.sh — 个性化满意度预测评估
# 从 detection/ 目录运行：bash scripts/eval_personalized.sh
# ============================================================
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

# ── 使用示例 ──────────────────────────────────────────────────
#
# 1. 单文件评估
#    result_file=outputs/personalized/gpt-4o_test_per_session.jsonl \
#      bash scripts/eval_personalized.sh
#
# 2. 单文件 + baseline 对比（计算 Personalization Gain）
#    result_file=outputs/personalized/gpt-4o_test_per_session.jsonl \
#    baseline_file=outputs/personalized/gpt-4o_test_none.jsonl \
#      bash scripts/eval_personalized.sh
#
# 3. 多文件对比
#    result_files="with_memory=outputs/personalized/gpt-4o_test_per_session.jsonl no_memory=outputs/personalized/gpt-4o_test_none.jsonl" \
#    output_json=outputs/personalized/comparison.json \
#      bash scripts/eval_personalized.sh
# ─────────────────────────────────────────────────────────────

result_file="${result_file:-}"
baseline_file="${baseline_file:-}"
result_files="${result_files:-}"
output_json="${output_json:-}"
min_samples="${min_samples:-3}"

# ── 组装参数 ──────────────────────────────────────────────────
args=(--min_samples "$min_samples")

if [ -n "$result_file" ]; then
    args+=(--result_file "$result_file")
fi
if [ -n "$baseline_file" ]; then
    args+=(--baseline_file "$baseline_file")
fi
if [ -n "$result_files" ]; then
    # result_files 是空格分隔的 name=path 列表
    # shellcheck disable=SC2206
    args+=(--result_files $result_files)
fi
if [ -n "$output_json" ]; then
    args+=(--output_json "$output_json")
fi

# ── 运行 ──────────────────────────────────────────────────────
echo "=========================================="
echo " 个性化满意度预测评估"
echo "  result_file  = ${result_file:-（未指定）}"
echo "  baseline     = ${baseline_file:-（未指定）}"
echo "  result_files = ${result_files:-（未指定）}"
echo "  output_json  = ${output_json:-（不保存）}"
echo "=========================================="

python eval/personalized.py "${args[@]}"

echo "Done."
