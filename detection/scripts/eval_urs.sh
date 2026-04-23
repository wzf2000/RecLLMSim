#!/usr/bin/env bash
# ============================================================
# eval_urs.sh — URS session-level 满意度预测评估
# 从 detection/ 目录运行：bash scripts/eval_urs.sh
# ============================================================
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

# ── 使用示例 ──────────────────────────────────────────────────
#
# 1. 单文件评估
#    result_file=outputs/urs/gpt-4o_test_per_session.jsonl \
#      bash scripts/eval_urs.sh
#
# 2. 单文件 + no_memory baseline 对比（Personalization Gain）
#    result_file=outputs/urs/gpt-4o_test_per_session.jsonl \
#    baseline_file=outputs/urs/gpt-4o_test_no_memory.jsonl \
#      bash scripts/eval_urs.sh
#
# 3. 多文件对比（包括 CDF 校准前后）
#    result_files="no_mem=outputs/urs/gpt-4o_test_no_memory.jsonl \
#                  mem=outputs/urs/gpt-4o_test_per_session.jsonl \
#                  mem_cdf=outputs/urs/gpt-4o_test_per_session_calCDF.jsonl" \
#    output_json=outputs/urs/comparison.json \
#      bash scripts/eval_urs.sh
# ─────────────────────────────────────────────────────────────

result_file="${result_file:-}"
baseline_file="${baseline_file:-}"
result_files="${result_files:-}"
output_json="${output_json:-}"
min_samples="${min_samples:-3}"

args=(--min_samples "$min_samples")
[ -n "$result_file" ]   && args+=(--result_file "$result_file")
[ -n "$baseline_file" ] && args+=(--baseline_file "$baseline_file")
if [ -n "$result_files" ]; then
    # shellcheck disable=SC2206
    args+=(--result_files $result_files)
fi
[ -n "$output_json" ] && args+=(--output_json "$output_json")

echo "=========================================="
echo " URS session-level 满意度预测评估"
echo "  result_file  = ${result_file:-（未指定）}"
echo "  baseline     = ${baseline_file:-（未指定）}"
echo "  result_files = ${result_files:-（未指定）}"
echo "  output_json  = ${output_json:-（不保存）}"
echo "=========================================="

python eval/personalized.py "${args[@]}"

echo "Done."
