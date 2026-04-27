#!/usr/bin/env bash
# ============================================================
# eval_uss.sh — USS 满意度预测评估（复用 eval/personalized.py）
# 从 detection/ 目录运行：bash scripts/eval_uss.sh
# ============================================================
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

# ── 使用示例 ──────────────────────────────────────────────────
#
# 1. 单文件评估
#    result_file=outputs/uss/gpt-4o_warmup_all5_test_per_session.jsonl \
#      bash scripts/eval_uss.sh
#
# 2. R1 vs R2 + no_memory baseline 对比
#    result_files="no_mem=outputs/uss/gpt-4o_warmup_all5_test_no_memory.jsonl \
#                  r1=outputs/uss/gpt-4o_warmup_all5_test_none.jsonl \
#                  r1_cdf=outputs/uss/gpt-4o_warmup_all5_test_none_calCDF.jsonl \
#                  r2=outputs/uss/gpt-4o_population_all5_test_none.jsonl" \
#    output_json=outputs/uss/comparison.json \
#      bash scripts/eval_uss.sh
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
echo " USS 满意度预测评估"
echo "  result_file  = ${result_file:-（未指定）}"
echo "  baseline     = ${baseline_file:-（未指定）}"
echo "  result_files = ${result_files:-（未指定）}"
echo "  output_json  = ${output_json:-（不保存）}"
echo "=========================================="

python eval/personalized.py "${args[@]}"

echo "Done."
