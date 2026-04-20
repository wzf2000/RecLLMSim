#!/usr/bin/env bash
# ============================================================
# diagnose_confusion.sh — 对已有满意度预测结果跑混淆矩阵诊断
# 从 detection/ 目录运行：bash scripts/diagnose_confusion.sh
# ============================================================
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

# ── 用法 ─────────────────────────────────────────────────────
# 环境变量：
#   result_files — 空格分隔的 name=path 列表（必填）
#   output_md    — 输出 markdown 报告路径（可选）
#
# 示例：
#   result_files="gpt4o_none=outputs/personalized/gpt-4o-mini_test_none_v2.jsonl \
#     qwen3_none=outputs/personalized/Qwen_Qwen3-8B_test_none.jsonl" \
#   output_md=reports/diagnose_confusion.md \
#     bash scripts/diagnose_confusion.sh
# ─────────────────────────────────────────────────────────────

result_files="${result_files:-}"
output_md="${output_md:-}"

if [ -z "$result_files" ]; then
    echo "错误：必须设置 result_files 环境变量（空格分隔的 name=path 列表）" >&2
    exit 1
fi

args=(--result_files $result_files)
[ -n "$output_md" ] && args+=(--output_md "$output_md")

echo "=========================================="
echo "  诊断：混淆矩阵 + 分数分布 + 用户级偏差"
echo "  result_files = $result_files"
echo "  output_md    = ${output_md:-（不保存）}"
echo "=========================================="

python eval/diagnose_confusion.py "${args[@]}"

echo "Done."
