#!/usr/bin/env bash
# 对预测结果 JSON 做细粒度分析：按分数/任务/轮次/reason 的准确率、大误差分布、
# reason 混淆矩阵、满意/不满意二分类等。
#
# 用法：
#   result_file=outputs/evaluation/xxx_results.json bash scripts/eval_analysis.sh
#   result_file=outputs/evaluation/xxx_results.json bash scripts/eval_analysis.sh --output outputs/evaluation/xxx_analysis.json
#   result_file=outputs/evaluation/xxx_results.json bash scripts/eval_analysis.sh --no_print --output outputs/evaluation/xxx_analysis.json

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DETECTION_DIR="$(dirname "$SCRIPT_DIR")"
cd "$DETECTION_DIR"
export PYTHONPATH="$DETECTION_DIR${PYTHONPATH:+:$PYTHONPATH}"

if [ -z "${result_file:-}" ]; then
  echo "错误：必须指定输入参数 result_file"
  exit 1
fi

: "${output_json:=outputs/evaluation/$(basename "${result_file%.json}")_analysis.json}"

echo "result_file: ${result_file}"
echo "output_json: ${output_json}"

python eval/analysis.py \
  "${result_file}" \
  --output "${output_json}" \
  "$@"
