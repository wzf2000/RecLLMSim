#!/usr/bin/env bash
# 二分类 SAT/DSAT 评估（含用户感知指标）。
# 用法：
#   bash scripts/eval_binary_sat.sh
#   bash scripts/eval_binary_sat.sh --models sft grpo
#   bash scripts/eval_binary_sat.sh --result_file path/to/results.jsonl --source llm
#   bash scripts/eval_binary_sat.sh --output_json outputs/evaluation/binary_sat_metrics.json

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DETECTION_DIR="$(dirname "$SCRIPT_DIR")"
cd "$DETECTION_DIR"

python eval_binary_sat.py \
  --min_samples 3 \
  --output_json outputs/evaluation/binary_sat_user_aware_metrics.json \
  "$@"
