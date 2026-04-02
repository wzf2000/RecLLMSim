#!/usr/bin/env bash
# 用户感知指标评估：per-user aggregation + within-user centering。
# 对比全局指标 vs. Per-user vs. Within-user-centered，量化用户打分偏差的影响。
#
# 用法：
#   bash scripts/eval_user_aware.sh
#   bash scripts/eval_user_aware.sh --models sft grpo
#   bash scripts/eval_user_aware.sh --result_file path/to/results.jsonl
#   bash scripts/eval_user_aware.sh --min_samples 5 --output_json outputs/evaluation/user_aware.json

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DETECTION_DIR="$(dirname "$SCRIPT_DIR")"
cd "$DETECTION_DIR"
export PYTHONPATH="$DETECTION_DIR${PYTHONPATH:+:$PYTHONPATH}"

python eval/user_aware.py \
  --min_samples 3 \
  --output_json outputs/evaluation/user_aware_metrics.json \
  "$@"
