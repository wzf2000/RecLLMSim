#!/usr/bin/env bash
# 在 detection/ 目录下训练 bert-base-chinese 满意度预测器（satisfaction_predictor.py）。
# 检查点写入 detection/ckpts/best.pt（由 Python 代码固定路径）。

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DETECTION_DIR="$(dirname "$SCRIPT_DIR")"
echo "DETECTION_DIR: $DETECTION_DIR"
cd "$DETECTION_DIR"
export PYTHONPATH="$DETECTION_DIR${PYTHONPATH:+:$PYTHONPATH}"

mkdir -p ckpts

# 按需改 GPU；不设则使用所有可见 GPU
: "${CUDA_VISIBLE_DEVICES:=0}"

export CUDA_VISIBLE_DEVICES

python predictor/bert.py "$@"
