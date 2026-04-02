#!/usr/bin/env bash
# 在 detection/ 目录下训练 Qwen3-8B+LoRA 满意度预测器（satisfaction_predictor_lora.py）。
# 检查点写入 detection/ckpts/llm_predictor/。

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DETECTION_DIR="$(dirname "$SCRIPT_DIR")"
echo "DETECTION_DIR: $DETECTION_DIR"
cd "$DETECTION_DIR"
export PYTHONPATH="$DETECTION_DIR${PYTHONPATH:+:$PYTHONPATH}"

mkdir -p ckpts/llm_predictor

# 按需改 GPU；不设则使用 GPU 0
: "${CUDA_VISIBLE_DEVICES:=0}"

export CUDA_VISIBLE_DEVICES

python predictor/lora.py \
  --model_name Qwen/Qwen3-8B \
  --max_len 1024 \
  --batch_size 2 \
  --eval_batch_size 8 \
  --num_epochs 10
