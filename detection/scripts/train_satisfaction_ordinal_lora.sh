#!/usr/bin/env bash
# 在 detection/ 目录下训练 Qwen3-8B+LoRA+Ordinal Head 满意度预测器（satisfaction_predictor_ordinal_lora.py）。
# 检查点写入 detection/ckpts/llm_predictor_ordinal/。
#
# 损失系数说明：
#   --alpha              ordinal satisfaction 损失系数（默认 1.0）
#   --beta               reason classification 损失系数（默认 2.0）
#   --gamma              单调性惩罚系数（默认 0.1）
#   --delta              跨任务一致性约束系数（0 表示禁用，默认 0.2）
#   --use_score_weights  启用分数逆频率权重，缓解低分样本学习不足
#   --use_reason_weights 启用 reason 类别逆频率权重，缓解满意类过多的不平衡问题

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DETECTION_DIR="$(dirname "$SCRIPT_DIR")"
echo "DETECTION_DIR: $DETECTION_DIR"
cd "$DETECTION_DIR"

mkdir -p ckpts/llm_predictor_ordinal

# 按需改 GPU；不设则使用 GPU 0
: "${CUDA_VISIBLE_DEVICES:=0}"

export CUDA_VISIBLE_DEVICES

python satisfaction_predictor_ordinal_lora.py \
  --model_name Qwen/Qwen3-8B \
  --max_len 1024 \
  --batch_size 2 \
  --num_epochs 10 \
  --output_dir ./ckpts/llm_predictor_ordinal \
  --alpha 1.0 \
  --beta 2.0 \
  --gamma 0.1 \
  --delta 0.2 \
  --consistency_temp 2.0 \
  --consistency_center 3.5 \
  --use_score_weights \
  --use_reason_weights
