#!/usr/bin/env bash
# 在 detection/ 目录下评估 GRPO 模型（grpo_from_sft.py）。
# 检查点写入 detection/ckpts/grpo_from_sft/。

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DETECTION_DIR="$(dirname "$SCRIPT_DIR")"
echo "DETECTION_DIR: $DETECTION_DIR"
cd "$DETECTION_DIR"
export PYTHONPATH="$DETECTION_DIR${PYTHONPATH:+:$PYTHONPATH}"

mkdir -p ckpts/grpo_from_sft

: "${CUDA_VISIBLE_DEVICES:=0}"
export CUDA_VISIBLE_DEVICES

# 指定输入参数 sft_checkpoint，要求必须给出
if [ -z "${sft_checkpoint:-}" ]; then
  echo "错误：必须指定输入参数 sft_checkpoint"
  exit 1
fi

echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "grpo_checkpoint: ./ckpts/grpo_from_${sft_checkpoint}"
echo "sft_checkpoint: ./ckpts/${sft_checkpoint}"
echo "metrics_json: ./outputs/evaluation/grpo_from_${sft_checkpoint}_metrics.json"
echo "output_jsonl: ./outputs/evaluation/grpo_from_${sft_checkpoint}_results.jsonl"

python eval/grpo.py \
  --grpo_checkpoint ./ckpts/grpo_from_${sft_checkpoint} \
  --sft_checkpoint  ./ckpts/${sft_checkpoint} \
  --base_model_name Qwen/Qwen3-8B \
  --metrics_json ./outputs/evaluation/grpo_from_${sft_checkpoint}_metrics.json \
  --output_jsonl ./outputs/evaluation/grpo_from_${sft_checkpoint}_results.jsonl \
  "$@"
