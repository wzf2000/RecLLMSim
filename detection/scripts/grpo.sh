#!/usr/bin/env bash
# 在 detection/ 目录下训练 GRPO 模型（grpo_from_sft.py）。

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DETECTION_DIR="$(dirname "$SCRIPT_DIR")"
echo "DETECTION_DIR: $DETECTION_DIR"
cd "$DETECTION_DIR"
export PYTHONPATH="$DETECTION_DIR${PYTHONPATH:+:$PYTHONPATH}"

: "${CUDA_VISIBLE_DEVICES:=0,1}"
export CUDA_VISIBLE_DEVICES

# 根据 gpu 数量决定 num_processes
num_gpus=$(echo "${CUDA_VISIBLE_DEVICES}" | tr ',' '\n' | wc -l)

if [ -z "${sft_checkpoint:-}" ]; then
  echo "错误：必须指定输入参数 sft_checkpoint"
  exit 1
fi

: "${num_generations:=8}"
: "${batch_size:=4}"
: "${gradient_accumulation_steps:=4}"

# 确保 batch_size 能被 num_gpus 整除
if [ $((batch_size % num_gpus)) -ne 0 ]; then
  echo "错误：batch_size 不能被 num_gpus 整除"
  exit 1
fi

per_device_train_batch_size=$((batch_size / num_gpus))

# 确保 batch_size * gradient_accumulation_steps 能被 num_generations 整除
if [ $((batch_size * gradient_accumulation_steps % num_generations)) -ne 0 ]; then
  echo "错误：batch_size * gradient_accumulation_steps 不能被 num_generations 整除"
  exit 1
fi

echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "num_processes: ${num_gpus}"
echo "sft_checkpoint: ./ckpts/${sft_checkpoint}"
echo "output_dir: ./ckpts/grpo_from_${sft_checkpoint}"
echo "num_generations: ${num_generations}"
echo "per_device_train_batch_size: ${per_device_train_batch_size}"
echo "gradient_accumulation_steps: ${gradient_accumulation_steps}"

if [ "${num_gpus}" -eq 1 ]; then
  python trace/grpo.py \
    --sft_checkpoint ./ckpts/${sft_checkpoint} \
    --base_model_name Qwen/Qwen3-8B \
    --output_dir ./ckpts/grpo_from_${sft_checkpoint} \
    --data_split train \
    --num_generations ${num_generations} \
    --max_completion_length 512 \
    --per_device_train_batch_size ${batch_size} \
    --gradient_accumulation_steps ${gradient_accumulation_steps} \
    --num_train_epochs 1 \
    --learning_rate 1e-6 \
    --lora_r 8 \
    --lora_alpha 16 \
    --reward_weights 0.2 0.5 0.3
else
  accelerate launch \
    --num_processes ${num_gpus} \
    grpo_from_sft.py \
    --sft_checkpoint ./ckpts/${sft_checkpoint} \
    --base_model_name Qwen/Qwen3-8B \
    --output_dir ./ckpts/grpo_from_${sft_checkpoint} \
    --data_split train \
    --num_generations ${num_generations} \
    --max_completion_length 512 \
    --per_device_train_batch_size ${per_device_train_batch_size} \
    --gradient_accumulation_steps ${gradient_accumulation_steps} \
    --num_train_epochs 1 \
    --learning_rate 1e-6 \
    --lora_r 8 \
    --lora_alpha 16 \
    --reward_weights 0.2 0.5 0.3
fi
