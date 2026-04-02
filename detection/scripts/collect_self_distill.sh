#!/usr/bin/env bash
# 在 detection/ 目录下收集 self-distill 轨迹（trace/collect_self_distill_v1.py 或 v2.py）。

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DETECTION_DIR="$(dirname "$SCRIPT_DIR")"
echo "DETECTION_DIR: $DETECTION_DIR"
cd "$DETECTION_DIR"
export PYTHONPATH="$DETECTION_DIR${PYTHONPATH:+:$PYTHONPATH}"

: "${CUDA_VISIBLE_DEVICES:=0}"
export CUDA_VISIBLE_DEVICES

: "${distill_version:=v2}"

# 根据 distill_version 决定 trace/collect_self_distill.py 或 trace/collect_self_distill_v1.py
if [ "${distill_version}" == "v2" ]; then
  collect_script="trace/collect_self_distill_v2.py"
else
  collect_script="trace/collect_self_distill_v1.py"
fi

if [ -z "${sft_checkpoint:-}" ]; then
  echo "错误：必须指定输入参数 sft_checkpoint"
  exit 1
fi

if [ -z "${output_jsonl:-}" ]; then
  echo "错误：必须指定输入参数 output_jsonl"
  exit 1
fi

if [ -z "${num_samples_per_prompt:-}" ]; then
  echo "错误：必须指定输入参数 num_samples_per_prompt"
  exit 1
fi

if [ -z "${min_reasoning_tokens:-}" ]; then
  echo "错误：必须指定输入参数 min_reasoning_tokens"
  exit 1
fi

: "${data_split:=train}"
: "${temperature:=0.7}"

# 如果存在 limit 参数，则添加到命令中
if [ -n "${limit:-}" ]; then
  limit="--limit ${limit}"
else
  limit=""
fi

echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "collect_script: ${collect_script}"
echo "sft_checkpoint: ./ckpts/${sft_checkpoint}"
echo "output_jsonl: ./outputs/${output_jsonl}.jsonl"
echo "data_split: ${data_split}"
echo "num_samples_per_prompt: ${num_samples_per_prompt}"
echo "min_reasoning_tokens: ${min_reasoning_tokens}"
echo "temperature: ${temperature}"
echo "limit: ${limit}"

python ${collect_script} \
  --sft_checkpoint ./ckpts/${sft_checkpoint} \
  --base_model_name Qwen/Qwen3-8B \
  --output_jsonl ./outputs/${output_jsonl}.jsonl \
  --data_split ${data_split} \
  --num_samples_per_prompt ${num_samples_per_prompt} \
  --min_reasoning_tokens ${min_reasoning_tokens} \
  --temperature ${temperature} \
  ${limit}
