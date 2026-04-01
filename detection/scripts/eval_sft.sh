#!/usr/bin/env bash
# 在 detection/ 目录下评估 SFT 模型（sft_from_traces.py）。
# 检查点写入 detection/ckpts/sft_from_traces/。

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DETECTION_DIR="$(dirname "$SCRIPT_DIR")"
echo "DETECTION_DIR: $DETECTION_DIR"
cd "$DETECTION_DIR"

: "${CUDA_VISIBLE_DEVICES:=0}"
export CUDA_VISIBLE_DEVICES

if [ -z "${checkpoint:-}" ]; then
  echo "错误：必须指定输入参数 checkpoint"
  exit 1
fi

# 根据 checkpoint 是否包含 reasoning，决定 think_wrap 和 flags
if [[ "${checkpoint}" == *"reasoning"* || "${checkpoint}" == *"self_distill"* ]]; then
  think_wrap="qwen3"
  flags="--include_reasoning_content"
else
  think_wrap="none"
  flags=""
fi

echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "checkpoint: ./ckpts/${checkpoint}"
echo "metrics_json: ./outputs/evaluation/${checkpoint}_metrics.json"
echo "think_wrap: ${think_wrap}"
echo "flags: ${flags}"

python eval_sft_from_traces.py \
  --checkpoint ./ckpts/${checkpoint} \
  --base_model_name Qwen/Qwen3-8B \
  --max_length 2048 \
  --max_new_tokens 512 \
  --max_history_turns 5 \
  --metrics_json ./outputs/evaluation/${checkpoint}_metrics.json \
  --think_wrap ${think_wrap} \
  ${flags}
