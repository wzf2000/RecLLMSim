#!/usr/bin/env bash
# 在 detection/ 目录下训练 SFT 模型（sft_from_traces.py）。
# 检查点写入 detection/ckpts/sft_from_traces/。

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DETECTION_DIR="$(dirname "$SCRIPT_DIR")"
echo "DETECTION_DIR: $DETECTION_DIR"
cd "$DETECTION_DIR"
export PYTHONPATH="$DETECTION_DIR${PYTHONPATH:+:$PYTHONPATH}"

: "${CUDA_VISIBLE_DEVICES:=0}"
export CUDA_VISIBLE_DEVICES

if [ -z "${input_jsonl:-}" ]; then
  echo "错误：必须指定输入参数 input_jsonl"
  exit 1
fi

if [ -z "${output_dir:-}" ]; then
  echo "错误：必须指定输入参数 output_dir"
  exit 1
fi

# 固定 batch_size * grad_accum = 16，根据 batch_size 计算 grad_accum
: "${batch_size:=1}"
grad_accum=$((16 / batch_size))
# 确认 grad_accum 是整数
if [ $((grad_accum * batch_size)) -ne 16 ]; then
  echo "错误：batch_size * grad_accum 必须等于 16"
  exit 1
fi

# 根据 output_dir 是否包含 reflection，来决定 trace_source
if [[ "${output_dir}" == *"reflection"* ]]; then
  trace_source="correct_plus_reflected_wrong"
else
  trace_source="correct_only"
fi

# 根据 output_dir 是否包含 reasoning 或 self_distill，决定 think_wrap 和 flags
if [[ "${output_dir}" == *"reasoning"* || "${output_dir}" == *"self_distill"* ]]; then
  think_wrap="qwen3"
  flags="--include_reasoning_content"
else
  think_wrap="none"
  flags=""
fi

echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "input_jsonl: ./outputs/${input_jsonl}"
echo "output_dir: ./ckpts/${output_dir}"
echo "batch_size: ${batch_size}"
echo "grad_accum: ${grad_accum}"
echo "trace_source: ${trace_source}"
echo "think_wrap: ${think_wrap}"
echo "flags: ${flags}"

python trace/sft.py \
  --input_jsonl ./outputs/${input_jsonl} \
  --output_dir ./ckpts/${output_dir} \
  --batch_size ${batch_size} \
  --grad_accum ${grad_accum} \
  --trace_source ${trace_source} \
  --think_wrap ${think_wrap} \
  ${flags} \
  "$@"
