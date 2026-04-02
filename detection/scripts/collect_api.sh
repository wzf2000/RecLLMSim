#!/usr/bin/env bash
# 在 detection/ 目录下收集 API 模型轨迹（collect_api_model_traces.py）。

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DETECTION_DIR="$(dirname "$SCRIPT_DIR")"
echo "DETECTION_DIR: $DETECTION_DIR"
cd "$DETECTION_DIR"
export PYTHONPATH="$DETECTION_DIR${PYTHONPATH:+:$PYTHONPATH}"

: "${model:=gpt-5}"

if [ -z "${output_jsonl:-}" ]; then
    echo "错误：必须指定输入参数 output_jsonl"
    exit 1
fi

# 如果 output_jsonl 包含 reflection，则 do_reflection 为 true 且需要指定 input_jsonl
if [[ "${output_jsonl}" == *"reflection"* ]]; then
    if [ -z "${input_jsonl:-}" ]; then
        echo "错误：必须指定输入参数 input_jsonl"
        exit 1
    fi
    flags="--do_reflection --input_jsonl ./outputs/${input_jsonl}.jsonl --reflection_output_jsonl ./outputs/${output_jsonl}.jsonl"
else
    : "${sample_size:=1000}"
    : "${data_split:=train}"
    flags="--sample_size ${sample_size} --data_split ${data_split} --output_jsonl ./outputs/${output_jsonl}.jsonl"
fi

echo "model: ${model}"
if [[ "${output_jsonl}" == *"reflection"* ]]; then
    echo "do_reflection: true"
    echo "input_jsonl: ./outputs/${input_jsonl}.jsonl"
    echo "reflection_output_jsonl: ./outputs/${output_jsonl}.jsonl"
else
    echo "sample_size: ${sample_size}"
    echo "output_jsonl: ./outputs/${output_jsonl}.jsonl"
    echo "data_split: ${data_split}"
fi

python trace/collect_api.py \
    --model ${model} \
    ${flags} \
    "$@"
