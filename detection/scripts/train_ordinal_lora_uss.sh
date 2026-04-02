#!/usr/bin/env bash
# 在 USS 公开数据集上训练/测试 Qwen3-8B+LoRA+Ordinal Head 满意度预测器。
# 数据来源：USS (Sun et al., SIGIR 2021)，需先运行 tools/preprocess_uss.py 完成预处理。
#
# 用法：
#   # 训练（默认全部 5 个子数据集）
#   bash scripts/train_ordinal_lora_uss.sh
#
#   # 仅使用中文子集 JDDC 训练
#   uss_datasets="JDDC" bash scripts/train_ordinal_lora_uss.sh
#
#   # 仅使用英文子集训练
#   uss_datasets="SGD MWOZ ReDial CCPE" bash scripts/train_ordinal_lora_uss.sh
#
#   # 测试已有 checkpoint
#   test_only=1 checkpoint=checkpoint-500 bash scripts/train_ordinal_lora_uss.sh
#
# 可选环境变量：
#   uss_datasets      子数据集列表（空格分隔），默认全部
#   uss_data_dir      USS 预处理 JSONL 目录，默认 ./data/uss/processed
#   output_dir        检查点输出目录名（ckpts/ 下），默认 ordinal_lora_uss
#   test_only         设为 1 时仅测试，需同时指定 checkpoint
#   checkpoint        test_only 模式下的 checkpoint 子目录名

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DETECTION_DIR="$(dirname "$SCRIPT_DIR")"
echo "DETECTION_DIR: $DETECTION_DIR"
cd "$DETECTION_DIR"
export PYTHONPATH="$DETECTION_DIR${PYTHONPATH:+:$PYTHONPATH}"

: "${CUDA_VISIBLE_DEVICES:=0}"
export CUDA_VISIBLE_DEVICES

: "${output_dir:=ordinal_lora_uss}"
: "${uss_data_dir:=./data/uss/processed}"

# 检查预处理数据是否存在
if [ ! -f "${uss_data_dir}/JDDC.jsonl" ] && [ ! -f "${uss_data_dir}/SGD.jsonl" ]; then
  echo "错误：未找到 USS 预处理数据，请先运行："
  echo "  python tools/preprocess_uss.py"
  exit 1
fi

mkdir -p "ckpts/${output_dir}"

# 构造 uss_datasets 参数
if [ -n "${uss_datasets:-}" ]; then
  datasets_flag="--uss_datasets ${uss_datasets}"
else
  datasets_flag=""
fi

echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "output_dir: ./ckpts/${output_dir}"
echo "uss_data_dir: ${uss_data_dir}"
echo "uss_datasets: ${uss_datasets:-all}"

if [ "${test_only:-0}" = "1" ]; then
  if [ -z "${checkpoint:-}" ]; then
    echo "错误：test_only=1 时必须指定 checkpoint"
    exit 1
  fi
  echo "checkpoint: ./ckpts/${output_dir}/${checkpoint}"
  python predictor/lora_ordinal.py \
    --data_source uss \
    --uss_data_dir "${uss_data_dir}" \
    ${datasets_flag} \
    --disable_reason \
    --test_only \
    --checkpoint_path "./ckpts/${output_dir}/${checkpoint}" \
    --use_score_weights \
    "$@"
else
  python predictor/lora_ordinal.py \
    --data_source uss \
    --uss_data_dir "${uss_data_dir}" \
    ${datasets_flag} \
    --output_dir "./ckpts/${output_dir}" \
    --disable_reason \
    --use_score_weights \
    "$@"
fi
