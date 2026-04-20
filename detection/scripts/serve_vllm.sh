#!/usr/bin/env bash
# 启动 vLLM OpenAI-compatible server（兼容 vLLM 0.18.x）
#
# 用法（从 detection/ 或任意目录）：
#   model=Qwen/Qwen3-8B bash scripts/serve_vllm.sh
#   model=Qwen/Qwen3-8B port=8001 tensor_parallel=2 bash scripts/serve_vllm.sh

set -euo pipefail

model="${model:-Qwen/Qwen3-8B}"
port="${port:-8000}"
tensor_parallel="${tensor_parallel:-1}"
gpu_memory_utilization="${gpu_memory_utilization:-0.90}"
max_model_len="${max_model_len:-16384}"

# 结构化输出后端（xgrammar / outlines / guidance / auto）
# vLLM 0.18.x 通过 --structured-outputs-config 传入 JSON 字符串
structured_outputs_backend="${structured_outputs_backend:-xgrammar}"

# Qwen3 thinking mode: 0=关闭（推荐，节省 token），1=开启
enable_thinking="${enable_thinking:-0}"

echo "=========================================="
echo " vLLM serve: ${model}"
echo " Port:       ${port}"
echo " TP:         ${tensor_parallel}"
echo " Max len:    ${max_model_len}"
echo " SO backend: ${structured_outputs_backend}"
echo " Thinking:   ${enable_thinking}"
echo "=========================================="

EXTRA_ARGS=()
if [ "${enable_thinking}" = "0" ]; then
    # 关闭 Qwen3 思考链，直接输出结构化 JSON（标注任务推荐）
    EXTRA_ARGS+=(--chat-template-content-format string)
fi

python -m vllm.entrypoints.openai.api_server \
    --model "${model}" \
    --port "${port}" \
    --tensor-parallel-size "${tensor_parallel}" \
    --gpu-memory-utilization "${gpu_memory_utilization}" \
    --max-model-len "${max_model_len}" \
    --structured-outputs-config "{\"backend\": \"${structured_outputs_backend}\"}" \
    --trust-remote-code \
    "${EXTRA_ARGS[@]}"
