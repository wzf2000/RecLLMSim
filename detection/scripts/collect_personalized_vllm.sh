#!/usr/bin/env bash
# 基于 vLLM 本地模型运行个性化满意度预测（training-free）
#
# 用法（从 detection/ 目录执行）：
#   model=Qwen/Qwen3-8B bash scripts/collect_personalized_vllm.sh
#
# 可选环境变量（均有默认值）：
#   model                — vLLM 中的模型名（对应 --model-name，默认 Qwen/Qwen3-8B）
#   vllm_base_url        — vLLM 服务地址（默认 http://localhost:8000/v1）
#   vllm_api_key         — vLLM API key（默认 EMPTY）
#   split                — test / train / all（默认 test）
#   memory_update_mode   — none / per_session / per_session_oracle（默认 none）
#   no_memory            — 1 则跳过记忆（默认 0）
#   max_workers          — 并发线程数（默认 4；本地模型吞吐有限，不宜过高）
#   limit                — 调试用，限制 block 数量（<=0 不限，默认 0）
#   output_jsonl         — 输出路径（留空自动生成）
#   memory_cache_dir     — 记忆缓存目录（默认 outputs/personalized/memory_cache）
#   turn_eval_prompt_version — v2 / qwen_short（默认 v2）

set -euo pipefail
cd "$(dirname "$0")/.."   # 切换到 detection/ 目录

model="${model:-Qwen/Qwen3-8B}"
vllm_base_url="${vllm_base_url:-http://localhost:8000/v1}"
vllm_api_key="${vllm_api_key:-EMPTY}"
split="${split:-test}"
memory_update_mode="${memory_update_mode:-none}"
no_memory="${no_memory:-0}"
max_workers="${max_workers:-4}"
limit="${limit:-0}"
output_jsonl="${output_jsonl:-}"
memory_cache_dir="${memory_cache_dir:-outputs/personalized/memory_cache}"
n_anchors="${n_anchors:-0}"
turn_eval_prompt_version="${turn_eval_prompt_version:-v2}"

# 构建参数列表
ARGS=(
    --model "${model}"
    --vllm_base_url "${vllm_base_url}"
    --vllm_api_key "${vllm_api_key}"
    --split "${split}"
    --memory_update_mode "${memory_update_mode}"
    --max_workers "${max_workers}"
    --memory_cache_dir "${memory_cache_dir}"
    --n_anchors "${n_anchors}"
    --turn_eval_prompt_version "${turn_eval_prompt_version}"
)

if [ "${no_memory}" = "1" ]; then
    ARGS+=(--no_memory)
fi

if [ -n "${output_jsonl}" ]; then
    ARGS+=(--output_jsonl "${output_jsonl}")
fi

if [ "${limit}" -gt 0 ]; then
    ARGS+=(--limit "${limit}")
fi

echo "=========================================="
echo " Model:       ${model}"
echo " vLLM URL:    ${vllm_base_url}"
echo " Split:       ${split}"
echo " Update mode: ${memory_update_mode}"
echo " No memory:   ${no_memory}"
echo " n_anchors:   ${n_anchors}"
echo " Prompt ver:  ${turn_eval_prompt_version}"
echo " Workers:     ${max_workers}"
echo "=========================================="

PYTHONPATH="$(pwd)" python trace/collect_personalized.py "${ARGS[@]}"
