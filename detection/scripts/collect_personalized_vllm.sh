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
#   memory_update_prompt_version — auto / v2 / v2_1 / v2_2 / v2_3 / v2_4 / v2_5 / v3（默认 auto）
#   no_memory            — 1 则跳过记忆（默认 0）
#   max_workers          — 并发线程数（默认 4；本地模型吞吐有限，不宜过高）
#   history_window_size  — turn-level prompt 中保留的最近对话轮数（默认 5）
#   min_history_sessions — 过滤 history session 数量不足的 block（默认 1）
#   history_session_budget — 0 表示全部历史；>0 表示每个 block 最多使用 K 个历史 session
#   history_budget_strategy — round_robin_task / original_order（默认 round_robin_task）
#   limit                — 调试用，限制 block 数量（<=0 不限，默认 0）
#   limit_users          — 调试用，按用户数量限制子集（<=0 不限，默认 0）
#   user_offset          — 按用户子集截取时的起始偏移（默认 0）
#   output_jsonl         — 输出路径（留空自动生成）
#   memory_cache_dir     — 记忆缓存目录（默认 outputs/personalized/memory_cache）
#   memory_model         — 用于构建/更新 memory 的模型；留空则等于 model
#   memory_vllm_base_url — memory_model 的 OpenAI-compatible 地址；
#                          留空时同模型沿用 vllm_base_url，异模型使用默认 API client
#   memory_version       — v2 / v3（默认 v2）
#   turn_eval_prompt_version — v2 / v3 / v3_1 / v3_two_stage / v3_two_stage_v2 / history_prior_delta / history_prior_delta_v2 / history_prior_delta_v3 / history_prior_delta_v3_1 / history_prior_delta_v3_episodic / history_prior_delta_v3_episodic_twopass / qwen_short / boundary_34 / boundary_34_refute / boundary_34_refute_v2 / boundary_34_selective_refute / boundary_34_selective_refute_v2 / boundary_34_selective_refute_v2_fullscale / boundary_34_selective_refute_v3 / boundary_34_selective_refute_v4（默认 v2）

set -euo pipefail
cd "$(dirname "$0")/.."   # 切换到 detection/ 目录

model="${model:-Qwen/Qwen3-8B}"
vllm_base_url="${vllm_base_url:-http://localhost:8000/v1}"
vllm_api_key="${vllm_api_key:-EMPTY}"
memory_vllm_base_url="${memory_vllm_base_url:-}"
memory_vllm_api_key="${memory_vllm_api_key:-EMPTY}"
split="${split:-test}"
memory_update_mode="${memory_update_mode:-none}"
memory_update_prompt_version="${memory_update_prompt_version:-auto}"
no_memory="${no_memory:-0}"
max_workers="${max_workers:-4}"
history_window_size="${history_window_size:-5}"
min_history_sessions="${min_history_sessions:-1}"
history_session_budget="${history_session_budget:-0}"
history_budget_strategy="${history_budget_strategy:-round_robin_task}"
limit="${limit:-0}"
limit_users="${limit_users:-0}"
user_offset="${user_offset:-0}"
output_jsonl="${output_jsonl:-}"
memory_cache_dir="${memory_cache_dir:-outputs/personalized/memory_cache}"
memory_model="${memory_model:-}"
memory_version="${memory_version:-v2}"
n_anchors="${n_anchors:-0}"
turn_eval_prompt_version="${turn_eval_prompt_version:-v2}"

# 构建参数列表
ARGS=(
    --model "${model}"
    --vllm_base_url "${vllm_base_url}"
    --vllm_api_key "${vllm_api_key}"
    --split "${split}"
    --memory_update_mode "${memory_update_mode}"
    --memory_update_prompt_version "${memory_update_prompt_version}"
    --max_workers "${max_workers}"
    --history_window_size "${history_window_size}"
    --memory_cache_dir "${memory_cache_dir}"
    --memory_version "${memory_version}"
    --min_history_sessions "${min_history_sessions}"
    --history_session_budget "${history_session_budget}"
    --history_budget_strategy "${history_budget_strategy}"
    --n_anchors "${n_anchors}"
    --turn_eval_prompt_version "${turn_eval_prompt_version}"
)

if [ -n "${memory_model}" ]; then
    ARGS+=(--memory_model "${memory_model}")
fi
if [ -n "${memory_vllm_base_url}" ]; then
    ARGS+=(--memory_vllm_base_url "${memory_vllm_base_url}" --memory_vllm_api_key "${memory_vllm_api_key}")
fi

if [ "${no_memory}" = "1" ]; then
    ARGS+=(--no_memory)
fi

if [ -n "${output_jsonl}" ]; then
    ARGS+=(--output_jsonl "${output_jsonl}")
fi

if [ "${limit}" -gt 0 ]; then
    ARGS+=(--limit "${limit}")
fi

if [ "${limit_users}" -gt 0 ]; then
    ARGS+=(--limit_users "${limit_users}" --user_offset "${user_offset}")
fi

echo "=========================================="
echo " Model:       ${model}"
echo " Memory model:${memory_model:-$model}"
echo " vLLM URL:    ${vllm_base_url}"
if [ -n "${memory_vllm_base_url}" ]; then
    memory_url_label="${memory_vllm_base_url}"
elif [ "${memory_model:-$model}" = "$model" ]; then
    memory_url_label="${vllm_base_url}"
else
    memory_url_label="default API client"
fi
echo " Memory URL:  ${memory_url_label}"
echo " Split:       ${split}"
echo " Update mode: ${memory_update_mode}"
echo " Update ver:  ${memory_update_prompt_version}"
echo " Memory ver:  ${memory_version}"
echo " No memory:   ${no_memory}"
echo " Min history: ${min_history_sessions} sessions"
echo " Hist budget: ${history_session_budget} sessions"
echo " Hist strat:  ${history_budget_strategy}"
echo " Hist window: ${history_window_size} turns"
echo " n_anchors:   ${n_anchors}"
echo " Prompt ver:  ${turn_eval_prompt_version}"
echo " Limit users: ${limit_users} (offset=${user_offset})"
echo " Workers:     ${max_workers}"
echo "=========================================="

PYTHONPATH="$(pwd)" python trace/collect_personalized.py "${ARGS[@]}"
