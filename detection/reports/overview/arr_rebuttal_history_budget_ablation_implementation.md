# ARR Rebuttal: History Budget Ablation Implementation

本文档记录针对 Reviewer fUbw W4 的 history amount robustness ablation 代码实现和推荐运行方式。
本次只完成可复用代码与命令设计，尚未实际运行完整 ablation。

## 1. Implementation Summary

新增参数：

- `--history_session_budget`: 每个 user-target block 最多使用的 source-history session 数量。
  `0` 表示使用全部历史，保持原始行为不变。
- `--history_budget_strategy`: history budget 的裁剪策略。
  当前支持 `round_robin_task` 和 `original_order`，默认使用 `round_robin_task`。

主要改动：

- `detection/lib/personalized_data.py`
  - 新增 `apply_history_session_budget(...)`。
  - 默认 `round_robin_task` 会按 source task 轮转选取历史 sessions，避免小 K 时总是来自同一任务类型。
  - `PersonalizedSample` 增加 budget 元信息和 `history_cache_tag`。
- `detection/trace/personalized_memory.py`
  - memory cache key 在 budget run 中追加 `histk{K}_{strategy}`，避免误读 full-history cache。
- `detection/trace/personalized_runner.py`
  - 输出记录增加 `history_session_budget`、`history_budget_strategy`、`n_history_sessions_used`、`n_history_sessions_before_budget`。
- `detection/trace/collect_personalized.py`
  - 增加 CLI 参数并在构建样本后应用 history budget。
  - 自动输出文件名会包含 `histk{K}_{strategy}`。
- `detection/scripts/collect_personalized.sh` 和 `detection/scripts/collect_personalized_vllm.sh`
  - 增加对应环境变量透传。

## 2. Verification

已完成不触发模型调用的本地验证：

```bash
PYTHONPYCACHEPREFIX=/tmp/rec_pycache conda run -n chat python -m py_compile \
  detection/lib/personalized_data.py \
  detection/trace/personalized_memory.py \
  detection/trace/personalized_runner.py \
  detection/trace/collect_personalized.py

bash -n detection/scripts/collect_personalized.sh
bash -n detection/scripts/collect_personalized_vllm.sh
```

并用 `conda run -n chat python -c` 检查了 `K=1,2,4` 的裁剪行为和 cache tag。

## 3. Recommended Run Commands

以下命令建议先在同一个 20-user subset 上跑 `all, K=4, K=2, K=1`。
为保证四个 setting 使用同一批 target turns，建议统一设置 `min_history_sessions=4`。
所有命令都使用独立 `memory_cache_dir`，避免不同 history budget 之间复用 memory cache。

### 3.1 Start vLLM

如果本地已有 Qwen3-8B OpenAI-compatible vLLM 服务，可以跳过此步。
否则在一个单独终端中运行：

```bash
cd detection
conda activate chat
CUDA_VISIBLE_DEVICES=3 \
model=Qwen/Qwen3-8B \
port=8000 \
tensor_parallel=1 \
max_model_len=32768 \
gpu_memory_utilization=0.90 \
bash scripts/serve_vllm.sh
```

如果端口或 GPU 已被占用，将 `CUDA_VISIBLE_DEVICES` 和 `port` 改成可用值，并在后续命令中同步修改 `vllm_base_url`。

### 3.2 Run All-History Baseline on the Same Subset

```bash
cd detection
conda activate chat
model=Qwen/Qwen3-8B \
vllm_base_url=http://localhost:8000/v1 \
memory_update_mode=none \
memory_version=v2 \
turn_eval_prompt_version=v2 \
min_history_sessions=4 \
history_session_budget=0 \
limit_users=20 \
user_offset=0 \
max_workers=4 \
memory_cache_dir=outputs/personalized/memory_cache_history_budget_all_u20 \
output_jsonl=outputs/personalized/qwen3_8b_history_budget_all_u20.jsonl \
bash scripts/collect_personalized_vllm.sh
```

### 3.3 Run K-History Settings

```bash
cd detection
conda activate chat
for k in 1 2 4; do
  model=Qwen/Qwen3-8B \
  vllm_base_url=http://localhost:8000/v1 \
  memory_update_mode=none \
  memory_version=v2 \
  turn_eval_prompt_version=v2 \
  min_history_sessions=4 \
  history_session_budget=${k} \
  history_budget_strategy=round_robin_task \
  limit_users=20 \
  user_offset=0 \
  max_workers=4 \
  memory_cache_dir=outputs/personalized/memory_cache_history_budget_k${k}_u20 \
  output_jsonl=outputs/personalized/qwen3_8b_history_budget_k${k}_u20.jsonl \
  bash scripts/collect_personalized_vllm.sh
done
```

### 3.4 Evaluate the Ablation

```bash
cd detection
conda activate chat
result_files="k1=outputs/personalized/qwen3_8b_history_budget_k1_u20.jsonl \
k2=outputs/personalized/qwen3_8b_history_budget_k2_u20.jsonl \
k4=outputs/personalized/qwen3_8b_history_budget_k4_u20.jsonl \
all=outputs/personalized/qwen3_8b_history_budget_all_u20.jsonl" \
output_json=outputs/personalized/qwen3_8b_history_budget_u20_eval.json \
bash scripts/eval_personalized.sh
```

建议在结果报告中优先摘取 Pearson、QWK、low-side F1，并额外检查每个 JSONL 的记录数是否一致。

## 4. Rebuttal Interpretation Template

如果性能随 K 增大逐步改善，可以回应为：

> We add a history-budget ablation that keeps the target turns fixed and limits memory construction to K source conversations per user-target block.
> The trend shows that additional user history improves personalized satisfaction evaluation, while the evaluator remains usable under sparse histories.

如果 K=1 明显下降，也可以合理解释为：

> This confirms that the method benefits from sufficient user-specific evidence and that sparse-history personalization remains an important limitation.
