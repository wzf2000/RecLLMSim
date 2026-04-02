# detection/ 目录结构与脚本用法

所有脚本均需在 `detection/` 目录下运行，或通过 `scripts/*.sh` 调用（脚本自动设置 `PYTHONPATH`）。

---

## 目录结构

```
detection/
├── lib/                    # 公共工具库（被其他模块 import）
│   ├── utils.py                 # 数据路径、通用工具
│   ├── data_split.py            # 按用户分组的数据划分
│   ├── satisfaction_constants.py # reason 标签、分数映射等常量
│   ├── metric_statistics.py     # 数据统计与可视化工具
│   ├── evaluation.py            # 通用评估指标（MAE/RMSE/Pearson/Spearman/Kappa）
│   ├── user_aware_metrics.py    # 用户感知指标（per-user aggregation + within-user centering）
│   ├── llm.py                   # OpenAI client 封装
│   └── qwen_lora_utils.py       # Qwen3+LoRA 加载工具
│
├── predictor/              # 满意度预测器（训练 + 推理）
│   ├── bert.py                  # BERT 回归预测器
│   ├── bert_ordinal.py          # BERT + Ordinal Head 预测器
│   ├── lora.py                  # Qwen3+LoRA 回归预测器
│   └── lora_ordinal.py          # Qwen3+LoRA + Ordinal Head 预测器（当前主力）
│
├── trace/                  # 训练数据收集（轨迹采集 + SFT/GRPO 格式化）
│   ├── collect_api.py           # 调用 GPT 等 API 模型采集推理轨迹
│   ├── collect_self_distill_v1.py  # Self-distill v1：teacher 注入 gold 答案生成轨迹
│   ├── collect_self_distill_v2.py  # Self-distill v2：当前主用版本
│   ├── sft.py                   # SFT 数据格式化 + 训练
│   └── grpo.py                  # GRPO 训练
│
├── eval/                   # 评估脚本
│   ├── sft.py                   # 评估 SFT 模型
│   ├── grpo.py                  # 评估 GRPO 模型（vLLM + LoRA stacking）
│   ├── binary_sat.py            # 二分类 SAT/DSAT 评估（≤3=DSAT, ≥4=SAT）
│   ├── spur.py                  # SPUR 满意度估计（Lin et al., ACL 2024）
│   ├── user_aware.py            # 用户感知指标对比（Global vs. PerUser vs. Centered）
│   └── analysis.py              # 细粒度分析（按分数/任务/轮次/reason）
│
├── scripts/                # 可执行 Shell 脚本（统一入口，自动设置 PYTHONPATH）
│   ├── runs.sh                  # 记录已运行过的完整命令（实验日志）
│   ├── collect_api.sh           # 采集 API 模型轨迹
│   ├── collect_self_distill.sh  # 采集 Self-distill 轨迹（v1/v2）
│   ├── sft.sh                   # SFT 训练
│   ├── grpo.sh                  # GRPO 训练
│   ├── eval_sft.sh              # 评估 SFT 模型
│   ├── eval_grpo.sh             # 评估 GRPO 模型
│   ├── eval_binary_sat.sh       # 二分类 SAT/DSAT 评估
│   ├── eval_user_aware.sh       # 用户感知指标评估
│   ├── eval_analysis.sh         # 细粒度结果分析
│   ├── run_spur.sh              # 运行 SPUR 方法
│   ├── train_satisfaction_bert.sh         # 训练 BERT 预测器
│   ├── train_satisfaction_ordinal.sh      # 训练 BERT+Ordinal 预测器
│   ├── train_satisfaction_lora.sh         # 训练 Qwen3+LoRA 预测器
│   ├── train_satisfaction_ordinal_lora.sh # 训练 Qwen3+LoRA+Ordinal 预测器（内部数据）
│   └── train_ordinal_lora_uss.sh          # 训练 Qwen3+LoRA+Ordinal 预测器（USS 数据）
│
├── tools/                  # 一次性工具脚本
│   ├── annotation_app.py        # 满意度标注 Web 应用
│   └── migrate_trace_jsonl_format.py  # 旧格式 JSONL 迁移工具
│
├── legacy/                 # 历史实验代码（不再维护）
│   ├── utt_detection.py         # 话语级不满意检测
│   ├── dataset.py               # 旧版数据集类
│   ├── ml.py                    # 传统 ML 分类器
│   ├── merge_data.py            # 数据合并脚本
│   ├── prompts.py               # 旧版 prompt 模板
│   ├── reason_data.py           # reason 数据处理
│   └── satisfaction_data.py     # 旧版满意度数据处理
│
├── ckpts/                  # 模型检查点（git ignored）
├── outputs/                # 评估结果、轨迹输出（git ignored）
└── STRUCTURE.md            # 本文件
```

---

## 脚本用法

> 所有脚本均从 `detection/` 目录调用，支持通过环境变量传参。

### 数据采集

#### `scripts/collect_api.sh` — 调用 API 模型采集推理轨迹

```bash
# 采集 GPT-5 推理轨迹（默认 1000 条，train split）
model=gpt-5 sample_size=1000 output_jsonl=gpt5_traces ./scripts/collect_api.sh

# 采集反思轨迹（需先有原始轨迹）
output_jsonl=gpt5_traces_reflection input_jsonl=gpt5_traces ./scripts/collect_api.sh
```

| 环境变量 | 默认值 | 说明 |
|---|---|---|
| `model` | `gpt-5` | API 模型名 |
| `output_jsonl` | 必填 | 输出文件名（自动加 `outputs/` 前缀和 `.jsonl` 后缀） |
| `sample_size` | `1000` | 采样数量（非 reflection 模式） |
| `data_split` | `train` | 数据划分（train/test） |
| `input_jsonl` | reflection 模式必填 | 原始轨迹文件名 |

#### `scripts/collect_self_distill.sh` — Self-distill 轨迹采集

```bash
# v2（默认）：当前主用版本
CUDA_VISIBLE_DEVICES=0 distill_version=v2 \
  sft_checkpoint=sft_qwen3_from_gpt5_correct_reasoning \
  output_jsonl=self_distill_v3 \
  num_samples_per_prompt=4 \
  min_reasoning_tokens=10 \
  ./scripts/collect_self_distill.sh

# v1：teacher 注入 gold 答案
CUDA_VISIBLE_DEVICES=0 distill_version=v1 \
  sft_checkpoint=sft_qwen3_from_gpt5_correct_reasoning \
  output_jsonl=self_distill_v2 \
  num_samples_per_prompt=32 \
  min_reasoning_tokens=20 \
  ./scripts/collect_self_distill.sh
```

| 环境变量 | 默认值 | 说明 |
|---|---|---|
| `distill_version` | `v2` | `v1` 或 `v2` |
| `sft_checkpoint` | 必填 | 检查点目录名（`ckpts/` 下） |
| `output_jsonl` | 必填 | 输出文件名 |
| `num_samples_per_prompt` | 必填 | 每条 prompt 采样数 |
| `min_reasoning_tokens` | 必填 | 最短推理 token 数 |
| `data_split` | `train` | 数据划分 |
| `temperature` | `0.7` | 采样温度 |
| `limit` | 不限 | 最大处理条数 |

---

### 训练

#### `scripts/sft.sh` — SFT 训练

```bash
CUDA_VISIBLE_DEVICES=0 batch_size=2 \
  input_jsonl=gpt5_traces_v2.jsonl \
  output_dir=sft_qwen3_from_gpt5 \
  ./scripts/sft.sh
```

- `output_dir` 名含 `reasoning` 或 `self_distill` → 自动启用 `--include_reasoning_content`
- `output_dir` 名含 `reflection` → `trace_source=correct_plus_reflected_wrong`
- 固定 `batch_size * grad_accum = 16`

#### `scripts/grpo.sh` — GRPO 训练

```bash
CUDA_VISIBLE_DEVICES=0,1 batch_size=4 \
  sft_checkpoint=sft_qwen3_from_gpt5_correct \
  ./scripts/grpo.sh
```

多卡时自动使用 `accelerate launch`。

#### `scripts/train_satisfaction_bert.sh` — BERT 预测器

```bash
CUDA_VISIBLE_DEVICES=0 ./scripts/train_satisfaction_bert.sh
```

检查点输出至 `ckpts/best.pt`。

#### `scripts/train_satisfaction_ordinal.sh` — BERT+Ordinal 预测器

```bash
CUDA_VISIBLE_DEVICES=0 ./scripts/train_satisfaction_ordinal.sh
```

检查点输出至 `ckpts/ordinal/best.pt`。

#### `scripts/train_satisfaction_lora.sh` — Qwen3+LoRA 预测器

```bash
CUDA_VISIBLE_DEVICES=0 ./scripts/train_satisfaction_lora.sh
```

检查点输出至 `ckpts/llm_predictor/`。

#### `scripts/train_satisfaction_ordinal_lora.sh` — Qwen3+LoRA+Ordinal 预测器（内部数据）

```bash
CUDA_VISIBLE_DEVICES=0 ./scripts/train_satisfaction_ordinal_lora.sh
```

#### `scripts/train_ordinal_lora_uss.sh` — Qwen3+LoRA+Ordinal 预测器（USS 数据）

需先运行 `python tools/preprocess_uss.py` 完成数据预处理。

```bash
# 全部 5 个子数据集训练
CUDA_VISIBLE_DEVICES=0 bash scripts/train_ordinal_lora_uss.sh

# 仅用中文子集 JDDC
CUDA_VISIBLE_DEVICES=0 uss_datasets="JDDC" bash scripts/train_ordinal_lora_uss.sh

# 仅用英文子集
CUDA_VISIBLE_DEVICES=0 uss_datasets="SGD MWOZ ReDial CCPE" bash scripts/train_ordinal_lora_uss.sh

# 测试已有 checkpoint
CUDA_VISIBLE_DEVICES=0 test_only=1 checkpoint=checkpoint-500 bash scripts/train_ordinal_lora_uss.sh
```

检查点输出至 `ckpts/llm_predictor_ordinal/`。损失系数：
- `--alpha 1.0`：ordinal satisfaction 损失
- `--beta 2.0`：reason 分类损失
- `--gamma 0.1`：单调性惩罚
- `--delta 0.2`：跨任务一致性约束
- `--use_score_weights`：低分样本逆频率权重
- `--use_reason_weights`：reason 类别逆频率权重

---

### 评估

#### `scripts/eval_sft.sh` — 评估 SFT 模型

```bash
CUDA_VISIBLE_DEVICES=0 checkpoint=sft_qwen3_from_gpt5_correct ./scripts/eval_sft.sh
```

结果输出至 `outputs/evaluation/<checkpoint>_results.jsonl` 和 `_metrics.json`。

#### `scripts/eval_grpo.sh` — 评估 GRPO 模型

```bash
CUDA_VISIBLE_DEVICES=0 sft_checkpoint=sft_qwen3_from_gpt5_correct ./scripts/eval_grpo.sh
```

#### `scripts/eval_binary_sat.sh` — 二分类 SAT/DSAT 评估

```bash
# 评估所有预设模型（sft / grpo / ordinal）
bash scripts/eval_binary_sat.sh

# 仅评估指定模型
bash scripts/eval_binary_sat.sh --models sft grpo

# 评估自定义结果文件
bash scripts/eval_binary_sat.sh --result_file path/to/results.jsonl --source llm
```

指标：Accuracy / F1-macro / F1-SAT / F1-DSAT / Precision / Recall / Kappa / AUC。

#### `scripts/eval_user_aware.sh` — 用户感知指标评估

```bash
# 评估所有预设模型
bash scripts/eval_user_aware.sh

# 自定义最小样本数
bash scripts/eval_user_aware.sh --min_samples 5

# 仅评估部分模型 + 保存结果
bash scripts/eval_user_aware.sh --models sft ordinal --output_json outputs/evaluation/ua.json
```

输出三列对比：Global / PerUser（Fisher z-transform 加权）/ Centered（within-user 去均值）。

#### `scripts/eval_analysis.sh` — 细粒度结果分析

```bash
result_file=outputs/evaluation/sft_results.json bash scripts/eval_analysis.sh

# 指定输出路径
result_file=outputs/evaluation/sft_results.json \
  output_json=outputs/evaluation/sft_analysis.json \
  bash scripts/eval_analysis.sh
```

分析维度：按分数(1-5)、按任务、按轮次、reason 混淆矩阵、大误差样本分布、二分类准确率等。

#### `scripts/run_spur.sh` — SPUR 方法（Lin et al., ACL 2024）

```bash
# 完整运行（Phase 1-3：LLM 提取 rubric + 评分）
bash scripts/run_spur.sh

# 启用 ada-002 embedding 分类器（Phase 4）
bash scripts/run_spur.sh --use_embeddings

# 跳过 rubric 提取，直接评估（需已有缓存）
bash scripts/run_spur.sh --skip_phase1 --skip_phase2

# 仅计算指标（跳过所有 LLM 调用）
bash scripts/run_spur.sh --only_eval
```

结果输出至 `outputs/spur/`，`--use_embeddings` 时额外输出三种分类器变体（rubric_only / embedding_only / combined）。

---

## 导入规范

所有模块使用绝对包路径，需在 `detection/` 目录下运行或由 shell 脚本设置 `PYTHONPATH`：

```python
from lib.utils import ...
from lib.satisfaction_constants import get_reason_to_id
from lib.user_aware_metrics import compute_user_aware_metrics
```

包内相对导入（如 `trace/` 内部）：

```python
from .sft import QWEN3_THINK_BEGIN
from .collect_api import get_rows_from_split
```
