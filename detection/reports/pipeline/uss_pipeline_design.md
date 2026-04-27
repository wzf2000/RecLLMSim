# USS Cross-Dataset 满意度预测 — Trace & Eval Pipeline 设计

> 把现有 RecLLMSim memory-agent (training-free) 流水线迁移到 USS（5 个公开任务对话子集）。
> USS 缺少持续 user identity，无法直接 cross-task split；本 pipeline 通过把 dialogue（R1）
> 或 subset（R2）当作"伪用户"，最大化复用 `trace/collect_personalized.py::run_agent_on_sample`
> 与现有 eval/calibrate 链路。
>
> 与 RecLLMSim / URS pipeline 的关系：
> - RecLLMSim 是真用户跨任务的标准案例（user × task × turn）
> - URS 是真用户但 session-level 标签（详见 `urs_pipeline_design.md`）
> - **USS 完全没有跨 dialogue 的 user 概念**，只能在 dialogue 内部或 subset 整体层面"伪个性化"

## 1. 数据特性回顾

`detection/data/uss/processed/` 下 5 个 jsonl 子集，均为 turn-level 记录：

| Subset | dialogues | turns | 平均轮数 | 语言 | test 子集 | dialogues ≥5 turns |
|---|---:|---:|---:|---|---:|---:|
| JDDC  | 3300 | 30241 | 9.2  | zh | 330 | 90.0% |
| SGD   | 1000 | 11833 | 11.8 | en | 100 | 98.9% |
| MWOZ  | 1000 | 10553 | 10.6 | en | 100 | 99.3% |
| ReDial| 1000 | 6792  | 6.8  | en | 100 | 87.6% |
| CCPE  |  500 | 5180  | 10.4 | en |  50 | 99.0% |

**关键观察 — 分数分布严重塌缩到 3 分**（来自全量数据，按 subset）：

| Subset | score 1 | 2 | **3** | 4 | 5 |
|---|---:|---:|---:|---:|---:|
| CCPE   | 0.1% | 6.2%  | **86.5%** | 7.2% | 0.0% |
| SGD    | 0.0% | 5.2%  | **80.1%** | 14.0% | 0.6% |
| MWOZ   | 0.1% | 5.3%  | **87.8%** | 6.7% | 0.1% |
| ReDial | 0.1% | 5.2%  | **80.1%** | 14.1% | 0.5% |
| JDDC   | 0.1% | 7.6%  | **82.1%** | 9.3% | 0.9% |

**这对 CDF 校准是结构性挑战**：每条 dialogue 的 warm-up 几乎注定全是 3 分，CDF 退化为常数映射，
无法提供 RecLLMSim/URS 上观测到的"用户评分基准重定标"信号。但 pipeline 仍可作为 ablation 跑通——
关键是验证 memory v2 rubric 在领域级（R2）是否仍带来 turn-level QWK / Pearson 提升。

## 2. 范围 / 不在范围内

**本 pipeline 覆盖**：
- R1 dialogue warm-up：每 dialogue 前 n_warm 个 assistant turn 当历史，剩余 turn 当目标
- R2 subset population rubric：每 subset 抽 K 条 train dialogue 当历史，所有 test dialogue 共享一份 memory
- 输出 JSONL schema 与 `eval/personalized.py` / `eval/calibrate.py` / `eval/diagnose_confusion.py`
  100% 兼容
- 复用 `collect_personalized.py::run_agent_on_sample` 的所有 prompt 版本（v2 / qwen_short / boundary_34_*）
  与 anchor retrieval、selective refute 等高级特性

**不在本迭代范围**：
- Dialogue-level overall score（`*_overall.jsonl`）评估——属于纯分类，与 turn-level memory rubric 不同任务
- Cross-subset 迁移（在 SGD memory 上跑 ReDial）——当前 cache key = `{subset}_population` 已隔离，但跨 subset 泛化是另一研究问题
- 与 `predictor/lora_ordinal.py` 监督基线的端到端对比（应在跑通后单独写报告）

## 3. 数据流总览

```
┌──────────────────────────────────────────────────────┐
│ data/uss/processed/{CCPE,SGD,MWOZ,ReDial,JDDC}.jsonl │
│   (turn-level 记录：dialogue_id / turn_idx / history │
│    / assistant_reply / gold_score / split / ...)     │
└────────────────────┬─────────────────────────────────┘
                     ▼
   ┌─────────────────────────────────────────────────────────┐
   │ lib/uss_pipeline.py                                     │
   │                                                         │
   │  R1: build_uss_warmup_samples()                         │
   │   per dialogue → PersonalizedSample(                    │
   │     user           = dialogue_id,                       │
   │     target_task    = subset,                            │
   │     history_sessions=[warmup_pseudo_session],           │
   │     target_sessions=[full_dialogue_session])            │
   │                                                         │
   │  R2: build_uss_population_samples()                     │
   │   per (subset, test_dialogue) → PersonalizedSample(     │
   │     user           = "{subset}_population",             │
   │     target_task    = subset,                            │
   │     history_sessions=[K sampled train dialogues],       │
   │     target_sessions=[test_dialogue_session])            │
   └────────────────────┬────────────────────────────────────┘
                        ▼
   ┌──────────────────────────────────────────────────────────┐
   │ trace/collect_uss.py                                     │
   │  - 调 run_agent_on_sample() (复用 collect_personalized)  │
   │  - warmup 模式：post-filter 掉 turn_idx < n_warm 的记录  │
   │  - 写入 sample_id 与现有 eval 兼容的字段                 │
   └────────────────────┬─────────────────────────────────────┘
                        ▼
   ┌────────────────────────────────────────────────────────┐
   │ eval/personalized.py + eval/calibrate.py 复用，无改动  │
   │  - calibrate.py 按 (user, target_task, model) 分块     │
   │     R1: cache key = {dialogue_id}__{subset}__{model}   │
   │     R2: cache key = {subset}_population__{subset}__... │
   │  - CDF 用历史 score_distribution → 自动适配两种 mode    │
   └────────────────────────────────────────────────────────┘
```

## 4. 文件清单

| 文件 | 行数 | 作用 |
|---|---:|---|
| `detection/lib/uss_pipeline.py`    | ~250 | R1 / R2 sample builder（PersonalizedSample 适配） |
| `detection/trace/collect_uss.py`   | ~210 | 推理 entry point（薄封装 + warm-up post-filter） |
| `detection/scripts/collect_uss.sh` |  ~80 | 环境变量 → CLI 封装 |
| `detection/scripts/eval_uss.sh`    |  ~50 | delegate 到 `eval/personalized.py` |

**不改动**：`lib/uss_data.py`（已有 supervised 路线用）、`lib/memory.py`、
`lib/personalized_data.py`、`eval/*`、`scripts/calibrate.sh`、`scripts/diagnose_confusion.sh`。

## 5. 核心适配策略

### 5.1 R1 — Dialogue Warm-up

```python
sample = PersonalizedSample(
    user="JDDC_00002",
    target_task="JDDC",
    history_sessions=[warmup_session],   # 前 5 个 assistant turn
    target_sessions=[full_dialogue],     # 完整 dialogue（含 warmup）
)
```

关键技巧：**target_session 包含完整 dialogue**（含 warm-up 部分），让 evaluate_session 在
预测每个 target turn 时都看得到完整上下文（不会丢失 warm-up 的对话历史）。
warm-up 部分会被 LLM 重新预测一次（浪费一些 API 调用），但 `collect_uss.py` 在拿到
records 后做 `[r for r in records if turn_idx >= warmup_turns]` 过滤掉，对评估无影响。

为什么不让 target_session 只含 target 部分？因为 evaluate_session 的 history_window 是从
session 内动态构建的，target_session 只含 target 时模型看不到 warm-up 之前发生了什么。

**Warm-up 长度选择**：默认 `n_warm = 5`，过滤掉 dialogue 总 assistant turn < 5 + 1 的样本
（test 集中实测被过滤的 dialogue：CCPE 0、SGD 0、MWOZ 0、ReDial 4、JDDC 12 — 总共 16 条；
test 共 664 个 R1 block）。

### 5.2 R2 — Subset Population Rubric

```python
# 对 CCPE 而言，所有 test dialogue 共享同一组 history_sessions:
shared_history = sample_K_train_dialogues("CCPE", k=8, seed=42)

# 然后每条 test dialogue 是一个独立 block，但 user / cache key 共享:
sample_a = PersonalizedSample(
    user="CCPE_population",
    target_task="CCPE",
    history_sessions=shared_history,
    target_sessions=[test_dialogue_a],
)
sample_b = PersonalizedSample(
    user="CCPE_population",  # 同 cache key
    target_task="CCPE",
    history_sessions=shared_history,
    target_sessions=[test_dialogue_b],
)
```

由于 `build_user_memory` 按 `{user}__{target_task}__{model}.json` 缓存，CCPE 内所有
test dialogue 第一个跑完之后，后续 dialogue 直接从 cache 读 memory，**memory 只构建 1 次 / subset**。

**Population K 选择**：默认 8 条 train dialogue（约 70-100 个 turn 的对比证据）。
`build_memory_prompt` 内部 `_MAX_SESSIONS_PROMPT = 8`，再多也会被裁。

### 5.3 sample_id / cache key 表

| 场景 | user | target_task | sample_id | memory cache 文件 |
|---|---|---|---|---|
| R1 | dialogue_id (JDDC_00002) | subset (JDDC) | `JDDC_00002__JDDC__uss::JDDC::JDDC_00002::target.json__turn_5` | `outputs/uss/memory_cache/JDDC_00002__JDDC__gpt-4o.json` |
| R2 | `{subset}_population` (CCPE_population) | subset (CCPE) | `CCPE_population__CCPE__uss::CCPE::CCPE_00450::target.json__turn_3` | `outputs/uss/memory_cache/CCPE_population__CCPE__gpt-4o.json` |

每个 record 还附加 `dataset=uss`、`uss_mode=warmup|population`、`uss_subset=...` 三个分析字段。

## 6. 脚本调用示例

全部从 `detection/` 目录运行。

### 6.1 R1 完整跑通（warm-up + CDF）

```bash
cd detection

# no_memory baseline（也要按 warm-up 切，输出对应 turn 才能对齐 PG）
no_memory=1 mode=warmup model=gpt-4o subsets="CCPE SGD MWOZ ReDial JDDC" \
  bash scripts/collect_uss.sh
# → outputs/uss/gpt-4o_warmup_all5_test_no_memory.jsonl

# memory v2 + 不更新 memory（warm-up 只构建一次就够）
mode=warmup memory_update_mode=none model=gpt-4o \
  bash scripts/collect_uss.sh
# → outputs/uss/gpt-4o_warmup_all5_test_none.jsonl
# → outputs/uss/memory_cache/{dialogue_id}__{subset}__gpt-4o.json (一份/对话)

# CDF 校准（min_history_turns 调到 3，因为 warm-up 通常只有 5 个 gold）
python eval/calibrate.py \
  --input_jsonl outputs/uss/gpt-4o_warmup_all5_test_none.jsonl \
  --memory_cache_dir outputs/uss/memory_cache \
  --method cdf --min_history_turns 3

# 评估
result_files="no_mem=outputs/uss/gpt-4o_warmup_all5_test_no_memory.jsonl \
              mem=outputs/uss/gpt-4o_warmup_all5_test_none.jsonl \
              mem_cdf=outputs/uss/gpt-4o_warmup_all5_test_none_calCDF.jsonl" \
  bash scripts/eval_uss.sh
```

### 6.2 R2 完整跑通（subset population）

```bash
# 只跑 CCPE + SGD 验证
mode=population subsets="CCPE SGD" model=gpt-4o n_population_dialogues=8 \
  bash scripts/collect_uss.sh
# → outputs/uss/gpt-4o_population_CCPE+SGD_test_none.jsonl
# → outputs/uss/memory_cache/CCPE_population__CCPE__gpt-4o.json (1 份/subset)

# 同样可以跑 no_memory 对照
no_memory=1 mode=population subsets="CCPE SGD" \
  bash scripts/collect_uss.sh
```

### 6.3 与 supervised baseline 对比（外部脚本）

`predictor/lora_ordinal.py` 已经在 USS 上跑过 LoRA ordinal supervised baseline。把
training-free 的 `eval_uss.sh` 输出与监督 baseline 的 metric 表对接，即可在
`reports/pipeline/uss_pipeline_design.md` 之外的结果报告里横向对比。

## 7. session-level vs turn-level 评估说明

USS 与 RecLLMSim 一样是 **turn-level 评估**（每条记录 1 个 assistant turn）：

| 指标 | 含义 |
|---|---|
| MAE / RMSE | 平均每 turn 的分数误差 |
| Pearson / Spearman / QWK | 全 turn ordinal 相关 |
| 3/4 Boundary F1 | turn 级 SAT/DSAT 二分类 |
| User-aware (PU/WC Pearson) | per-dialogue（R1）或 per-subset（R2）内的相关性 |
| Personalization Gain | turn-level MAE 差 |

注意 USS 上 SAT/DSAT 边界严重不平衡（94%+ 是 DSAT，因为 score=3 落在 DSAT 那边），
boundary metrics 的 F1-SAT 会非常低，是数据本身的问题，不一定反映模型差。
**主指标建议盯 QWK 与 Pearson**。

## 8. Calibration 注意事项

`eval/calibrate.py` 在 USS 上的局限：

1. **R1 score_distribution 严重退化**：warm-up 5 个 turn 几乎全是 3 分，CDF 是 [0,0,1,0,0]
   的常数函数，所有预测被映射回 3 分。**预期 R1 + CDF ≈ 把所有预测压成 3**——这是数据
   skewed 的物理后果，不是 bug。
2. **R2 score_distribution 来自 K=8 个 train dialogue**，约 70-100 个 turn 的样本。
   分布仍然偏向 3，但有少量 2 / 4 分提供边界信号，CDF 能轻微生效。
3. `min_history_turns` 默认 5，R1 warm-up 刚好满足；R2 需保持 >=5。
4. R1 中很多 block 只有 1-2 个目标 turn（CCPE/MWOZ 等单一目的对话短的居多），
   `min_block_size=2` 会让一些 block 走 identity fallback。可以把 R1 的 `min_block_size`
   降到 1，但 CDF 需要 ≥ 2 个待校准点才有意义。

**结论**：CDF 校准在 USS 上**不期望**带来类似 RecLLMSim 的 −10% MAE 增益。R1 + CDF 主要
是 ablation，确认"distribution-aware calibration"在退化分布下不会破坏指标。

## 9. 推荐的 Pilot 实验

样本缩到 1-2 个 subset 快速验证 pipeline：

```bash
cd detection

# Pilot 1：CCPE 上 R1 跑通（仅 50 个 test dialogue）
mode=warmup subsets="CCPE" model=gpt-4o memory_update_mode=none \
  output_jsonl=outputs/uss/pilot_ccpe_r1.jsonl \
  bash scripts/collect_uss.sh

# Pilot 2：CCPE 上 R2 跑通
mode=population subsets="CCPE" model=gpt-4o \
  output_jsonl=outputs/uss/pilot_ccpe_r2.jsonl \
  bash scripts/collect_uss.sh

# Pilot 3：no_memory baseline（只跑 R1 切分模式即可，方便 PG 对齐）
no_memory=1 mode=warmup subsets="CCPE" \
  output_jsonl=outputs/uss/pilot_ccpe_baseline.jsonl \
  bash scripts/collect_uss.sh

# CDF 校准
python eval/calibrate.py \
  --input_jsonl outputs/uss/pilot_ccpe_r1.jsonl \
  --memory_cache_dir outputs/uss/memory_cache \
  --method cdf --min_history_turns 3 --min_block_size 1

# 综合评估
result_files="baseline=outputs/uss/pilot_ccpe_baseline.jsonl \
              r1=outputs/uss/pilot_ccpe_r1.jsonl \
              r1_cdf=outputs/uss/pilot_ccpe_r1_calCDF.jsonl \
              r2=outputs/uss/pilot_ccpe_r2.jsonl" \
  output_json=outputs/uss/pilot_ccpe_comparison.json \
  bash scripts/eval_uss.sh
```

成功标志（与 RecLLMSim 量级对比）：

- **R2 QWK > no_memory QWK**（即使只是几个百分点）→ subset 级 rubric 提供了有用的领域校准
- **R2 false_dsat_rate 下降**（少把真实 SAT 误判成 DSAT）→ memory v2 帮模型更好识别 4 分
- **R1 CDF 后 MAE 不显著恶化**（预期 ±0.05）→ pipeline 在退化分布下行为合理
- **R1 不退化优于 R2**：因为 R1 的 score_distribution 几乎全是 3 分，没有正向信号；
  R2 的 K=8 dialogue 综合分布更可参考

如果 R1 / R2 在 QWK 上都没有超过 no_memory baseline：
1. 先看 memory cache 里实际抽到的字段（`scoring_style` 是否被退化成"该用户大多打 3 分"）
2. 试 `n_population_dialogues=16` 或 `--save_memory_snapshots` 检查 memory 内容
3. 考虑改写 prompt 适配 USS 的 "task-success" 评分语义（不再是个性化偏好，而是任务完成度）

## 10. 与 cross_dataset_feasibility.md 的对齐

本 pipeline 落实了 `cross_dataset_feasibility.md` §2 的两条降级路线：

| §2 路线 | 本 pipeline 落实 |
|---|---|
| R1 对话内 warm-up 校准 | `build_uss_warmup_samples` + warmup post-filter |
| R2 subset 级 rubric memory | `build_uss_population_samples` + 共享 cache key |
| R3 dialogue-level 满意度 | **不实现**——本 pipeline 专注 turn-level 与 memory v2 rubric 的对接 |

`§9` 的 USS R1/R2 pilot 假设对应本文 §9 的"成功标志"。

## 11. 已知局限 / 后续工作

1. **R1 warm-up 浪费**：每个 warmup turn 都被 LLM 重跑一次预测（仅为提供历史窗口完整性），
   实际丢弃。改进方向：`evaluate_session` 增加 `skip_first_n_turns` 参数，让 warm-up 部分
   只入历史窗口、不出预测。
2. **Score skew 限制 CDF 价值**：USS 80%+ 的 turn 是 3 分，distribution-aware calibration
   退化为常数。可能的替代：`mean_shift` 在 USS 上比 CDF 更合适（已支持，传 `--method mean_shift`）。
3. **Memory prompt 与 USS 评分语义不完全对齐**：v2 rubric 是 "用户偏好 vs. 助手命中"
   语境，USS 是 "task-success"（订餐成功 / 推荐到合适电影 / ...），可能需要重写 task_context
   片段以提示模型"这是任务对话，应聚焦任务完成度"。
4. **跨 subset 迁移**：当前 cache 完全按 subset 隔离；如果想验证"subset A 的 memory 在 subset B
   上是否仍能 generalize"，需要扩展 collect_uss.py 支持 `--memory_subset_override`。
5. **与 supervised baseline 的端到端对比**：`predictor/lora_ordinal.py` 已经在 USS 上跑过 LoRA
   ordinal；本 training-free pipeline 跑通后应单独写 results 报告（`reports/pipeline/uss_pipeline_results.md`）
   做横向对比。
