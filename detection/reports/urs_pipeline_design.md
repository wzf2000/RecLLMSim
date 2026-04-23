# URS 个性化满意度预测 — Trace & Eval Pipeline 设计

> 目的：把现有 RecLLMSim memory-agent（training-free）流水线迁移到 URS（User Reported
> Satisfaction）数据集。URS 的 label 是 **session-level**，与 RecLLMSim 的 turn-level
> 不同；本 pipeline 通过把每个 URS session 视为"单 assistant turn"的 SessionData，
> 最大化复用 `trace/collect_personalized.py` / `eval/personalized.py` / `eval/calibrate.py`
> 的现有代码路径，仅新增一套轻量适配层。

## 1. 范围 / 不在范围内

**本 pipeline 覆盖**：

- 读 URS `chinese_merged.json` / `english_processed.json` → `SessionData` / `PersonalizedSample`
- Cross-intent split（等价于 RecLLMSim 的 Cross-Task split）
- Memory v2 构建（复用 `lib/memory.py`，无改动）
- Session-level 评分预测（新增 `lib/urs_memory.py` 提供 prompt；`trace/collect_urs.py`
  提供 collection loop）
- 输出 JSONL schema 与现有 `eval/personalized.py` / `eval/calibrate.py` / `eval/diagnose_confusion.py`
  100% 兼容（sample_id / user / target_task / gold_score / pred_score / model）

**不在本迭代范围**：

- Per-turn 伪标签 / self-distillation（URS 无 turn-level gold，后续可作为 research direction）
- Anchor retrieval few-shot（URS 无逐 turn label，rank-match 不直接适用——先不接）
- Boundary_34 / selective_refute 等特化 prompt（先保证 v2 rubric 基线跑通）

## 2. 数据流总览

```
┌──────────────────────────┐   ┌──────────────────────────┐
│ data/urs/*.json          │   │ data/urs/*.json          │
│  (zh 107 user / 515 sess)│   │ (en 74 user / 305 sess)  │
└────────────┬─────────────┘   └────────────┬─────────────┘
             │                              │
             └──────────────┬───────────────┘
                            ▼
         ┌──────────────────────────────────────┐
         │ lib/urs_data.py::load_urs_sessions() │
         │  - uid 加语言前缀 (zh_/en_)          │
         │  - canonical intent 映射             │
         │  - satisfaction 1-5 映射             │
         │  - SessionData.satisfaction_scores   │
         │    = [session_score]  (len=1)        │
         └──────────────┬───────────────────────┘
                        ▼
         ┌─────────────────────────────────────────┐
         │ build_urs_personalized_samples(split=…) │
         │  → list[PersonalizedSample]             │
         │  - cross-intent, user-level 80/20 split │
         └──────────────┬──────────────────────────┘
                        ▼
     ┌──────────────────────────────────────────────────┐
     │ trace/collect_urs.py                             │
     │  Phase 1 build_user_memory (复用，memory v2)     │
     │  Phase 2 evaluate_urs_session (新；整段对话出1分)│
     │  Phase 3 update_memory_urs (复用 prompt；        │
     │          n_history_turns 按 session 递增)        │
     │                                                  │
     │  输出 JSONL: 每条记录对应一个 URS session        │
     └──────────────┬───────────────────────────────────┘
                    ▼
     ┌────────────────────────────────────────────────┐
     │ eval/personalized.py (复用，无改动)            │
     │  - 全局指标 / 边界 SAT/DSAT / user-aware /     │
     │    Personalization Gain                        │
     │                                                │
     │ eval/calibrate.py (复用，无改动)               │
     │  - CDF / mean_shift / identity                 │
     │  - 从 outputs/urs/memory_cache 读 score_dist   │
     │                                                │
     │ eval/diagnose_confusion.py (复用)              │
     └────────────────────────────────────────────────┘
```

## 3. 文件清单

**新增文件**：

| 文件 | 行数 | 作用 |
|---|---:|---|
| `detection/lib/urs_data.py`        | ~220 | URS 数据加载 + cross-intent split |
| `detection/lib/urs_memory.py`      |  ~85 | session-level 评分 prompt（带/不带 memory） |
| `detection/trace/collect_urs.py`   | ~320 | URS session-level 推理 entry point |
| `detection/scripts/collect_urs.sh` |  ~65 | shell 封装（环境变量 → CLI） |
| `detection/scripts/eval_urs.sh`    |  ~50 | 直接 delegate 到 `eval/personalized.py` |

**不改动的现有文件**：`lib/memory.py`、`lib/personalized_data.py`、`lib/satisfaction_constants.py`、
`eval/personalized.py`、`eval/calibrate.py`、`eval/diagnose_confusion.py`、`eval/user_aware.py`、
`scripts/calibrate.sh`、`scripts/diagnose_confusion.sh`。

## 4. 核心适配策略

### 4.1 SessionData "单 turn" 约定

URS session 是 session-level 标签 + 多轮对话。我们的约定：

```python
SessionData(
    user="zh_1",                          # 带语言前缀
    task="professional",                  # canonical intent slug
    file_path="urs::zh::00000.json",      # 伪文件名（保证 sample_id 唯一）
    task_context="[professional] {title}",
    profile={},                           # URS 无 profile；memory prompt 有 fallback
    history=[...],                        # 完整对话（供评分 prompt 展示）
    satisfaction_scores=[4],              # len=1，session-level
    dissatisfaction_reasons=["满意"],     # len=1；score<=3 时填 "其它"
    chat_model="ChatGPT",                 # URS 里的 llm 字段
)
```

这么做的好处：

- **memory building（`build_memory_prompt`）完全不用改**——它在 `_collect_turns_by_score`
  里就有 `if assistant_idx < len(satisfaction_scores)` 的 guard，多余的 assistant turn
  自动跳过；`n_history_turns = sum(s.assistant_turns)` 在 URS 下恰好 = `n_history_sessions`。
- **`sample_id = {user}__{target_task}__{file}__turn_0`** 与 RecLLMSim 同构，下游 eval /
  calibrate / diagnose 代码无需任何 if 分支。

### 4.2 Session-level 评分 prompt（新增）

`lib/urs_memory.py` 提供两个 builder：

- `build_session_eval_prompt(memory, profile, task_context, session_history)`
  —— 带 memory v2 rubric 的 session-level 评分 prompt，输出 1 个 1-5 分数
- `build_session_eval_prompt_no_memory(profile, task_context, session_history)`
  —— 无记忆 baseline

与 turn-level prompt 的关键差异：

1. **把整段对话（所有 user + assistant 轮）作为【完整对话】区块**，而不是 history_window + assistant_reply
2. **明确指令"对整段对话打 1 个分数"**，避免模型只评价最后一个 assistant 回复
3. **Step A/B/C 推理流程保持与 turn-level v2 一致**（3→4 门槛 / 4→5 门槛 / 缺陷分级），
   最大化"评分风格跨任务迁移"假设 H 的可比性

Reason prediction 仍用 `satisfaction_constants.py` 的中文 reason 标签（URS 无 reason gold，
仅用于格式合规校验；预测出的 reason 不参与主指标评估）。

### 4.3 Cross-Intent Split

与 RecLLMSim 的 Cross-Task Split 完全对称：

- `target_sessions` = 该用户在目标 intent 下的所有 session
- `history_sessions` = 该用户在**其他**6 个 intent 下的所有 session（标签可见）
- 用户层面 `GroupShuffleSplit(train_ratio=0.2, random_state=42)`；同一用户不会跨 train/test

实测样本数（`min_history_sessions=1`）：

| 配置 | users | blocks | target sessions |
|---|---:|---:|---:|
| split=all, languages=zh+en | 145 | 426 | 731 |
| split=test, languages=zh+en | 116 | 339 | 584 |
| split=test, languages=zh | 70 | 214 | 376 |
| split=test, languages=en | 46 | — | — |

## 5. 脚本调用示例

全部从 `detection/` 目录运行（脚本内部会 `cd "$SCRIPT_DIR/.."`）。

### 5.1 基本跑一遍（zh+en，API 模式）

```bash
cd detection

# 1) no_memory baseline
no_memory=1 model=gpt-4o split=test \
  bash scripts/collect_urs.sh
# → outputs/urs/gpt-4o_test_no_memory.jsonl

# 2) memory v2 + per_session update
model=gpt-4o split=test memory_update_mode=per_session \
  bash scripts/collect_urs.sh
# → outputs/urs/gpt-4o_test_per_session.jsonl
# → outputs/urs/memory_cache/{user}__{intent}__gpt-4o.json  (自动缓存)

# 3) CDF 校准（复用 eval/calibrate.py）
python eval/calibrate.py \
  --input_jsonl outputs/urs/gpt-4o_test_per_session.jsonl \
  --memory_cache_dir outputs/urs/memory_cache \
  --method cdf
# → outputs/urs/gpt-4o_test_per_session_calCDF.jsonl

# 4) 评估
result_files="no_mem=outputs/urs/gpt-4o_test_no_memory.jsonl \
              mem=outputs/urs/gpt-4o_test_per_session.jsonl \
              mem_cdf=outputs/urs/gpt-4o_test_per_session_calCDF.jsonl" \
  bash scripts/eval_urs.sh
```

### 5.2 单语言消融（验证 zh/en 评分风格是否可比）

```bash
# zh-only
languages="zh" output_jsonl=outputs/urs/gpt-4o_test_per_session_zh.jsonl \
  bash scripts/collect_urs.sh

# en-only
languages="en" output_jsonl=outputs/urs/gpt-4o_test_per_session_en.jsonl \
  bash scripts/collect_urs.sh
```

### 5.3 跨语言迁移（利用 zh/en 用户完全不相交的特性）

先用 `split=train` + `languages=zh` 构 memory cache，再用 `split=all` + `languages=en`
且**重用同一个 memory_cache_dir** 吗？不——因为用户不相交，不存在 memory 跨语言复用
路径。真正的跨语言迁移要看的是**评分风格是否跨用户群稳定**，做法：

```bash
# 在 zh 上完整跑通
languages="zh" bash scripts/collect_urs.sh
# 在 en 上完整跑通
languages="en" bash scripts/collect_urs.sh
# 比较两条 run 的 MAE / Pearson 差异，以及 CDF 校准增益是否一致
```

### 5.4 vLLM 本地模型

```bash
vllm_base_url=http://localhost:8000/v1 \
  model=Qwen/Qwen3-8B \
  bash scripts/collect_urs.sh
```

## 6. 输出 JSONL schema（每行一个 session）

```json
{
  "sample_id": "zh_1__professional__urs::zh::00000.json__turn_0",
  "user": "zh_1",
  "target_task": "professional",
  "target_file": "urs::zh::00000.json",
  "turn_idx": 0,
  "gold_score": 2,
  "pred_score": 3,
  "gold_reason": "其它",
  "reason_prediction": "不满足需求",
  "analysis": "StepA: ... StepB: ... StepC: ...",
  "model": "gpt-4o",
  "with_memory": true,
  "memory_update_mode": "per_session",
  "dataset": "urs",
  "chat_model": "ChatGPT"
}
```

`turn_idx=0` 恒定；`target_file` 是伪文件名。与 RecLLMSim 输出的字段完全对齐，只多了
`dataset` 和 `chat_model` 两个信息性字段（下游 eval 代码里未使用，不会报错）。

## 7. 评估指标的 session-level 解释

由于 URS 每个 session 产出 1 条记录，`eval/personalized.py` 的所有指标含义**按 session 聚合**：

| 指标 | RecLLMSim 解释（turn-level） | URS 解释（session-level） |
|---|---|---|
| MAE / RMSE | 平均每 turn 的分数误差 | 平均每 session 的分数误差 |
| Pearson / Spearman | 全 turn 相关性 | 全 session 相关性 |
| Kappa (quadratic) | turn-level QWK | session-level QWK |
| 3/4 Boundary F1 | turn 级 SAT/DSAT 分类 | session 级 SAT/DSAT 分类 |
| User-aware (PU / WC Pearson) | 每用户 turn 序列内相关 | 每用户 session 序列内相关 |
| Personalization Gain | 每 turn MAE 差 | 每 session MAE 差 |

**样本规模**：RecLLMSim test 集约 1800 turns；URS test（zh+en）约 584 sessions——样本数
约 1/3，**标准误更大**，对 small effect 的敏感度降低。结论需要更大 effect size 才能
确信。

## 8. Calibration 复用细节

`eval/calibrate.py` 按 `(user, target_task, model)` 分块，URS 下完全适用：

- `block key = (zh_1, professional, gpt-4o)` → 对应 memory cache
  `outputs/urs/memory_cache/zh_1__professional__gpt-4o.json`
- `min_history_turns` 参数 URS 下等价于"最少几个历史 session"——默认 5 可能偏严，
  URS 里有 **67 个用户只有 1-4 session 的 history**（按 `urs_dataset_stats`），建议
  `--min_history_turns 3` 开始实验
- CDF 的 quantile 映射 `q=(rank+0.5)/n` 要求块内预测数 ≥ 2；URS 中
  `sum(|target_sessions|) / n_blocks ≈ 1.7`，**很多 block 只有 1 个 target session**，
  会走 identity fallback。这是 session-level 的本质限制，解决办法：
  - 在 `calibrate.py` 里允许降级到 mean_shift（只需 hist_mean，不需要块内排序）
  - 或把多个 block 按 user 聚合做 calibration（跨 intent 做校准）——需要扩展
    `calibrate.py` 支持 `block_key_fn` 参数，留给后续改造

## 9. 本迭代不实现但值得后续做的点

1. **Anchor retrieval**：URS 无 turn-level label，`AnchorRetriever` 的"检索相似 turn
   拿分数"不适用。可以改造为"检索相似 session 拿 session 分数"——需要扩展
   `lib/anchor_retrieval.py`（或新写 `urs_anchor.py`）
2. **Boundary_34 系列 prompt**：v2 baseline 跑完后，若 3/4 边界误判仍主导误差，可以
   像 RecLLMSim 那样写 `build_session_eval_prompt_boundary_34` 系列
3. **Cross-language calibration**：zh 和 en 用户不相交，但如果把"ChatGPT 打分风格"
   视为跨语言共享，可以做一个按 `chat_model` 而不是 `user` 的 population-level calibration，
   作为 lower bound baseline
4. **Per-turn pseudo-label**：用 session-level gold 当弱监督，让 LLM 生成 per-turn
   pseudo-label，再做 turn-level eval——这能不能带来信号要消融

## 10. 快速 pilot（推荐先跑）

样本缩到 30-50 blocks、zh-only、gpt-4o 快速确认 pipeline：

```bash
cd detection

# 小样本 pilot：zh，限 50 users，先跑 no_memory baseline
no_memory=1 languages="zh" limit_users=50 model=gpt-4o \
  output_jsonl=outputs/urs/pilot_zh_no_memory.jsonl \
  bash scripts/collect_urs.sh

# 对应的 memory 版本
languages="zh" limit_users=50 model=gpt-4o \
  output_jsonl=outputs/urs/pilot_zh_per_session.jsonl \
  bash scripts/collect_urs.sh

# Calibration
python eval/calibrate.py \
  --input_jsonl outputs/urs/pilot_zh_per_session.jsonl \
  --memory_cache_dir outputs/urs/memory_cache \
  --method cdf --min_history_turns 3

# 评估对比
result_files="no_mem=outputs/urs/pilot_zh_no_memory.jsonl \
              mem=outputs/urs/pilot_zh_per_session.jsonl \
              mem_cdf=outputs/urs/pilot_zh_per_session_calCDF.jsonl" \
  output_json=outputs/urs/pilot_zh_comparison.json \
  bash scripts/eval_urs.sh
```

成功标志（与 RecLLMSim 上观察到的数量级对比）：

- **memory v2 Pearson 显著 > no_memory Pearson**（RecLLMSim 上 +0.03~+0.1）→ 评分风格
  跨任务迁移假设 H 在 URS 上成立
- **CDF 校准后 |user bias| 显著下降**（RecLLMSim 上 −70%+）→ 用户评分基准个性化有效
- **session-level QWK > 0.3**（与 RecLLMSim 0.30-0.36 可比）→ pipeline 在 session 粒度
  依然给出有意义的 ordinal 预测

如果 pilot 数字远低于上述阈值，优先排查：
1. 单语言 prompt 是否被另一种语言干扰（`lib/urs_memory.py` 当前 prompt 是中文系统消息，
   对 en session 可能要换英文 prompt——留作第二轮优化）
2. `min_history_sessions=1` 是否放得太宽，可以提到 3 看有效用户数和 QWK 变化
3. CDF 校准 block 内只有 1 个 target session 时的 identity fallback 比例（查日志
   `n_fallback_identity / n_blocks`）

## 11. 与 cross_dataset_feasibility.md 的对齐

本 pipeline 实现的是该报告 §3 `URS（本地）` 中的 **U1 路线**，与 §8.1 列出的 6 步改造
清单一一对应：

| §8.1 步骤 | 本 pipeline 落实 |
|---|---|
| 1. 数据读取，uid 语言前缀 | `lib/urs_data.py::load_urs_sessions` |
| 2. `lib/urs_data.py` 映射 + sample builder | 完成 |
| 3. `trace/collect_personalized.py` sample_builder 抽象 | 本迭代没有改 collect_personalized，而是新写 `collect_urs.py` 复用其 `_structured_parse` / `build_user_memory`；避免打破 RecLLMSim 入口 |
| 4. memory prompt 略 profile / task_context | `_format_profile` 已有空 profile fallback；URS 无需改 `lib/memory.py` |
| 5. `calibrate.py` 无需改动 | 确认：`block key` 和 `memory_cache` 目录通过 CLI 参数即可切到 URS |
| 6. `eval/personalized.py` session-level 解释 | 本文 §7 明确 |

§9 的 3 条 pilot 假设对应本文 §10 的"成功标志"。
