# 个性化满意度感知实验结果报告

> 本文档面向后续研究者或 LLM，完整记录了任务定义、数据构造、代码架构、实验流程及结果分析，读后应能独立复现实验。

---

## 0. 任务背景与目标

**RecLLMSim** 是一个模拟真实用户与 LLM 助手对话的数据集，涵盖旅行规划、礼物准备、菜谱规划、技能学习规划 4 类任务，每个 session 由多轮对话组成，每个 assistant 轮次均附有人工标注的满意度分数（1-5）和不满意原因标签。

**本实验任务**：*Training-free 个性化满意度预测*。给定一个用户在某类目标任务上的新对话，仅利用该用户在其他任务上的历史满意度数据（无需训练），预测其对每轮 assistant 回复的满意度评分。

**核心问题**：不同用户对"好回复"的标准差异显著（有人严格要求具体步骤，有人只需大体正确），如何从历史数据中提炼个性化评分标准并应用到新任务中？

---

## 1. 数据划分（Cross-Task Split）

### 划分逻辑

对每个 `(用户, 目标任务类型)` 对构造一个 **PersonalizedSample**：

```
PersonalizedSample:
  user             = "User_0"
  target_task      = "旅行规划"
  history_sessions = User_0 在 [礼物准备, 菜谱规划, 技能学习规划] 中的所有 session
                     （满意度标签完全可见，用于构建记忆）
  target_sessions  = User_0 在 [旅行规划] 中的所有 session
                     （满意度标签用于评测，推理时不可见）
```

4 类任务 × 每用户 = 4 个 block。此设计强制模型进行**跨任务迁移**，避免直接利用目标任务历史信息作弊。

### 用户级划分

使用 `sklearn.GroupShuffleSplit`（`group=user_id`，`seed=42`）：
- **训练用户**（未使用）：22 人（20%）
- **测试用户**：90 人（80%），共 356 blocks，6474 个 assistant turns

代码位置：`detection/lib/personalized_data.py` → `build_personalized_samples()`

---

## 2. 核心数据结构

### SessionData（`lib/personalized_data.py`）

```python
@dataclass
class SessionData:
    user: str
    task: str                         # 任务类型（4 类之一）
    file_path: str
    task_context: str                 # 任务背景描述
    profile: dict                     # 用户画像（性别/年龄/职业/兴趣等）
    history: list[dict]               # 对话历史 [{"role": ..., "content": ...}]
    satisfaction_scores: list[int]    # 每个 assistant 轮的满意度 1-5
    dissatisfaction_reasons: list[str]
    chat_model: str
```

### UserMemory（`lib/memory.py`，v2 schema）

LLM 从历史 sessions 中提炼的用户个性化记忆，分为 LLM 生成部分（`UserMemoryContent`）和程序元信息（`UserMemory`）：

```python
class UserMemoryContent(BaseModel):
    avg_satisfaction_score: float          # 历史平均分（校准绝对分值）
    score_distribution: ScoreDistribution  # 1-5 分各自出现次数
    scoring_style: str                     # 评分风格：严格/宽松 + 校准说明
    four_vs_five_distinction: str          # ★ 4分→5分的具体门槛（对比式）
    three_vs_four_distinction: str         # ★ 3分→4分的具体门槛（对比式）
    user_specific_requirements: list[str]  # ★ 用户独有要求（禁止泛化描述）
    preferred_response_format: str         # 偏好的回复格式/结构
    task_specific_observations: list[TaskObservation]  # 各任务特定观察

class UserMemory(UserMemoryContent):
    memory_version: str = "v2"
    source_tasks: list[str]
    n_history_sessions: int
    n_history_turns: int
```

> **v2 设计要点**：核心字段 `four_vs_five_distinction` 和 `three_vs_four_distinction` 强制 LLM 输出**对比式评分边界**（基于实际历史轮次的反例），而非 v1 中容易生成的泛化描述（"回复要详细"等对所有用户都成立的废话）。

---

## 3. Agent Pipeline（三阶段流程）

代码位置：`detection/trace/collect_personalized.py`

```
Phase 1: Memory Building（每个 block 只需一次，可缓存）
  输入：history_sessions（含满意度标签）
  处理：build_memory_prompt() → LLM → UserMemoryContent → UserMemory
  输出：缓存到 outputs/personalized/memory_cache/{user}__{task}__{model}.json

Phase 2: Turn Evaluation（对每个 target session 逐轮预测）
  输入：UserMemory + 当前对话上下文（最近 5 轮历史窗口）+ assistant 回复
  处理：build_turn_eval_prompt() → LLM → TurnPrediction(classification, reason, analysis)
  输出：pred_score (1-5), pred_reason, analysis

Phase 3: Memory Update（可选，根据 memory_update_mode）
  none              → 不更新
  per_session       → 每个 target session 结束后更新（用预测分）
  per_session_oracle→ 每个 target session 结束后更新（用真实分，oracle 上界）
  per_turn          → 每轮预测后立即更新
```

### 关键 Prompt 设计

**Memory Building Prompt** 分两部分：
1. 按任务顺序展示历史 sessions（含满意度标签）
2. **按分数分组的对比证据**（将所有 5 分/4 分/3 分轮次分别聚合展示），迫使 LLM 直接比较相邻分数之间的差异

**Turn Evaluation Prompt**（v2 rubric 式）：
```
Step 1: 对照 three_vs_four_distinction → 是否达到 4 分基线？
Step 2: 若达到 → 对照 four_vs_five_distinction → 是否满足 5 分？
Step 3: 若未达 4 分 → 按缺陷严重度给 1/2/3 分
```

### 结构化输出

所有 LLM 调用均使用 `client.chat.completions.parse()` + Pydantic 模型，兼容 OpenAI API 和 vLLM（通过 `json_schema` response_format）。

---

## 4. 关键代码文件

| 文件 | 功能 |
|------|------|
| `lib/personalized_data.py` | 数据加载、Cross-Task Split、PersonalizedSample 构造 |
| `lib/memory.py` | UserMemory schema（v2）、Memory Building/Update/Eval 三类 prompt 构造 |
| `trace/collect_personalized.py` | 完整推理 pipeline，支持 4 种更新模式、断点续跑、vLLM 切换 |
| `eval/personalized.py` | 评测脚本：全局指标 + PU/WC 用户感知指标 + 多文件对比表 |
| `scripts/collect_personalized.sh` | OpenAI API 推理入口 |
| `scripts/collect_personalized_vllm.sh` | vLLM 本地推理入口 |
| `scripts/serve_vllm.sh` | vLLM 服务启动（含结构化输出配置） |
| `scripts/eval_personalized.sh` | 评测入口 |

---

## 5. 复现步骤

```bash
cd detection

# ── OpenAI API（GPT-4o-mini）────────────────────────────────────────────────
# 无记忆 baseline
model=gpt-4o-mini no_memory=1 bash scripts/collect_personalized.sh

# 有记忆（不更新）
model=gpt-4o-mini memory_update_mode=none bash scripts/collect_personalized.sh

# 有记忆（oracle 更新）
model=gpt-4o-mini memory_update_mode=per_session_oracle bash scripts/collect_personalized.sh

# ── vLLM 本地（Qwen3-8B）────────────────────────────────────────────────────
# Step 1：启动 vLLM 服务（激活 chat 环境后）
model=Qwen/Qwen3-8B bash scripts/serve_vllm.sh

# Step 2：运行推理
model=Qwen/Qwen3-8B memory_update_mode=none bash scripts/collect_personalized_vllm.sh

# ── 评测 ────────────────────────────────────────────────────────────────────
result_files="gpt4o_none=outputs/personalized/gpt-4o-mini_test_none_v2.jsonl \
  qwen3_none=outputs/personalized/Qwen_Qwen3-8B_test_none.jsonl" \
  bash scripts/eval_personalized.sh
```

---

## 6. 评测指标说明

| 指标 | 含义 |
|------|------|
| **MAE / RMSE** | 预测分与真实分的绝对/平方误差，越低越好 |
| **Pearson / Spearman** | 全局线性/秩相关，反映整体预测趋势 |
| **QW-Kappa** | 加权 Kappa，考虑评分偏差大小 |
| **PU-Pearson/Spearman/Kappa** | Per-User aggregation：对每个用户单独计算后取均值，衡量跨用户个性化区分 |
| **WC-Pearson/Spearman** | Within-user Centering：用户内去均值后计算，衡量对同一用户相对高低的判断 |

> PU/WC 是比全局指标更适合评测"个性化"能力的指标：全局指标可能因分数分布校准而虚高，PU/WC 真正反映模型是否学到了用户特异性偏好。

---

**任务**：Training-free 个性化满意度预测（跨任务 Cross-Task Split，测试集 90 用户 / 356 blocks / 6474 turns）  
**评测日期**：2026-04-20  
**模型**：GPT-4o-mini（API）、Qwen3-8B（vLLM 本地）

---

## 1. 完整指标对比

### 1.1 全局指标

| 模型 | Memory 配置 | MAE↓ | RMSE↓ | Pearson↑ | Spearman↑ | Kappa↑ |
|------|-------------|------:|------:|---------:|----------:|-------:|
| GPT-4o-mini | no\_memory | 0.7634 | 1.1512 | 0.1690 | 0.1055 | 0.1059 |
| GPT-4o-mini | none (v2) | 0.6517 | 0.9591 | 0.3146 | 0.2911 | 0.3002 |
| GPT-4o-mini | per\_session (v2) | 0.6608 | 0.9638 | 0.3065 | 0.2844 | 0.2914 |
| GPT-4o-mini | per\_session\_oracle (v2) | **0.6489** | **0.9510** | **0.3330** | **0.3064** | **0.3185** |
| Qwen3-8B | no\_memory | 0.7471 | 1.1300 | 0.1868 | 0.1184 | 0.1096 |
| Qwen3-8B | none | 0.7110 | 1.0122 | 0.2967 | 0.2820 | 0.2815 |
| Qwen3-8B | per\_session | 0.7133 | 1.0203 | 0.2812 | 0.2619 | 0.2694 |
| Qwen3-8B | per\_session\_oracle | 0.7141 | 1.0239 | 0.2737 | 0.2563 | 0.2632 |
| Qwen3-8B | per\_turn | **0.7087** | **1.0173** | 0.2794 | 0.2641 | 0.2694 |

### 1.2 用户感知指标

| 模型 | Memory 配置 | PU-Pearson↑ | PU-Spearman↑ | PU-Kappa↑ | WC-Pearson↑ | WC-Spearman↑ |
|------|-------------|------------:|-------------:|----------:|------------:|-------------:|
| GPT-4o-mini | no\_memory | 0.1231 | 0.0978 | 0.0779 | 0.1591 | 0.0886 |
| GPT-4o-mini | none (v2) | 0.1493 | 0.1138 | 0.1220 | 0.1571 | 0.0911 |
| GPT-4o-mini | per\_session (v2) | 0.1304 | 0.1003 | 0.1041 | 0.1460 | 0.0845 |
| GPT-4o-mini | per\_session\_oracle (v2) | 0.1507 | 0.1199 | 0.1250 | **0.1759** | **0.1054** |
| Qwen3-8B | no\_memory | 0.1538 | 0.1165 | 0.0890 | 0.1845 | 0.1006 |
| Qwen3-8B | none | **0.1997** | **0.1798** | **0.1646** | 0.2154 | 0.1637 |
| Qwen3-8B | per\_session | 0.1920 | 0.1703 | 0.1590 | 0.2071 | 0.1507 |
| Qwen3-8B | per\_session\_oracle | 0.1933 | 0.1737 | 0.1610 | 0.2073 | 0.1622 |
| Qwen3-8B | per\_turn | 0.1899 | 0.1702 | 0.1598 | 0.2049 | 0.1588 |

> **PU**（Per-User aggregation）：各用户内单独计算相关系数后取均值，反映模型对不同用户的区分能力。  
> **WC**（Within-user mean-Centering）：对每用户的预测/标注去均值后计算相关，衡量对同一用户内相对高低的判断能力。

---

## 2. 关键发现

### 2.1 Memory 对两个模型均有显著提升

| 对比 | MAE 改善 | Pearson 改善 |
|------|----------:|-------------:|
| GPT-4o-mini: no\_memory → none(v2) | −0.1117 (−14.6%) | +0.1456 (+86.2%) |
| Qwen3-8B: no\_memory → none | −0.0361 (−4.8%) | +0.1099 (+58.8%) |

Memory 在两个模型上均带来明显的相关性提升，但 GPT-4o-mini 的绝对增益更大，主要因其 no_memory baseline 质量更低（score=5 偏置更严重）。

### 2.2 Memory 更新策略的不同表现

**GPT-4o-mini（v2 memory）**

- `per_session_oracle` > `none` > `per_session`（更新噪声在 non-oracle 模式下略有负面影响）
- oracle 更新带来有意义的提升（Pearson +0.018），说明 v2 update prompt 能有效整合真实标签

**Qwen3-8B**

- `none` > `per_turn` > `per_session` > `per_session_oracle`（均差异甚小，< 0.003 MAE）
- oracle 更新反而略低于 none，说明 Qwen3-8B 对 update prompt 的响应较弱，更新基本无效
- per_turn 在 MAE 上略优于其他更新模式，但差异微小

### 2.3 用户感知指标（PU/WC）的反转现象

Qwen3-8B 的全局 Pearson/Spearman/Kappa **低于** GPT-4o-mini(v2)，但其 PU-Pearson、WC-Pearson 等用户感知指标**明显高于** GPT-4o-mini：

| | GPT-4o-mini none(v2) | Qwen3-8B none |
|---|---|---|
| Pearson（全局） | **0.3146** | 0.2967 |
| PU-Pearson | 0.1493 | **0.1997** |
| WC-Pearson | 0.1571 | **0.2154** |

**解读**：GPT-4o-mini 的全局指标优势来自更好的分数分布校准（整体 score 分布接近 gold），而 Qwen3-8B 在 per-user 和 within-user 维度上更善于区分同一用户内部的相对高低，体现出更强的个性化感知能力。两个模型在不同维度上各有优势。

### 2.4 模型能力对比小结

| 维度 | 优胜者 | 说明 |
|------|--------|------|
| 全局预测精度（MAE/RMSE） | GPT-4o-mini | v2 memory 下 MAE 低 0.065 |
| 全局相关性（Pearson/Kappa） | GPT-4o-mini | v2 memory 下领先约 0.02 |
| 用户内个性化区分（PU/WC） | Qwen3-8B | PU-Pearson 领先约 0.05 |
| Memory 更新收益 | GPT-4o-mini | oracle 更新有效；Qwen3 更新基本无效 |
| no\_memory baseline 质量 | Qwen3-8B | MAE 低 0.016，相关性高约 0.02 |

---

## 3. 实验配置说明

### 数据划分
- **Cross-Task Split**：每个用户的 4 类任务中，3 类作为历史（history），1 类作为目标（target）
- **用户级划分**：20% 训练用户 / 80% 测试用户（GroupShuffleSplit，seed=42）
- **测试集规模**：90 用户，356 blocks，6474 个 assistant turns

### Memory 版本
- **v1**（仅 GPT-4o-mini 早期实验，未列入本报告）：使用泛化满意/不满意模式列表，质量较差
- **v2**（本报告所有结果）：使用对比式评分边界（`four_vs_five_distinction` / `three_vs_four_distinction`）+ 按分数分组的对比证据，Pearson 相较 v1 提升约 +0.09

### Memory 更新模式
| 模式 | 说明 |
|------|------|
| `no_memory` | 无记忆 baseline，不构建 UserMemory |
| `none` | 构建记忆但不更新，整个 block 复用初始记忆 |
| `per_session` | 每个 target session 结束后用模型预测分更新记忆 |
| `per_session_oracle` | 每个 target session 结束后用真实标签更新（oracle 上界） |
| `per_turn` | 每轮预测后立即更新（仅 Qwen3-8B 测试） |

### 运行环境
- GPT-4o-mini：OpenAI-compatible API，16 workers 并发
- Qwen3-8B：vLLM 0.18.x 本地部署，`max_model_len=16384`，4 workers 并发
