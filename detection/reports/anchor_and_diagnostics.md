# Anchor 改造 + 5×5 混淆矩阵诊断报告

> 基于 memory v2 基线的第二轮改进。先通过诊断定位两个 backbone 模型各自的系统性错误模式，再尝试在 turn eval prompt 中插入 few-shot 历史锚点（anchor turns）改善校准。
>
> 相关代码（本次新增/修改）：
> - `lib/anchor_retrieval.py`（新）：TF-IDF 字符 n-gram 检索器
> - `lib/memory.py`（改）：`build_turn_eval_prompt` 增加 `anchor_turns` 入参 + rank-match 式评分步骤
> - `trace/collect_personalized.py`（改）：`--n_anchors` CLI 参数 + 每 sample 构建一次 retriever
> - `eval/diagnose_confusion.py`（新）：5×5 混淆矩阵 + 分数分布 + 用户级偏差
> - `scripts/diagnose_confusion.sh`（新）、`scripts/collect_personalized*.sh`（改）
>
> 相关输出：
> - `reports/diagnose_confusion.md`：对 6 份关键历史结果的完整诊断报告
> - `outputs/personalized/gpt-4o-mini_limit20_none_anchor3_qmatch.jsonl`：anchor 改造后的 20-block 小规模对照结果

---

## 1. 诊断工具与发现

### 1.1 工具

`eval/diagnose_confusion.py` 从 JSONL 结果文件直接计算，无需重跑：

- **5×5 混淆矩阵**（行 = gold，列 = pred，每格 count + 行归一化%）
- **边缘分布对比**（gold 分布 vs pred 分布）
- **每 gold 分数上的 MAE 分解 + 主要误判去向**
- **用户级均值偏差**（bias = mean_pred − mean_gold，阈值 ±0.2 归档"高估/低估/对齐"）

运行：
```bash
result_files="gpt4o_none=outputs/personalized/gpt-4o-mini_test_none_v2.jsonl \
  qwen3_none=outputs/personalized/Qwen_Qwen3-8B_test_none.jsonl" \
  output_md=reports/diagnose_confusion.md \
  bash scripts/diagnose_confusion.sh
```

### 1.2 对 6 份历史结果的诊断要点

完整数据见 `reports/diagnose_confusion.md`，核心归纳：

#### (A) 两个模型共享的主要失误：5 → 4 误判

| 结果 | 真实 5 分被预测为 4 的比例 | 5 分行 MAE |
|------|------:|------:|
| GPT-4o-mini none (v2) | **51.1%** | 0.654 |
| GPT-4o-mini per_session_oracle (v2) | 51.4% | 0.656 |
| Qwen3-8B none | 53.8% | 0.801 |
| Qwen3-8B per_session_oracle | 52.5% | 0.779 |

- gold=5 是最多的样本段（2918 / 6474，约 45%），两个模型在这一档都把过半样本压到 4 分
- 这直接导致全局 MAE 的主要亏空来源，也是 memory 更新（oracle 或非 oracle）未能显著改善的根源

#### (B) 两个模型相反的分布偏差方向

**GPT-4o-mini（v2 memory）：对 4 过度集中**

| 分数 | gold% | pred%（none v2） | Δ |
|---|---:|---:|---:|
| 3 | 11.3% | 9.4% | −1.9% |
| 4 | 37.8% | **59.5%** | **+21.6%** |
| 5 | 45.1% | 29.2% | −15.8% |

整体均值 pred 4.155 ≈ gold 4.203（差 0.05），看似对齐；但这是"把本该是 5 的压到 4、把本该是 3 的抬到 4"的**分布塌陷**。

**Qwen3-8B：整体向下偏移 + 3 分档被高估**

| 分数 | gold% | pred%（none） | Δ |
|---|---:|---:|---:|
| 3 | 11.3% | **16.6%** | **+5.3%** |
| 4 | 37.8% | 57.0% | +19.1% |
| 5 | 45.1% | 23.6% | −21.5% |

整体均值 pred=4.008 vs gold=4.203，**系统性低估 −0.20**。对低 gold 均值用户（严格型）高估最多、对高 gold 均值用户（宽松型）低估最多。

#### (C) 用户级偏差体现 backbone 特征

| 模型 | 用户数 | |bias|平均 | 高估用户 | 低估用户 | 对齐用户 |
|------|------:|---:|---:|---:|---:|
| GPT-4o-mini none (v2) | 90 | 0.202 | 16 | 24 | 50 |
| GPT-4o-mini oracle (v2) | 90 | 0.199 | 14 | 25 | 51 |
| Qwen3-8B none | 90 | 0.304 | 8 | 44 | 38 |
| Qwen3-8B oracle | 90 | 0.298 | 13 | 43 | 34 |

- Qwen3-8B 在 44 个用户上显著低估，主要集中于 mean_gold ≥ 4.6 的"宽松高分型"用户（如 User_82 gold=4.95 pred=4.02，bias=−0.93）
- GPT-4o-mini 偏差更对称且更小，50/90 用户在 ±0.2 内对齐

#### (D) Memory 更新对两个模型的影响

对比 none vs per_session_oracle：

| 指标 | GPT-4o-mini none→oracle | Qwen3-8B none→oracle |
|------|------:|------:|
| MAE | 0.6517 → 0.6489（−0.003） | 0.7110 → 0.7141（+0.003） |
| 5 分行 MAE | 0.654 → 0.656 | 0.801 → 0.779 |
| 1 分主要误判 | →4 (45.5%) → →3 (38.2%) | →3 (35.0%) → →3 (30.9%) |

Oracle 更新**几乎不改变分布**（pred% 变化均 < 1%），说明 update prompt 生成的新 memory 在评分决策层面没有带来实质影响。

---

## 2. Anchor Turn 改造

### 2.1 假设

诊断显示 rubric 提供的是抽象门槛描述（"5 分要求提供可执行的具体步骤…"），LLM 能读懂但无法把当前 turn 精确对上。插入 k 条**该用户历史上最相似的、带真实分数的 turn** 作为 few-shot 锚点，让模型从"应用抽象规则"降级为"找最相似的参考并对齐"，理论上能修正压 5→4 的问题。

### 2.2 实现

**检索器**（`lib/anchor_retrieval.py`）

- 语料：每个 `PersonalizedSample` 的 `history_sessions` 中所有 (user_msg, assistant_reply, score, reason) 四元组
- 向量化：`TfidfVectorizer(analyzer='char_wb', ngram_range=(2,4), sublinear_tf=True)`
- 索引文本：`user_msg user_msg assistant_reply`（user_msg 权重翻倍，按问题相似性检索，而非按回复风格）
- 检索：top-k 余弦相似度（默认不按分数多样化）
- 每个 block 构建一次，所有 target turns 复用

**Prompt 改造**（`lib/memory.py::build_turn_eval_prompt`）

- 在 rubric 之后、任务上下文之前插入"参考案例"块，每条案例展示任务、用户提问、助手回复（截断 200 字符）、真实满意度标签
- 评分步骤从 3 步扩为 5 步：
  - **Step 0 (Rank-Match)**：找出和当前回复整体质量最接近的 1 条案例，用它的分数作为初始估计
  - **Step 1 (Sanity-Check)**：用 rubric 校验；**仅当 rubric 明确提示重大差异时调整，否则保持 rank-match 分数**
  - Step A/B/C：原三步（3→4/4→5/<4）作为 rubric-only 回退路径
- 显式告知模型"不要把这些案例当作**完美标杆**去挑当前回复的毛病"——这是第一轮失败设计的直接补救

### 2.3 三次 Smoke 设计迭代（各 5 blocks / 87 turns / 同 1 用户集）

| 版本 | 检索文本权重 | 评分框架 | MAE（anchor）| MAE（baseline v2 none）| 核心问题 |
|------|------|------|------:|------:|------|
| **v-init**：diversify-by-score | user + reply × 2 | Step 0 critique + rubric 3 步 | 0.7241 | 0.6092 | 多样化强制展示不同分数，模型选"中间值"，压分 |
| **v-topk**：纯 top-k | user + reply × 2 | Step 0 critique + rubric 3 步 | 0.7816 | 0.6092 | 检索偏向长篇精致回复（都是高分），模型把其当标杆、挑当前回复毛病，更狠地压分 |
| **v-rankmatch-reply**：top-k + rank-match | user + reply × 2 | Step 0 rank-match + Step 1 sanity + rubric 回退 | 0.6552 | 0.6092 | 比 v-topk 改善但仍差；reply 权重导致检索仍偏向"看起来像"的高分案例 |
| **v-rankmatch-query ✅** | **user × 2 + reply** | rank-match + sanity-check | **0.5632** | 0.6092 | 按问题相似性检索，对 5 分预测变多，方向正确 |

关键教训：
1. **按回复相似性检索会引入"长篇高分"偏置**——top-k 下大部分 anchor 都是精致长回复，模型拿这些做标杆后系统性挑当前回复毛病压分
2. **critique 框架是致命错误**：告诉模型"对照案例判断是否达到 5 分"，会触发批评家模式系统性找理由给更低分
3. **rank-match 框架**+ 按 query 检索 是目前唯一让 MAE 改善的组合

### 2.4 扩大到 20 blocks 的小规模验证

配置：`memory_update_mode=none`, `n_anchors=3`, v-rankmatch-query，对照同 sample_id 的 v2 baseline。

| 指标 | baseline v2 none | anchor3 qmatch | Δ |
|------|------:|------:|------:|
| n matched turns | 392 | 392 | - |
| MAE | 0.6837 | **0.6429** | −0.041（−6.0%）✅ |
| RMSE | **0.9742** | 1.0000 | +0.026 |
| Pearson | **0.2364** | 0.2011 | −0.035 |
| Spearman | 0.2126 | **0.2502** | +0.038 ✅ |
| QW-Kappa | **0.2266** | 0.1954 | −0.031 |
| user mean \|bias\| | 0.201 | **0.130** | −0.071 ✅ |

预测分布对比（gold 分布：3×37 / 4×153 / 5×184）：

| 分数 | baseline pred | anchor pred | 向 gold 的漂移 |
|---|---:|---:|---|
| 3 | 49 | 46 | ≈ |
| 4 | 218 | 200 | −18（接近 gold 153） |
| 5 | 122 | **144** | +22（接近 gold 184） |

**结论**：20-block 小规模上 anchor 改造**改善了校准**（MAE −6%，用户级 \|bias\| −35%，5 分预测数 +18%），代价是 **Pearson 和 Kappa 小幅下降 ~3%**。Spearman 反而略升，说明秩序关系未显著受损、但线性关系变弱。

### 2.5 局限性说明

- smoke n=392 turns / 5 users，统计显著性不足
- Pearson/Kappa 小幅下降需在全量 6474 turns / 90 users 上验证是否真实
- Qwen3-8B 路径需要 vLLM 服务启动，本轮未跑

---

## 3. 当前状态与下一步建议

### 3.1 已交付

1. ✅ 诊断工具 `eval/diagnose_confusion.py` + 6 份历史结果的完整诊断（`reports/diagnose_confusion.md`）
2. ✅ Anchor retriever + prompt 改造 + CLI（`--n_anchors` 参数），默认 0 关闭
3. ✅ 20-block smoke 显示 anchor 改造在 GPT-4o-mini 上**改善 MAE / 用户级偏差**，但 **Pearson / Kappa 小幅下降**

### 3.2 待决策

**选项 A**：committing to full run
- `n_anchors=3` 跑完整 356 blocks，GPT-4o-mini 约需 3 小时 / $1-2 API 成本
- 全量结果才能回答"Pearson 下降是否统计显著 vs 噪声"

**选项 B**：组合 anchor + 原 rubric
- 同时跑 `n_anchors=3` 和原 baseline，用 **oracle-style ensemble**：对每个 turn 取两个预测的中位数/平均
- 预期：MAE 优势保留 + Pearson 损失被 baseline 部分挽回

**选项 C**：切换到 post-hoc 校准
- 保留现有 v2 结果，不改 prompt，改为在 JSONL 层面做**每用户 CDF 对齐**（用该用户历史分数分布校准预测）
- 优点：零额外 API 成本、纯数据后处理
- 缺点：依赖每用户足够多的历史样本（本数据集平均 12 session 够用）

> ✅ **选项 C 已实现并跑通**，详见 [`calibration_results.md`](./calibration_results.md)。
> 关键结论：Qwen3-8B +MS/CDF 全线大幅提升（MAE −10%+，Pearson +20%+，QWK +27%+），验证了"相对排序好、绝对校准差"的假设；GPT-4o-mini 的均值已近 gold，CDF 主要修复分布坍缩（边缘分布从 59% on 4 几乎完美对齐到 gold，用户级 |bias| −77%）。代码：`detection/eval/calibrate.py`、`detection/scripts/calibrate.sh`。

### 3.3 Qwen3-8B 路径

诊断显示 Qwen3-8B 的问题与 GPT 不同（系统性低估 + 3 分过多），anchor 改造是否同样工作属未知。建议：
1. ~~先做选项 C 的 per-user CDF 校准~~ **已做**：Qwen none MAE 0.711 → 0.628（MS）/ 0.636（CDF）；Qwen oracle MAE 0.714 → 0.641 / 0.636。
2. 若校准带来明显提升，再考虑 anchor 全量 —— 目前 MAE 已显著下降，但 Pearson 仍 <0.40，anchor 或其他 prompt 改进仍值得尝试。

---

## 4. 复现步骤

### 4.1 诊断

```bash
cd detection
result_files="gpt4o_none=outputs/personalized/gpt-4o-mini_test_none_v2.jsonl \
  gpt4o_oracle=outputs/personalized/gpt-4o-mini_test_per_session_oracle_v2.jsonl \
  qwen3_none=outputs/personalized/Qwen_Qwen3-8B_test_none.jsonl \
  qwen3_oracle=outputs/personalized/Qwen_Qwen3-8B_test_per_session_oracle.jsonl" \
  output_md=reports/diagnose_confusion.md \
  bash scripts/diagnose_confusion.sh
```

### 4.2 Anchor 推理（OpenAI API，GPT-4o-mini）

```bash
# 启用 anchor（n_anchors=3）
model=gpt-4o-mini memory_update_mode=none n_anchors=3 \
  bash scripts/collect_personalized.sh
# 输出文件：outputs/personalized/gpt-4o-mini_test_none_anchor3.jsonl
```

### 4.3 Anchor 推理（vLLM，Qwen3-8B）

```bash
# Step 1：启动 vLLM
model=Qwen/Qwen3-8B bash scripts/serve_vllm.sh

# Step 2：运行
model=Qwen/Qwen3-8B memory_update_mode=none n_anchors=3 \
  bash scripts/collect_personalized_vllm.sh
```

### 4.4 对 anchor 结果跑诊断

```bash
result_files="gpt4o_none=outputs/personalized/gpt-4o-mini_test_none_v2.jsonl \
  gpt4o_anchor3=outputs/personalized/gpt-4o-mini_test_none_anchor3.jsonl" \
  output_md=reports/diagnose_confusion_anchor.md \
  bash scripts/diagnose_confusion.sh
```
