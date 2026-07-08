# ARR May Rebuttal 实验与修订计划

本文档用于整理 rebuttal 周内建议优先完成的实验、统计和论文修改。
除可以直接复制到 rebuttal 的英文句子外，本文档主要使用中文。

## 1. 优先级总览

| 优先级 | 任务 | 主要回应 reviewer | 预计成本 | 预期收益 |
|---|---|---|---:|---|
| P0 | 补 raw/refMS/refCDF benchmark sensitivity 和 Kendall tau | 8qA9, dEr8 | 低 | 直接回应 calibrated-only 和 ranking stability。 |
| P0 | 把 simple history-only baselines 加入 evaluator verification | dEr8 | 低 | 证明比较更公平，并区分 user prior 与 response-aware judging。 |
| P0 | 修改 SAT/DSAT 术语并补 trinary metric view | fUbw | 中 | 回应 borderline reviewer 的核心概念担忧。 |
| P0 | 补 score distribution 和 user-facing assistant model statistics | fUbw | 低 | 直接回答 Q2/Q3。 |
| P0/P1 | 小规模 replay human validation | 8qA9, dEr8 | 高 | 如果可行，是最能增强 benchmark 可信度的补充。 |
| P1 | Table 2 的 bootstrap confidence intervals | dEr8, fUbw | 低/中 | 回应 agreement reference 和 single-run statistics。 |
| P1 | History amount robustness ablation | fUbw | 中/高 | 回应 history sparsity，但优先级低于 calibration 和 terminology。 |
| P1 | SPUR clarification + induced rubric appendix | fUbw | 中 | 降低 SPUR unfair comparison 的风险。 |
| P2 | 修改 Table 1 Direct feedback 和 contribution typo | dEr8 | 低 | 必要清理项。 |
| P2 | 扩展 limitations 中的 domain / mixed signal | fUbw | 低 | 控制 scope。 |

## 2. P0: Benchmark calibration sensitivity

### 2.1 目标

回应：

- 8qA9 Weakness 2。
- dEr8 Weakness 3。

Reviewer 关注点：

- 目前 benchmark 主图只展示 reference-CDF calibrated scores。
- Calibration 可能压缩模型差异，导致 leaderboard 不可靠。
- 需要 raw、mean-shift、calibrated 多视角结果和 rank correlation。

### 2.2 需要报告什么

至少需要一个 compact table：

- Raw user-macro score。
- Reference mean-shift user-macro score。
- Reference-CDF user-macro score。
- Reference-CDF low-side rate。
- Kendall's tau：raw vs refCDF、refMS vs refCDF、可选 refTransfer vs refCDF。

已有结果来源：

- `detection/reports/pipeline/static_replay_hard_model_comparison.md`
- `detection/reports/pipeline/static_replay_reference_calibration_design.md`

建议表格草案：

| Model | Raw user macro | RefMS user macro | RefCDF user macro | RefCDF low-side rate |
|---|---:|---:|---:|---:|
| kimi-k2.6 | 4.8400 | 4.8535 | 4.8121 | 0.0133 |
| glm-5.1 | 4.8415 | 4.8273 | 4.8003 | 0.0267 |
| deepseek-v4-pro | 4.8024 | 4.8153 | 4.7911 | 0.0267 |
| gpt-5.5 | 4.7213 | 4.7578 | 4.6893 | 0.0567 |
| claude-opus-4-7 | 4.5553 | 4.6338 | 4.6361 | 0.0467 |
| gemini-3.1-pro-preview | 4.6310 | 4.6516 | 4.6283 | 0.0600 |
| minimax-m2.7 | 4.2057 | 4.3621 | 4.4256 | 0.1367 |

注意：

- 上表混合了已有报告中的结果，正式进论文前最好用统一脚本重新导出，避免口径差异。
- Raw 来自 old hard replay report；RefMS/RefCDF 来自 reference-based calibration report。
- 如果正文空间紧，可以放 appendix，正文只写一句 sensitivity 结论。

可直接用于 rebuttal 的英文：

> We agree that calibrated-only reporting can obscure score-scale effects. We will add raw and reference mean-shift benchmark results together with the reference-CDF view and report Kendall rank correlations across these leaderboards. The closest models may swap order, but the broad tiers remain stable.

## 3. P0: Simple history-only baselines

### 3.1 目标

回应：

- dEr8 Weakness 2。

Reviewer 关注点：

- Table 2 缺少 majority/mean predictors。
- user-history CDF baseline 是关键控制，因为它不读 response，只匹配用户评分分布。

### 3.2 现有结果

已有报告：

- `detection/reports/overview/history_baselines_results.md`

建议加入的 baseline：

- global mean。
- global majority。
- user-history mean。
- user-history median。
- user-history CDF-hash。
- nearest-history turn / nearest-history k3 可以保留为 retrieval baseline。

关键数值：

| Baseline | Pearson | Spearman | QWK | F1-low-side |
|---|---:|---:|---:|---:|
| global mean | N/A | N/A | 0.0000 | 0.0000 |
| global majority | N/A | N/A | 0.0000 | 0.0000 |
| user-history mean | 0.3560 | 0.3917 | 0.3105 | 0.2255 |
| user-history median | 0.3526 | 0.3972 | 0.3075 | 0.1643 |
| user-history CDF-hash | 0.2167 | 0.2579 | 0.2167 | 0.2752 |
| nearest-history turn | 0.2550 | 0.2891 | 0.2543 | 0.3106 |
| memory evaluator + CDF | 0.3601 | 0.3716 | 0.3595 | 0.3655 |

### 3.3 推荐解释

加入这些 baselines 后，结论应从“所有指标均超过所有 baseline”调整为更准确的版本：

- History-only baseline 很强，说明 user strictness / leniency 在该任务中非常重要。
- 但 history-only baseline 不读取当前 assistant response，不能用于 candidate response evaluation。
- Memory evaluator 的优势主要体现在 QWK 和 low-satisfaction-side detection，说明 response-aware judging 仍有价值。

可直接用于 rebuttal 的英文：

> Adding distribution-only history baselines shows that user rating-style priors explain a large part of the full-score correlation. However, these baselines do not read the target response and are substantially weaker on low-satisfaction-side detection. This supports the need for a response-aware personalized evaluator rather than only matching each user's historical score distribution.

## 4. P0: SAT/DSAT terminology 和 trinary view

### 4.1 目标

回应：

- fUbw W1。
- fUbw W3。

Reviewer 关注点：

- 论文把 score <= 3 叫 DSAT，但 score 3 在 rubric 中其实是 neutral。
- 需要说明为什么使用 1--5，而不是 SAT-Neutral-DSAT。
- 需要 clarifiy genuine continuation 是否自动代表 SAT。

### 4.2 修改建议

术语替换：

- `DSAT detection` 改成 `low-satisfaction-side detection` 或 `non-satisfied/neutral-side detection`。
- 如果表头太长，可以用 `Low-side F1`，caption 解释 low side means scores at most 3。
- `SAT rate` 可以保留为 score >= 4。
- `DSAT rate` 如果是 score <= 3，应改成 `Low-side rate`。

正文解释：

- score 3 是 neutral，但低于 minimum satisfaction boundary。
- continuation 不决定标签。
- 用户是在对话后对每个 assistant turn 打分。

### 4.3 Trinary metric

建议补一个 appendix table：

- DSAT: score 1--2。
- Neutral: score 3。
- SAT: score 4--5。

可报告指标：

- macro-F1。
- weighted-F1。
- class-wise F1。
- optional QWK。

最小实验范围：

- Main evaluator。
- user-history mean/median。
- nearest-history turn。
- zero-shot/few-shot 或 Prometheus judge。
- SPUR-style baseline 如果保留，需要注明 binary adaptation 不产生 neutral。

可直接用于 rebuttal 的英文：

> We agree that score 3 should not be described as dissatisfied. We will revise the terminology to "low-satisfaction/neutral-side" for the 3/4 boundary and add a trinary SAT-Neutral-DSAT analysis with scores 1--2, 3, and 4--5.

## 5. P0: Data statistics 和 user-facing assistant model statistics

### 5.1 Score distribution

回应：

- fUbw Q2。

当前可用统计：

| Split / source | Score 1 | Score 2 | Score 3 | Score 4 | Score 5 | Total |
|---|---:|---:|---:|---:|---:|---:|
| Test target turns | 123 | 251 | 733 | 2449 | 2918 | 6474 |
| Source histories used by test blocks, block-duplicated | 367 | 746 | 2188 | 7310 | 8707 | 19318 |

放置建议：

- Rebuttal 中直接给简表。
- 论文附录 Data Statistics 里补一行/表。
- Caption 要说明 source histories 是按 evaluator block 展开的，因此可能重复统计同一个原始 turn。

### 5.2 User-facing assistant model distribution

回应：

- fUbw Q3。

当前可用统计：

| User-facing assistant model | All labeled assistant turns | Sessions |
|---|---:|---:|
| gpt-4-turbo-preview | 2561 | 526 |
| gemini-2.0-pro-exp-02-05 | 2065 | 498 |
| claude-3-7-sonnet-20250219 | 2039 | 465 |
| deepseek-v3 | 1395 | 344 |

Test users 中：

| User-facing assistant model | Labeled assistant turns | Sessions |
|---|---:|---:|
| gpt-4-turbo-preview | 2034 | 421 |
| claude-3-7-sonnet-20250219 | 1662 | 376 |
| gemini-2.0-pro-exp-02-05 | 1648 | 399 |
| deepseek-v3 | 1130 | 274 |

关键解释：

- 数据收集阶段的 user-facing assistant 不包含 Qwen3-8B。
- 主 evaluator 使用 Qwen3-8B，因此 original-response verification 不存在同模型自评偏置。
- Replay benchmark 中仍可能有 judge-family bias，应放 limitation 或 sensitivity。

可直接用于 rebuttal 的英文：

> The user-facing assistants used during data collection were GPT-4 Turbo, Gemini 2.0 Pro experimental, Claude 3.7 Sonnet, and DeepSeek-V3. Qwen3-8B was not used as a user-facing assistant, which reduces direct same-model bias in the evaluator verification stage.

## 6. P0/P1: Replay human validation

### 6.1 目标

回应：

- 8qA9 Weakness 1。
- dEr8 Weakness 3。

这是最能提升 benchmark 可信度的补充，但成本最高。

### 6.2 最理想版本

联系原始用户，让他们对 candidate replay responses 重新评分或做 pairwise preference。
这最符合 personalized satisfaction 的定义，但现实上可能很难在 rebuttal 周内完成。

### 6.3 可行替代版本

找 independent annotators。
给定：

- user profile。
- task background。
- conversation prefix。
- current user request。
- candidate response A/B。

让 annotator 判断哪个 response 更可能让该用户满意。

### 6.4 推荐设计

优先做 pairwise preference，而不是 absolute 1--5 rating。
Pairwise 更容易、噪声更低，也更贴合 ranking validation。

采样建议：

- 60--100 pairs。
- 包括 top vs bottom、top vs middle、middle vs middle。
- 包括 evaluator margin 大和 margin 小的样本。
- 覆盖四个 task。

报告：

- Human preference 与 evaluator preference 的一致率。
- High-margin subset 的一致率。
- Bootstrap CI。

可直接用于 rebuttal 的英文：

> We are conducting a small stratified pairwise validation over replayed candidate responses. Annotators are given the user profile, task background, conversation prefix, and two candidate responses, and asked which response would better satisfy the target user. We will report agreement with the frozen evaluator, especially on high-margin comparisons.

如果做不了，则使用：

> We agree that direct human validation of counterfactual candidate responses is the most important next step. We will revise the paper to state this limitation explicitly and interpret PersTurnBench as an automatic proxy benchmark for screening rather than a final human preference ranking.

## 7. P1: Table 2 bootstrap confidence intervals

回应：

- dEr8 W1。
- fUbw Q1。

设计：

- 按 user-level bootstrap，而不是 turn-level bootstrap。
- 每次有放回采样 users，包含其所有 target turns。
- 计算 Pearson、Spearman、QWK、Low-side F1。
- 报 95% CI。

原因：

- 同一用户内 turns 不独立。
- Personalized evaluation 更关心跨用户稳健性。

论文放置：

- 如果主表太拥挤，主表仍放 point estimate，appendix 放 CI。
- Rebuttal 中给主要 evaluator 和强 baseline 的 CI 即可。

## 8. P1: Repeated-run stability

回应：

- fUbw W1。
- fUbw Q1。

设计：

- 固定一个小 subset，例如 20 users 或 300--500 turns。
- Qwen3-8B memory evaluator 重跑 3 次。
- 报 run-to-run exact score agreement、run-to-run QWK、各指标标准差。

注意：

- 这只能说明 evaluator nondeterminism。
- 不能代替 human self-consistency ceiling。
- 不要在 rebuttal 中把它包装成人类可靠性上界。

## 9. P1: History amount robustness ablation

回应：

- fUbw W4。

设计：

- 固定 target turns。
- 构造 memory 时限制 source history 数量。
- 选项：
  - K=1 source conversation。
  - K=2 source conversations。
  - K=4 source conversations。
  - all source conversations。
- 如果 full split 太贵，先用 20-user subset。

指标：

- Pearson。
- QWK。
- Low-side F1。
- memory parse/failure rate。

预期解释：

- 如果 K 增大性能稳定提升，可以说明 evaluator 确实利用历史。
- 如果 K=1 下降明显，也可以作为合理 limitation：方法需要一定历史信号。

可直接用于 rebuttal 的英文：

> To assess robustness under sparse histories, we will add a history-budget ablation that limits memory construction to K source conversations while keeping the target turns fixed.

## 10. P1: SPUR clarification 和 rubric appendix

回应：

- fUbw W2。
- fUbw Q4。

最低限度修改：

- Table 2 footnote 改写。
- 明确当前是 binary SPUR-style adaptation。
- 明确 rubric 是从 personalized train split 重新 induced。
- 不再说 SPUR 本身不能产生 neutral。

更好版本：

- 把 induced SPUR rubrics 放 appendix。
- 或者在 supplementary/release 中说明路径。
- 如果实现 trinary SPUR 成本太高，不建议本周强行做。

可直接用于 rebuttal 的英文：

> The SPUR-style rubric was re-induced from the personalized training split; it was not borrowed directly from the SPUR paper. We will provide the induced rubric and revise the description to avoid implying that the original SPUR pipeline cannot produce neutral labels.

## 11. P2: 论文文本修订 checklist

### Section 1

- Reword dEr8 提到的 awkward contribution bullet。
- 不再暗示 benchmark ranking 等价于 human ranking。

### Section 2

- 修改 Table 1 的 Direct column。
- 区分：
  - evaluator verification 有 direct original-user labels。
  - candidate replay responses 没有 direct original-user labels。

### Section 4

- 加 simple history baseline。
- 加 Table 2 CIs 或 appendix reference。
- 修改 DSAT terminology。
- 明确 score 3 是 neutral/lower side。

### Section 5

- 加 raw/refMS/refCDF sensitivity。
- 用 broad groups / tiers 替代 strict ranking。
- 明确 reference-CDF 没有 identity fallback。
- 如果完成人类验证，加入结果；否则加入 limitation。

### Appendix

- 加 score distribution。
- 加 user-facing assistant model distribution。
- 加 SPUR induced rubric 或实现细节。
- 加 trinary metric table。
- 加 history amount ablation，如果完成。

### Limitations

- 加 unknown human self-consistency ceiling。
- 加 counterfactual candidate responses not rated by original users。
- 加 scalar label collapses mixed satisfaction signals。

## 12. 一周执行顺序建议

### Day 1

- 统一导出 raw/refMS/refCDF benchmark sensitivity table。
- 计算 Kendall tau。
- 整理 history-only baselines 并决定放主表还是附录。
- 整理 score distribution 和 source assistant model distribution。

### Day 2

- 做 trinary metric conversion。
- 修改 SAT/DSAT terminology 的论文草稿。
- 整理 SPUR induced rubric 和 footnote 修正。

### Day 3--5

- 尽量做人类 pairwise validation。
- 同时跑 Table 2 bootstrap CI。

### Day 5--6

- 如果 compute 允许，跑 history amount robustness。
- 如果 human validation 做不了，跑 repeated-run stability 作为次优补充。

### Day 6--7

- 写 rebuttal。
- 优先呈现已经完成的新增实验。
- 不要承诺无法在 revision 中落地的内容。

## 13. Rebuttal 写作注意

不要写成：

> Our evaluator is reliable enough to replace human evaluation.

应写成：

> Our evaluator provides a reproducible personalized proxy for screening candidate models, and we will make the uncertainty and scope of this proxy more explicit.

不要把 calibrated ranking 说成最终 ranking。
应使用：

- broad groups。
- tiers。
- screening signal。
- sensitivity across scoring views。

