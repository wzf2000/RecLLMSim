# ARR 2026 May 审稿意见分析与 Rebuttal 策略

本文档用于整理三位 reviewer 的评分、主要意见、潜在风险以及 rebuttal 的整体回复方向。
除可以直接复制进 rebuttal 的英文表述外，本文档主要使用中文，方便后续讨论和执行。

## 1. 总体判断

目前论文处于一个可以争取的状态。
三位 reviewer 中有两位给出了 Findings 倾向，另一位是 borderline Findings。
他们整体认可问题设定、数据价值、comparative memory 设计、baseline 覆盖和可复现性；主要担忧集中在“自动 evaluator 的可靠性是否足以支撑 benchmark 结论”。

| Reviewer | Overall | Soundness | Excitement | Confidence | 总体态度 |
|---|---:|---:|---:|---:|---|
| 8qA9 | 3.0 Findings | 3.5 | 3 | 3 | 整体正面，主要要求 benchmark 的 human validation 和 calibration sensitivity。 |
| dEr8 | 3.0 Findings | 3.0 | 3 | 4 | 整体正面且阅读细致，主要要求 evaluator agreement 的参考点和简单 history baseline。 |
| fUbw | 2.5 Borderline Findings | 2.5 | 3 | 4 | 最关键的 borderline reviewer，主要关注评分尺度、SAT/DSAT 术语、SPUR 比较、公平性、history 稀疏性和 domain scope。 |

整体 rebuttal 不适合强调“我们的 evaluator 已经足够接近人类评估”。
更稳妥的主线是：承认 PersTurnBench 是一个 calibrated automatic screening benchmark，而不是替代真实用户研究的最终 human preference leaderboard。
同时通过补充控制实验和 sensitivity analysis 来证明：虽然绝对自动评估仍有限，但我们的方法相比 baseline 有明确增益，benchmark 的 broad tiers 是相对稳定的。

## 2. Rebuttal 的核心主张

建议 rebuttal 主线围绕以下几点展开：

1. 我们同意 automatic personalized satisfaction judge 不能替代 direct user study。
2. 论文已经将 PersTurnBench 定位为 screening layer，修订版会进一步避免严格 leaderboard 式表述。
3. 我们会补充控制实验，区分 user rating-style prior、response-aware judging、post-hoc calibration 三类信号。
4. 我们会报告 raw / mean-shift / reference-CDF 的 sensitivity analysis，说明 broad model tiers 稳定，但 top models 之间的小差距需要谨慎解读。
5. 我们会澄清 1--5 satisfaction scale、修改 SAT/DSAT 术语，并修正 SPUR-style baseline 的描述。

可直接用于 rebuttal 的总体英文表述：

> We agree with the reviewers that PersTurnBench should be interpreted as an automatic screening benchmark rather than a replacement for direct user studies. In the revision, we will make this scope more explicit, add missing distribution-only controls, report raw and calibrated benchmark sensitivity analyses, and revise the SAT/DSAT terminology to better reflect the 1--5 annotation anchors.

## 3. Reviewer 8qA9

### 3.1 总体意见

8qA9 对论文整体较为正面。
他们准确理解了 personalized turn-level satisfaction evaluation、comparative memory、rubric-ordered LLM judge 和 post-hoc calibration。
他们明确认可 comparative memory 是区别于 profile/RAG personalization 的实质性设计，也认可 Table 2 和 ablation 的结果。
主要问题集中在 Section 5 的 benchmark 阶段，而不是 Section 4 的 evaluator verification。

### 3.2 Weakness 1: benchmark ranking 没有人类验证

Reviewer 的核心担忧：

- 七个 candidate models 的 ranking 完全依赖 frozen evaluator。
- evaluator 在 original human labels 上的 agreement 只有 moderate level。
- counterfactual candidate responses 并没有被原始用户打分。
- per-turn agreement 不必然推出 system-level ranking 正确。
- reviewer 希望有一个 small stratified human validation。

回复方向：

- 直接承认这是合理担忧，不要强辩。
- 强调 PersTurnBench 的 intended use 是低成本 screening，而不是直接替代人类评测。
- 如果一周内可行，优先补一个小规模 human validation。
- 如果不能联系原始用户，可以使用 independent annotators，并给他们 user profile、task background、conversation prefix、candidate response，让他们做 pairwise preference 或 1--5 rating。
- 需要明确说明这不是 original-user ceiling，但可以验证 frozen evaluator 的 counterfactual ranking signal。

最优补充实验：

- 50--80 个 pairwise comparisons，或者 60--100 个 candidate-response absolute ratings。
- Stratify by model pair：top vs middle、middle vs bottom、top vs bottom、close-pair。
- 报告 human preference 与 evaluator preference 的一致率，以及 high-margin / low-margin 子集的一致率。

如果无法完成 human validation：

- 在 rebuttal 中承认这是未来最重要的验证。
- 加强 raw/calibrated sensitivity 和 tiered interpretation。
- 在正文和 limitation 中明确 counterfactual candidate responses 没有 direct original-user labels。

可直接用于 rebuttal 的英文表述：

> We agree that validating the counterfactual replay ranking with human judgments is important. We will revise the paper to avoid interpreting PersTurnBench as a definitive human-preference leaderboard and instead present it as a reproducible screening benchmark. We are also preparing a small stratified human validation over replayed candidate responses; if completed within the revision window, we will report human-evaluator agreement on pairwise model comparisons.

### 3.3 Weakness 2: benchmark 只报告 calibrated scores

Reviewer 的核心担忧：

- Figure 2 和 Tables 8--9 都使用 reference-CDF calibrated score。
- Calibration 可能压缩模型之间的真实差异。
- 他们希望看到 raw 和 mean-shift 的 leaderboard，并计算 rank correlation。

回复方向：

- 这是高优先级、低成本问题，因为仓库里已有 raw、calMS、reference-CDF 相关结果。
- 应补充一个 sensitivity table：raw、reference mean-shift、reference-CDF，必要时再加 reference-transfer。
- 报告 Kendall's tau，说明 broad tiers 稳定，但 top/middle 内部小差距不应过度解释。

已有结果来源：

- `detection/reports/pipeline/static_replay_hard_model_comparison.md`
- `detection/reports/pipeline/static_replay_reference_calibration_design.md`

现有结果显示的核心结论：

- Top tier 基本稳定：`kimi-k2.6`、`glm-5.1`、`deepseek-v4-pro`。
- Middle group 基本稳定：`gpt-5.5`、`claude-opus-4-7`、`gemini-3.1-pro-preview`。
- `minimax-m2.7` 在不同 scoring views 下都明显较弱。
- 近似 rank correlation：raw vs refCDF 的 Kendall's tau 约 0.81；refMS/refTransfer vs refCDF 约 0.90。

可直接用于 rebuttal 的英文表述：

> We agree that reporting only calibrated scores can obscure score-scale effects. In the revision, we will add raw and reference mean-shift benchmark results together with the reference-CDF scores and report Kendall rank correlations across these leaderboards. The exact order of close models changes slightly, but the broad model tiers remain stable.

## 4. Reviewer dEr8

### 4.1 总体意见

dEr8 是最细致且相对正面的 reviewer。
他们明确肯定了问题设定、数据、comparative memory、reproducibility 和 honest scoping。
他们的意见非常适合作为修订论文的主线：补足 reference point、history-only baseline、benchmark sensitivity，以及修正 Table 1 中 Direct feedback 的表述。

### 4.2 Weakness 1: agreement scores 缺少参考点

Reviewer 的核心担忧：

- Table 2 的 Pearson/QWK/F1 数值没有 human agreement、test-retest reliability 或 human baseline 作为参考。
- Satisfaction 是 self-reported，因此理论 ceiling 可能远低于 1.0。
- 只看 baseline ranking 有意义，但不知道 0.36 离 ceiling 多远。

回复方向：

- 承认目前每个 turn 只有原始用户的一次满意度标注，因此没有直接的 repeated-label ceiling。
- 如果可行，补小规模 repeated rating 或 human-reference validation。
- 如果不可行，也需要在 limitation 里明确 unknown self-consistency ceiling。
- 至少补 user-level bootstrap confidence intervals，说明 Table 2 不是多次随机实验的均值，而是 fixed split 上的结果。

优先补充：

- Table 2 的 user-level bootstrap 95% CI。
- 小规模 human validation 如果可以和 8qA9 的需求合并完成。

可直接用于 rebuttal 的英文表述：

> We agree that the absolute agreement values need a reference point. The current dataset contains one original-user rating per assistant turn, so a direct test-retest ceiling is not available. We will add user-level bootstrap confidence intervals and explicitly state this limitation. If the additional validation study is completed, we will also report a human-reference agreement estimate for a stratified replay subset.

### 4.3 Weakness 2: Table 2 缺少 simple baselines

Reviewer 的核心担忧：

- 代码中已有 majority/mean predictors 和 user-history CDF baseline，但论文没有报告。
- user-history CDF/mean/median 是关键控制，因为它可以检验相关性是否主要来自用户评分分布，而不是 response-aware judging。

回复方向：

- 直接同意，并把这些 baseline 加进主表或附录。
- 这类 baseline 的结果已经在 `detection/reports/overview/history_baselines_results.md` 中。
- 需要注意解释：history-only baseline 在 full-score correlation 上很强，说明用户 strictness/leniency 很重要；但它们不读当前 response，在 low-satisfaction-side detection 上明显弱于 evaluator。

已有关键数值：

| Baseline | Pearson | Spearman | QWK | F1-low-side |
|---|---:|---:|---:|---:|
| global mean | N/A | N/A | 0.0000 | 0.0000 |
| global majority | N/A | N/A | 0.0000 | 0.0000 |
| user-history mean | 0.3560 | 0.3917 | 0.3105 | 0.2255 |
| user-history median | 0.3526 | 0.3972 | 0.3075 | 0.1643 |
| user-history CDF-hash | 0.2167 | 0.2579 | 0.2167 | 0.2752 |
| nearest-history turn | 0.2550 | 0.2891 | 0.2543 | 0.3106 |
| memory evaluator + CDF | 0.3601 | 0.3716 | 0.3595 | 0.3655 |

注意：

- 加入 history mean/median 后，不能再简单说 “our evaluator outperforms all baselines on all metrics”。
- 更准确说法是：history-only baseline 解释了很大一部分 correlation，但 evaluator 在 QWK 和 low-satisfaction-side F1 上更强，并且是 response-aware。

可直接用于 rebuttal 的英文表述：

> We agree that distribution-only history baselines are important controls. After adding them, we find that user rating-style priors explain a large fraction of full-score correlation. However, these baselines do not read the current response and are substantially weaker on low-satisfaction-side detection. The memory evaluator improves QWK over user-history mean and improves low-side F1 from 0.2255 to 0.3655, supporting the need for response-aware personalized judging.

### 4.4 Weakness 3: benchmark 依赖 weakly verified judge，且 calibration fallback 可能混合 raw/calibrated

Reviewer 的核心担忧：

- evaluator agreement moderate。
- top models 的 CI overlap。
- 旧版 calibration 对部分 replay records fallback to identity，可能混合 raw 和 calibrated values。

回复方向：

- 对 “judge moderate” 的部分用 screening benchmark + tiered interpretation 回应。
- 对 fallback 问题，使用 reference-based calibration 结果修正：目前 reference-CDF 可以覆盖全部 300 records 和 180 blocks，没有 identity fallback。
- 需要在论文中说明 official benchmark 使用 reference-based calibration，而不是 old block-wise calibration。

已有证据：

- `detection/reports/pipeline/static_replay_reference_calibration_design.md` 说明 reference-CDF 覆盖 300/300 records，无 identity fallback。

可直接用于 rebuttal 的英文表述：

> The submitted description did not make the calibration coverage sufficiently clear. In the revision, we will use the reference-based calibration protocol, which calibrates all 300 replay records using the full original-response prediction set as block-level reference context and does not fall back to identity mapping.

### 4.5 其他意见

Table 1 中 Direct feedback 的问题：

- Reviewer 认为 Direct 只适用于 Section 4 evaluator verification，而不适用于 benchmark candidate responses。
- 应修改 Table 1 caption 或列名。
- 推荐把 Direct 拆成 “direct labels for evaluator verification” 与 “direct labels for candidate responses”，或者加注说明 PersTurnBench 的 candidate responses 是 frozen judge 自动评分。

Contribution bullet typo：

- 直接感谢并修改即可。

## 5. Reviewer fUbw

### 5.1 总体意见

fUbw 是最需要重点争取的 reviewer。
他们认可主题及时、数据有用、baseline 全面、ablation 足够，但 soundness 只给 2.5。
他们的主要问题不是某一个结果，而是对 rating schema、SAT/DSAT terminology、SPUR mapping 和 history robustness 的整体疑虑。

### 5.2 W1: 1/2 和 4/5 边界不够清楚，缺少 trinary schema 比较

Reviewer 的核心担忧：

- 1/2 和 4/5 的区分价值没有充分讨论。
- 评分式 evaluation 可能存在 user self-inconsistency 和 evaluator nondeterminism。
- 可能 SAT-Neutral-DSAT 三分类更合理。

回复方向：

- 明确列出 1--5 annotation anchors。
- 解释为什么保留 1--5：
  - 3/4 是 minimum satisfaction threshold。
  - 4/5 是 excellence threshold。
  - 1/2 反映严重 failure 的强度差异。
  - Benchmark 聚合需要比 binary/trinary 更细的 satisfaction signal。
- 同时承认 score 3 是 neutral，不应简单叫 DSAT。
- 补一个 trinary SAT-Neutral-DSAT analysis：
  - DSAT: 1--2。
  - Neutral: 3。
  - SAT: 4--5。
  - 指标可用 macro-F1 / weighted-F1 / QWK。

可直接用于 rebuttal 的英文表述：

> We agree that the distinction between the 3/4 satisfaction boundary and the trinary SAT-Neutral-DSAT view should be clearer. The 1--5 scale is useful because our memory explicitly models both the minimum-satisfaction threshold and the excellence threshold, but we will add a trinary analysis with scores 1--2 as DSAT, score 3 as Neutral, and scores 4--5 as SAT.

### 5.3 W2: SPUR-style labels 映射到 3/4 不公平

Reviewer 的核心担忧：

- SPUR-style binary label 映射到 score 3/4 后参与 1--5 ordinal metrics 不太公平。
- reviewer 认为论文 footnote 误解了 SPUR，因为 SPUR pipeline 本身可以产生 neutral label。
- reviewer 希望更清楚说明 rubric 是重新从历史数据 induced，还是直接借用 SPUR paper。

回复方向：

- 承认当前 footnote 过度简化，可能误导。
- 澄清我们的实现是 “SPUR-style binary rubric-induction adaptation”，不是完整复现 SPUR 原 pipeline。
- 该 rubric 是在 personalized train split 上重新 induced，不是从 SPUR paper 借用。
- 如果保留在 1--5 表中，应明确它主要作为 boundary/rubric baseline，而不是 full-score predictor。
- 更好的修订是：在 trinary/boundary setting 中比较 SPUR-style baseline，主 1--5 表中弱化它。

现有实现：

- `detection/eval/spur/personalized.py` 将 gold score >= 4 作为 SAT，score <= 3 作为 DSAT。
- 预测时 SAT -> pred_score 4，DSAT -> pred_score 3。
- 因此它确实是 binary adaptation。

可直接用于 rebuttal 的英文表述：

> We agree that the current SPUR-style row should be described more carefully. Our implementation is a binary SPUR-style rubric-induction adaptation over the personalized train split, not a full reproduction of the original SPUR neutral-label pipeline. We will revise the footnote, provide the induced rubrics, and interpret this row as a boundary/rubric baseline rather than a full 1--5 predictor.

### 5.4 W3: SAT/DSAT 术语误导

Reviewer 的核心担忧：

- Score 3 在 rubric 中是 neutral，但论文把 <=3 都称作 DSAT。
- 需要澄清用户继续对话是否自动意味着 SAT 或 neutral。

回复方向：

- 全文尽量把 DSAT detection 改成 “low-satisfaction-side detection” 或 “non-satisfied/neutral-side detection”。
- 明确 score 3 是 neutral，但位于 3/4 minimum satisfaction boundary 的 lower side。
- 说明 continuation 本身不决定标签；用户是在对话后对每个 assistant turn 评分。
- score 4 表示 helpful but improvable；score 3 表示有启发但不足以满足需求。

可直接用于 rebuttal 的英文表述：

> We agree that using "DSAT" for all scores at most 3 is imprecise. We will revise the terminology to "low-satisfaction/neutral-side" for the 3/4 boundary and explicitly state that score 3 is neutral. Continued interaction is not treated as satisfaction by default; each assistant turn is rated by the user after the conversation.

### 5.5 W4: 缺少不同 user history amount 的鲁棒性实验

Reviewer 的核心担忧：

- Evaluator 可能依赖较多历史数据。
- 当 history sparse 时是否仍然有效未知。

回复方向：

- 如果计算预算允许，补 K-history ablation。
- 推荐在 20-user subset 上先做，和已有 ablation 对齐。
- K 可以设置为 1/2/4/all source conversations 或 5/10/20/all source turns。
- 指标：Pearson、QWK、low-side F1、memory construction parse/failure rate。

可直接用于 rebuttal 的英文表述：

> We agree that history sparsity is an important robustness question. We will add a history-budget ablation that constructs memory from a limited number of source conversations and compares it with the full-history setting on the same target turns.

### 5.6 W5: domain narrow + mixed satisfaction signals

Reviewer 的核心担忧：

- 数据和 evaluator 都在 planning-oriented Chinese conversations 上。
- 真实用户可能在同一 turn 内有 mixed SAT/DSAT signals。
- 单一 1--5 score 可能压缩了多方面反馈。

回复方向：

- 这是 limitation，应承认。
- 强调当前任务定义是预测用户最终 scalar satisfaction judgment，而不是 aspect-level feedback。
- Dissatisfaction reason 只是 low-side turns 的 primary reason，不覆盖所有 latent sub-reactions。
- 后续可以扩展到 aspect-level 或 multi-label satisfaction。

可直接用于 rebuttal 的英文表述：

> We agree that a scalar score cannot represent all mixed user reactions within a turn. Our current formulation targets the user's final turn-level satisfaction judgment, while the dissatisfaction reason records only the primary low-satisfaction category. We will clarify this scope and add aspect-level or multi-label satisfaction modeling as future work.

### 5.7 Q1: Table 2 是单次运行还是多次结果

回复方向：

- 说明 Table 2 是 fixed split 上的一次主运行结果。
- LLM-backed methods 使用固定 prompt 和低温设置，但仍可能有 decoding nondeterminism。
- 补 user-level bootstrap CI。
- 如果有时间，补小子集 repeated-run stability。

### 5.8 Q2: user history data 和 testing data 的 per-score distribution

当前可用统计：

| Split / source | Score 1 | Score 2 | Score 3 | Score 4 | Score 5 | Total |
|---|---:|---:|---:|---:|---:|---:|
| Test target turns | 123 | 251 | 733 | 2449 | 2918 | 6474 |
| Source histories used by test blocks, block-duplicated | 367 | 746 | 2188 | 7310 | 8707 | 19318 |

需要注意：

- “Source histories used by test blocks” 是按 user-target block 展开的，会重复计算同一个 source conversation，因为同一用户不同 target scenario 会使用不同 source history set。
- 如果论文附录中放表，应该清楚说明是 block-level evaluator input distribution。

### 5.9 Q3: 数据收集时 user-facing assistant 是什么模型

当前可用统计：

| User-facing assistant model | All labeled assistant turns | Sessions |
|---|---:|---:|
| gpt-4-turbo-preview | 2561 | 526 |
| gemini-2.0-pro-exp-02-05 | 2065 | 498 |
| claude-3-7-sonnet-20250219 | 2039 | 465 |
| deepseek-v3 | 1395 | 344 |

Test users 中的分布：

| User-facing assistant model | Labeled assistant turns | Sessions |
|---|---:|---:|
| gpt-4-turbo-preview | 2034 | 421 |
| claude-3-7-sonnet-20250219 | 1662 | 376 |
| gemini-2.0-pro-exp-02-05 | 1648 | 399 |
| deepseek-v3 | 1130 | 274 |

关键 rebuttal 点：

- Qwen3-8B 没有作为数据收集阶段的 user-facing assistant。
- 因此主 evaluator verification 不存在 “Qwen evaluator judging Qwen responses collected from users” 的直接 same-model bias。
- 但 replay benchmark 中仍可能存在 judge-family bias，应在 limitation 或 sensitivity 中承认。

### 5.10 Q4: SPUR rubric 是重新 induced 还是借用

回复方向：

- 明确是从 personalized train split 中重新 induced。
- 不是直接借用 SPUR paper 的 rubric。
- 在 appendix 或 supplementary 中提供 induced rubric。

## 6. 跨 Reviewer 的问题矩阵

| 问题 | Reviewer | 优先级 | 建议处理 |
|---|---|---:|---|
| Benchmark ranking 缺少 human validation | 8qA9, dEr8 | P0/P1 | 尽量补小规模 human validation；否则加强 proxy benchmark 和 tiered interpretation。 |
| 只报告 calibrated benchmark results | 8qA9, dEr8 | P0 | 补 raw/refMS/refCDF sensitivity 和 Kendall tau。 |
| 缺 simple history baselines | dEr8 | P0 | 加 global mean/majority、user-history mean/median/CDF。 |
| Agreement 缺少 human ceiling | dEr8, fUbw | P0/P1 | 补 user bootstrap CI；尝试 human validation；limitation 写清楚。 |
| SAT/DSAT terminology 不准确 | fUbw | P0 | 改成 low-satisfaction/neutral-side；补 trinary view。 |
| SPUR comparison 不公平 | fUbw | P0 | 修正 footnote；说明 binary adaptation；提供 induced rubric。 |
| History amount robustness | fUbw | P1 | 做 K-history ablation，至少 20-user subset。 |
| Score distribution / assistant model stats | fUbw | P0 | 附录补表，rebuttal 直接回答。 |
| Domain/mixed signals | fUbw | P2 | limitation 中承认并限定 scope。 |
| Table 1 Direct overclaim | dEr8 | P0 | 改表头/注释，区分 verification labels 和 candidate-response labels。 |
| Contribution typo | dEr8 | P2 | 直接修改。 |

## 7. 推荐 Rebuttal 结构

建议 rebuttal 不按三个 reviewer 完全逐条长篇回应，而是按核心问题组织：

1. Benchmark reliability and calibration sensitivity。
2. Evaluator verification controls and uncertainty。
3. Rating-scale terminology and SPUR-style baseline clarification。
4. Data statistics, source assistant models, and scope limitations。

这样可以同时回应多个 reviewer，避免重复。

可直接用于 rebuttal 的开头英文：

> We thank the reviewers for the constructive feedback. The main shared concern is whether a verified automatic evaluator can support replay-based model comparison. We will revise the paper to make the benchmark scope more explicit, report raw and calibrated sensitivity analyses, add missing history-only controls, and clarify the satisfaction-scale terminology and SPUR-style baseline.

