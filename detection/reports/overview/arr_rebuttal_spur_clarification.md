# ARR Rebuttal: SPUR-Style Baseline Clarification

本文档整理 Reviewer fUbw 关于 SPUR-style baseline 的问题和可直接用于 rebuttal / revision 的澄清内容。
本次检查不需要重新运行 SPUR，不新增实验代码。
当前 induced rubric 已经存在于本地输出中。
结论是这部分可以直接处理：在 rebuttal 中澄清当前实现是基于本文训练集重新诱导 rubric 的 binary boundary-oriented SPUR-style adaptation，并在附录或 artifact 中给出 induced rubric。

## 1. Reviewer Concern

Reviewer fUbw 的主要问题包括：

- 当前 Table 2 中 SPUR-style label 到 score 3/4 的映射可能不公平。
- 原稿 footnote 容易被理解为 SPUR 原方法不能产生 neutral label，但 reviewer 指出 SPUR pipeline 本身可以产生 neutral。
- 需要澄清本文使用的 rubric 是重新从本文数据中 induced，还是直接借用了 SPUR paper 的 rubric。

## 2. What Our Current Baseline Actually Does

当前实现是一个 binary SPUR-style rubric-induction adaptation，而不是 SPUR 原论文完整 neutral-label pipeline 的复现。
它使用 personalized train split 重新诱导 rubric，不直接使用 SPUR paper 中的现成 rubric。

实现来源：

- Code: `detection/eval/spur/personalized.py`
- Rubric extraction/summarization: `detection/eval/spur/rubrics.py`
- Scoring prompt: `detection/eval/spur/scoring.py`
- Run script: `detection/scripts/run_personalized_spur.sh`
- Existing design note: `detection/reports/overview/spur_personalized_baseline_design.md`

输出来源：

- Phase-1 candidates: `detection/outputs/spur_personalized/qwen3_8b_direct/phase1_candidates.json`
- Phase-2 induced rubrics: `detection/outputs/spur_personalized/qwen3_8b_direct/phase2_rubrics_k10.json`
- Test predictions: `detection/outputs/personalized/spur_direct_qwen3_8b_personalized_test.jsonl`
- Metrics: `detection/outputs/personalized/spur_direct_qwen3_8b_personalized_test_metrics.json`

Training split statistics from the metrics file:

- train users: 22
- train blocks: 85
- train target turns: 1,413
- test users: 90
- test blocks: 356
- test target turns: 6,474

Rubric induction details:

- Binary label construction: `score >= 4` is SAT, `score <= 3` is low-side / DSAT in the original code variable names.
- Phase 1 samples up to 150 rows per label and asks the LLM to extract 3 rubric candidates per row.
- The cached Phase-1 file contains 450 SAT candidates and 450 low-side candidates.
- Phase 2 summarizes candidates into 10 SAT rubrics and 10 low-side rubrics.
- Phase 3 applies the learned rubrics to each target turn and predicts only `SAT` or `DSAT`.
- For Table 2 ordinal metrics, `SAT` is mapped to `pred_score=4`, and `DSAT` is mapped to `pred_score=3`.

Therefore, this row should be interpreted as a boundary-oriented rubric baseline.
It should not be presented as a full 1--5 satisfaction predictor or a full trinary SAT-Neutral-DSAT predictor.
The revised text should also avoid implying that the original SPUR framework cannot support neutral labels.

## 3. Induced Rubrics

The following rubrics are copied from `detection/outputs/spur_personalized/qwen3_8b_direct/phase2_rubrics_k10.json`.
They are the actual induced rubrics used by the reported Qwen3-8B SPUR-style baseline.

| # | SAT rubric | Low-side rubric |
|---:|---|---|
| 1 | 个性化定制与需求匹配 | 未充分考虑用户预算限制及价格透明度 |
| 2 | 结构化信息呈现提升可读性 | 缺乏个性化定制与兴趣偏好适配 |
| 3 | 预算控制与成本优化建议 | 信息不完整或关键细节缺失 |
| 4 | 情感支持与积极反馈 | 推荐方案缺乏灵活性与调整空间 |
| 5 | 实用技巧与避坑指南 | 未明确区分需求优先级导致偏离核心 |
| 6 | 多维度资源整合与推荐 | 操作指引不清晰或步骤执行困难 |
| 7 | 分阶段规划与任务分解 | 未考虑特殊需求（健康/文化/场景等） |
| 8 | 灵活调整方案适配场景 | 时间规划不合理或节奏安排不足 |
| 9 | 安全健康注意事项强调 | 推荐内容同质化缺乏创新性与独特性 |
| 10 | 多平台资源获取优化 | 资源获取障碍或实施路径不明确 |

English translations for paper appendix:

| # | SAT rubric | Low-side rubric |
|---:|---|---|
| 1 | Personalized tailoring and requirement matching | Insufficient consideration of budget limits and price transparency |
| 2 | Structured information presentation for readability | Lack of personalization for interests and preferences |
| 3 | Budget control and cost-optimization advice | Incomplete information or missing key details |
| 4 | Emotional support and positive feedback | Lack of flexibility or room for adjustment |
| 5 | Practical tips and risk-avoidance guidance | Failure to distinguish priorities, causing drift from the core need |
| 6 | Multi-dimensional resource integration and recommendation | Unclear operational guidance or difficult-to-execute steps |
| 7 | Stage-wise planning and task decomposition | Failure to consider special needs such as health, culture, or scenario constraints |
| 8 | Flexible adjustment to scenario changes | Unreasonable time planning or insufficient pacing |
| 9 | Emphasis on safety and health considerations | Homogeneous recommendations lacking novelty or distinctiveness |
| 10 | Multi-platform resource access optimization | Resource-access barriers or unclear implementation path |

## 4. Recommended Paper Revision

Main Table 2 footnote:

> The SPUR-style row is a binary boundary-oriented adaptation: rubrics are re-induced from the personalized training split, and SAT/low-side predictions are mapped to scores 4/3 for ordinal metrics.

Appendix baseline details:

> We do not claim to reproduce the full neutral-label SPUR pipeline.
> Instead, we adapt SPUR-style rubric induction as a supervised boundary baseline under our personalized split.
> The induced rubrics are learned from the training users and are not borrowed from the SPUR paper.

## 5. Rebuttal-Ready Wording

> We agree that the SPUR-style row should be described more carefully.
> Our implementation is a binary SPUR-style rubric-induction adaptation over the personalized training split, not a full reproduction of the original SPUR neutral-label pipeline.
> The rubrics are re-induced from our training users rather than borrowed from the SPUR paper.
> We will provide the induced rubrics in the appendix and revise the footnote to interpret this row as a boundary-oriented rubric baseline, with SAT/low-side predictions mapped to scores 4/3 for ordinal metrics.
> The trinary analysis also makes this limitation explicit: this binary adaptation cannot predict the score-1--2 DSAT class because its low-side output is mapped to score 3.
