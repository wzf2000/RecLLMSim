# Response Draft for Reviewer fUbw

本文档是给 Reviewer fUbw 的 rebuttal 草稿。
这是最关键的 borderline reviewer，因此回复需要承认合理问题、澄清概念，并尽量补实验。
英文段落可直接进入 rebuttal，中文 TODO 表示还需要补实验或最终确认。

## Reviewer Position

fUbw 认可工作主题、数据潜在价值、baseline 覆盖和 ablation，但对 soundness 给出 borderline。
核心担忧是：

- 1--5 rating schema 和 1/2、4/5 边界不够清楚；
- score 3 是 neutral，但论文把 score <= 3 作为 DSAT terminology，容易误导；
- SPUR-style baseline 的 3/4 映射不够公平；
- 缺少 user history amount ablation；
- domain narrow 和 mixed satisfaction signals 可能限制适用性。

## Response to W1: Rating Boundaries and Trinary Schema

### Confirmed Result

Detailed numbers and interpretation are recorded in `detection/reports/overview/arr_rebuttal_trinary_metrics.md`.

| Method | Macro-F1 | Weighted-F1 | F1-DSAT | F1-Neutral | F1-SAT |
|---|---:|---:|---:|---:|---:|
| Global mean | 0.3022 | 0.7515 | 0.0000 | 0.0000 | 0.9065 |
| User-history mean | 0.3517 | 0.7632 | 0.0000 | 0.1559 | 0.8993 |
| Nearest-history turn | 0.4144 | 0.7508 | 0.1729 | 0.2045 | 0.8656 |
| SPUR-style evaluator | 0.3422 | 0.7033 | 0.0000 | 0.2063 | 0.8202 |
| Zero-shot judge | 0.3770 | 0.7587 | 0.1612 | 0.0765 | 0.8935 |
| Few-shot judge | 0.3944 | 0.7402 | 0.1809 | 0.1414 | 0.8610 |
| Prometheus-rubric judge | 0.3946 | 0.7540 | 0.1937 | 0.1088 | 0.8812 |
| Memory evaluator + CDF | 0.4680 | 0.7706 | 0.2820 | 0.2456 | 0.8763 |

### Draft Response

Thank you for pointing out that the 1--5 rating scale and its relation to a trinary SAT-Neutral-DSAT schema should be clearer.
In our data collection, users were given explanations and examples for all five satisfaction levels before rating assistant turns, so the intended score boundaries were anchored during annotation rather than left implicit.
We chose a five-level scale because graded satisfaction feedback is a common way to collect user utility judgments in search and recommendation-style evaluation settings, and it preserves ordinal distinctions beyond a coarse three-way label.
This design is not perfect and does not eliminate user-level inconsistency, but it provides more information about minimum satisfaction, excellence, and severe failures than a trinary-only label.
However, we agree that a trinary view is important for interpreting the results.
We therefore add a trinary analysis that maps scores 1--2 to DSAT, score 3 to Neutral, and scores 4--5 to SAT.
The memory evaluator achieves the best macro-F1 among the compared methods, with the largest gains on the minority DSAT and Neutral classes.

### TODO

- [x] 补 trinary metric table。
- [x] 指标覆盖 macro-F1、weighted-F1、class-wise F1。
- [x] 覆盖 main evaluator、history-only baseline、generic LLM judge、SPUR-style baseline。

## Response to W2: SPUR-Style Mapping

### Confirmed Result

Detailed clarification and the induced rubrics are recorded in `detection/reports/overview/arr_rebuttal_spur_clarification.md`.
The current SPUR-style implementation re-induces 10 SAT rubrics and 10 low-side rubrics from the personalized training split.
The output file is `detection/outputs/spur_personalized/qwen3_8b_direct/phase2_rubrics_k10.json`.
The training split used for induction contains 22 users, 85 user--scenario blocks, and 1,413 target turns.
The test split contains 90 users, 356 user--scenario blocks, and 6,474 target turns.

### Draft Response

We agree that the current SPUR-style row should be described more carefully.
Our implementation is a binary SPUR-style rubric-induction adaptation over the personalized training split, not a full reproduction of the original SPUR neutral-label pipeline.
The induced rubric is learned from the training split of our data and is not directly borrowed from the SPUR paper.
We will revise the footnote, provide the induced rubric, and interpret this row as a boundary/rubric baseline rather than a full 1--5 or trinary predictor.
The trinary analysis makes this limitation explicit: the binary SPUR-style adaptation maps its two outputs to scores 4 and 3, so it cannot predict the 1--2 DSAT class.

### TODO

- [x] 找到当前 SPUR induced rubric 的输出位置或重新导出。
- [x] Appendix 中放 induced rubric 或说明 artifact path。
- [x] 修改 Table 2 footnote，不再暗示原 SPUR pipeline 无 neutral label。

## Response to W3: SAT/DSAT Terminology and Score 3

### Draft Response

We agree that using "DSAT" for all scores at most 3 is imprecise because score 3 is defined as neutral in our annotation anchor.
We will revise the terminology to "low-satisfaction/neutral-side" for the 3/4 boundary and explicitly state that score 3 is neutral but falls below the minimum satisfaction boundary.
Continued interaction is not treated as satisfaction by default; each assistant turn is rated by the user after the conversation.
This terminology revision is paired with the trinary results above, where score 3 is evaluated as its own Neutral class rather than being merged with DSAT.

### Planned Paper Revision

- Rename `DSAT detection` to `low-side detection` or `low-satisfaction/neutral-side detection`.
- Rename `DSAT rate` to `low-side rate` when it means score <= 3.
- Keep `SAT rate` as score >= 4.
- Clarify score 3 in the annotation instructions and metrics section.

## Response to W4: User History Amount Robustness

### Draft Response

We agree that robustness under sparse user histories is important.
We will add a history-budget ablation that constructs user memory from a limited number of source conversations while keeping the same target turns.
This analysis will show how evaluator performance changes as historical evidence becomes sparse.

### TODO

- [ ] 运行 K-history ablation。
- [ ] 推荐 K：1、2、4、all source conversations。
- [ ] 如果 full split 成本高，先使用 20-user subset。
- [ ] 报 Pearson、QWK、low-side F1、memory parse/failure rate。

## Response to W5: Narrow Domain and Mixed Satisfaction Signals

### Draft Response

We agree that the current data are limited to planning-oriented Chinese conversations and that a scalar satisfaction score cannot capture all mixed reactions within a turn.
Our current contribution is to move from conversation-level satisfaction to turn-level personalized satisfaction, which already localizes user feedback to the assistant turn that triggered the judgment.
We agree that even turn-level labels can still collapse multiple factors within the same response.
The dissatisfaction reason records only the primary low-satisfaction category, so finer factor-level or multi-label satisfaction modeling is a natural next step.
We will clarify this scope and add aspect-level satisfaction modeling as future work.

## Response to Q1: Single Run or Summary Statistics

### Confirmed Result

Detailed numbers and interpretation are recorded in `detection/reports/overview/arr_rebuttal_table2_bootstrap_ci.md`.

The main Table 2 values are point estimates from fixed-output evaluator runs on a fixed test split.
We add user-level bootstrap intervals by resampling users rather than turns.
For the memory evaluator, the intervals are Pearson 0.3601 [0.3003, 0.4103], Spearman 0.3716 [0.3148, 0.4214], QWK 0.3595 [0.2996, 0.4094], and low-side F1 0.3655 [0.3074, 0.4183].

### Draft Response

The main Table 2 results are computed on a fixed test split from one main run of each evaluator.
In the revision, we add user-level bootstrap confidence intervals to better quantify uncertainty across users.
We will also clarify the decoding settings and note that prompt-based evaluators may still have decoding nondeterminism.

### TODO

- [x] 补 user-level bootstrap CI。
- [ ] 如果有时间，补 small-subset repeated-run stability。

## Response to Q2: Per-Score Distribution

### Confirmed Result

Detailed numbers and interpretation are recorded in `detection/reports/overview/arr_rebuttal_evaluator_controls.md`.

| Split / source | Score 1 | Score 2 | Score 3 | Score 4 | Score 5 | Total |
|---|---:|---:|---:|---:|---:|---:|
| Test target turns | 123 | 251 | 733 | 2449 | 2918 | 6474 |
| Source histories used by test blocks, block-duplicated | 367 | 746 | 2188 | 7310 | 8707 | 19318 |

### Draft Response

We will add the per-score distributions for both the target test turns and the source histories used by the evaluator.
The target test split contains 6,474 assistant turns with the distribution shown above.
The source-history distribution is reported at the evaluator-block level, so the same original history turn can appear in multiple user--target-scenario blocks.

## Response to Q3: User-Facing Assistant Model and Same-Model Bias

### Confirmed Result

Detailed numbers and interpretation are recorded in `detection/reports/overview/arr_rebuttal_evaluator_controls.md`.

The user-facing assistant models used during data collection were:

| User-facing assistant model | All labeled assistant turns | Conversation sessions |
|---|---:|---:|
| gpt-4-turbo-preview | 2561 | 526 |
| gemini-2.0-pro-exp-02-05 | 2065 | 498 |
| claude-3-7-sonnet-20250219 | 2039 | 465 |
| deepseek-v3 | 1395 | 344 |

For test users:

| User-facing assistant model | Labeled assistant turns | Conversation sessions |
|---|---:|---:|
| gpt-4-turbo-preview | 2034 | 421 |
| claude-3-7-sonnet-20250219 | 1662 | 376 |
| gemini-2.0-pro-exp-02-05 | 1648 | 399 |
| deepseek-v3 | 1130 | 274 |

### Draft Response

The user-facing assistants used during data collection were GPT-4 Turbo, Gemini 2.0 Pro experimental, Claude 3.7 Sonnet, and DeepSeek-V3.
Qwen3-8B was not used as a user-facing assistant in data collection, which reduces direct same-model bias in the evaluator verification stage.
We will add these statistics to the appendix.

## Response to Q4: SPUR Rubric Source

### Confirmed Result

The induced rubric is available at `detection/outputs/spur_personalized/qwen3_8b_direct/phase2_rubrics_k10.json` and is summarized in `detection/reports/overview/arr_rebuttal_spur_clarification.md`.
It contains 10 SAT rubrics and 10 low-side rubrics induced from our personalized training split.

### Draft Response

The SPUR-style rubric used in our baseline was re-induced from the personalized training split.
It was not directly borrowed from the SPUR paper.
We will clarify this in the baseline description and provide the induced rubric in the appendix and released artifact.
