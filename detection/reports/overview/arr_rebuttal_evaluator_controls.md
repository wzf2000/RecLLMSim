# ARR Rebuttal: Evaluator Verification Controls

本文档整理 rebuttal 阶段可以直接使用的 evaluator verification controls。
这些结果主要回应 Reviewer dEr8 对 simple history-only baselines 的要求，以及 Reviewer fUbw 对 score distribution 和 user-facing assistant model distribution 的问题。

本次整理没有新增持久代码文件，也没有重新调用 LLM 生成或评分。
统计来自现有输出文件和 `data/` 目录中的原始匿名 conversation JSON。

## 1. Source Files

- 主 evaluator 输出：`detection/outputs/personalized/Qwen_Qwen3-8B_test_none.jsonl`
- History-only baseline 结果：`detection/outputs/personalized/history_baselines/history_baselines_comparison.json`
- History-only baseline 既有报告：`detection/reports/overview/history_baselines_results.md`
- 原始匿名 conversation 数据：`data/User_*/*/*.json`

## 2. Control 1: History-Only Baselines

Reviewer dEr8 指出 Table 2 缺少 majority / mean predictors，以及只匹配用户历史评分分布但不读取当前 response 的控制实验。
现有代码库已经实现了这些 baseline，因此可以直接作为 rebuttal 和修订论文的补充结果。

| Evaluator / control | Pearson | Spearman | QWK | Low-side F1 |
|---|---:|---:|---:|---:|
| global mean | N/A | N/A | 0.0000 | 0.0000 |
| global majority | N/A | N/A | 0.0000 | 0.0000 |
| task mean | N/A | N/A | 0.0000 | 0.0000 |
| task majority | 0.0258 | 0.0348 | 0.0165 | 0.0000 |
| user-history mean | 0.3560 | 0.3917 | 0.3105 | 0.2255 |
| user-history median | 0.3526 | 0.3972 | 0.3075 | 0.1643 |
| user-history majority | 0.2895 | 0.3573 | 0.2454 | 0.1286 |
| user-history CDF-hash | 0.2167 | 0.2579 | 0.2167 | 0.2752 |
| nearest-history turn | 0.2550 | 0.2891 | 0.2543 | 0.3106 |
| nearest-history turn k=3 | 0.3076 | 0.3406 | 0.2974 | 0.2963 |
| memory evaluator + CDF | 0.3601 | 0.3716 | 0.3595 | 0.3655 |

### Interpretation

History-only controls are strong on full-score correlation.
In particular, user-history mean reaches Pearson 0.3560, close to the memory evaluator's Pearson 0.3601, and user-history median reaches Spearman 0.3972, higher than the memory evaluator's Spearman 0.3716.
This means the revised paper should not claim that the evaluator is best on every metric once these controls are included.

The stronger and more defensible conclusion is narrower:
history-only controls show that user rating-style priors explain a large part of the 1--5 score correlation, but these controls do not read the target assistant response and cannot score counterfactual candidate responses.
The memory evaluator remains stronger on ordinal agreement measured by QWK and on low-side detection, improving low-side F1 from 0.2255 for user-history mean and 0.3106 for nearest-history turn to 0.3655.
This supports the need for a response-aware personalized evaluator, rather than only matching each user's historical score distribution.

### Rebuttal-ready Wording

> We agree that distribution-only history baselines are important controls and will add them in the revision.
> These controls show that user rating-style priors explain a large fraction of full-score correlation.
> However, they do not read the target assistant response and are substantially weaker on low-satisfaction/neutral-side detection.
> For example, user-history mean reaches Pearson 0.3560 but only 0.2255 low-side F1, while the memory evaluator reaches Pearson 0.3601 and improves low-side F1 to 0.3655.
> We will therefore revise the claim to state that the evaluator is competitive on score correlation and stronger on QWK and low-side detection, rather than claiming it is best on every metric after adding these controls.

## 3. Control 2: Score Distribution

Reviewer fUbw asked for the per-score distribution because the 1--5 scale is skewed toward satisfied turns.
The following table reports the target test turns used for evaluator verification and the source histories available to the evaluator in the cross-scenario protocol.

| Split / source | Score 1 | Score 2 | Score 3 | Score 4 | Score 5 | Total |
|---|---:|---:|---:|---:|---:|---:|
| Full labeled data | 139 | 279 | 843 | 3,102 | 3,697 | 8,060 |
| Test target turns | 123 | 251 | 733 | 2,449 | 2,918 | 6,474 |
| Source histories for test blocks | 367 | 746 | 2,188 | 7,310 | 8,707 | 19,318 |

For the test target turns, scores 4--5 account for 82.90% of examples, while scores 1--3 account for 17.10%.
The source-history row is counted at the evaluator-block level: the same original history turn can be counted multiple times when it is used as source history for different target scenarios of the same user.
This is the correct reporting view for evaluator input coverage, but the caption or text should explicitly state the duplication.

### Rebuttal-ready Wording

> We will add per-score distributions for both the target test turns and the source histories used by the evaluator.
> The target test split contains 6,474 assistant turns, with 123, 251, 733, 2,449, and 2,918 examples for scores 1--5, respectively.
> Source histories are reported at the evaluator-block level because the same user history can serve as evidence for multiple target scenarios.

## 4. Control 3: User-Facing Assistant Models

Reviewer fUbw asked whether the evaluator could benefit from same-model bias.
The user-facing assistants used during data collection were GPT-4 Turbo, Gemini 2.0 Pro experimental, Claude 3.7 Sonnet, and DeepSeek-V3.
Qwen3-8B was not used as a user-facing assistant in data collection.
Therefore, the main Qwen3-8B evaluator verification does not directly evaluate Qwen3-8B responses against Qwen3-8B-generated references.

| User-facing assistant model | Labeled assistant turns | Conversation sessions |
|---|---:|---:|
| gpt-4-turbo-preview | 2,561 | 526 |
| gemini-2.0-pro-exp-02-05 | 2,065 | 498 |
| claude-3-7-sonnet-20250219 | 2,039 | 465 |
| deepseek-v3 | 1,395 | 344 |

For the 90 users in the evaluator verification test split:

| User-facing assistant model | Labeled assistant turns | Conversation sessions |
|---|---:|---:|
| gpt-4-turbo-preview | 2,034 | 421 |
| claude-3-7-sonnet-20250219 | 1,662 | 376 |
| gemini-2.0-pro-exp-02-05 | 1,648 | 399 |
| deepseek-v3 | 1,130 | 274 |

One DeepSeek-V3 conversation file in the full data has no labeled assistant turn, but the session count above follows the conversation-file count used in the dataset statistics.

### Rebuttal-ready Wording

> The user-facing assistants used during data collection were GPT-4 Turbo, Gemini 2.0 Pro experimental, Claude 3.7 Sonnet, and DeepSeek-V3.
> Qwen3-8B was not used as a user-facing assistant in data collection, which reduces direct same-model bias in the evaluator verification stage.
> We will add these model-source statistics to the appendix.

## 5. Recommended Paper Revision

For Section 4, the safest revision is:

- Add history-only controls either to the main evaluator table or to a compact supplementary table referenced from the main text.
- Replace claims like "strongest across all metrics" with "strongest on QWK and low-side detection, and competitive on score correlation."
- Rename DSAT-related metrics when they use the threshold score <= 3, because score 3 is neutral in the annotation anchor.
- Add score distribution and user-facing assistant distribution to the data/statistics appendix.

The remaining evaluator-verification controls that are useful but not yet completed are:

- user-level bootstrap confidence intervals for Table 2;
- trinary SAT-Neutral-DSAT metrics;
- history amount robustness ablation;
- optional repeated-run stability on a small subset.
