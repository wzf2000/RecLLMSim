# Response Draft for Reviewer dEr8

本文档是给 Reviewer dEr8 的 rebuttal 草稿。
英文段落可直接进入 rebuttal，中文 TODO 表示还需要补实验或最终确认。

## Reviewer Position

dEr8 整体偏正面，阅读非常细致。
他们认可 personalized turn-level satisfaction 的问题设定、数据收集、comparative memory、reproducibility 和 scope。
主要 concerns 是：

- agreement score 缺少 human reliability reference；
- Table 2 缺少 simple history-only baselines；
- benchmark 使用 moderately verified judge，需要更谨慎解释；
- Table 1 的 Direct feedback 可能 overclaim。

## Response to W1: Agreement Scores Need a Reference Point

### Confirmed Result

Detailed numbers and interpretation are recorded in `detection/reports/overview/arr_rebuttal_table2_bootstrap_ci.md`.

For the memory evaluator, user-level bootstrap over 90 test users gives:

| Metric | Point estimate | 95% CI |
|---|---:|---:|
| Pearson | 0.3601 | [0.3003, 0.4103] |
| Spearman | 0.3716 | [0.3148, 0.4214] |
| QWK | 0.3595 | [0.2996, 0.4094] |
| Low-side F1 | 0.3655 | [0.3074, 0.4183] |

### Draft Response

Thank you for highlighting that the absolute agreement values need a reliability reference point.
The current dataset contains one original-user satisfaction label per assistant turn, so a direct test-retest ceiling is not available in the submitted version.
We will make this limitation explicit in the revision and avoid interpreting the evaluator's agreement as human-level reliability.

To better quantify uncertainty, we add user-level bootstrap confidence intervals for the meta-evaluation metrics.
The memory evaluator obtains Pearson 0.3601 [0.3003, 0.4103], Spearman 0.3716 [0.3148, 0.4214], QWK 0.3595 [0.2996, 0.4094], and low-side F1 0.3655 [0.3074, 0.4183].
These intervals are not a substitute for test-retest reliability, but they prevent turn-level overconfidence by resampling users rather than individual turns.
We have also prepared a stratified replay-pair validation protocol for counterfactual candidate responses.
If completed within the revision window, we will report human-evaluator agreement and inter-annotator agreement; otherwise, we will explicitly state that direct validation of counterfactual candidate responses remains an open limitation.

### TODO

- [x] 计算 Table 2 的 user-level bootstrap CI。
- [ ] 如果完成人类 replay validation，在这里加入结果。
- [ ] Limitations 中加入 unknown within-user/test-retest ceiling。

## Response to W2: Missing Simple History-Only Baselines

### Confirmed Result

We agree that history-only baselines are important controls.
They are already implemented in the codebase, and adding them clarifies how much signal comes from user rating-style priors.
Detailed numbers and interpretation are recorded in `detection/reports/overview/arr_rebuttal_evaluator_controls.md`.

| Baseline | Pearson | Spearman | QWK | Low-side F1 |
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

### Draft Response

We agree that the distribution-only history baselines are important controls and will add them in the revision.
These results show that user rating-style priors explain a large fraction of full-score correlation; in particular, user-history mean and median are strong correlation baselines.
However, these baselines do not read the target assistant response and are substantially weaker on low-satisfaction-side detection.
For example, user-history mean reaches Pearson 0.3560 but only 0.2255 low-side F1, while the memory evaluator reaches Pearson 0.3601 and improves low-side F1 to 0.3655.
Nearest-history retrieval is a stronger response-agnostic control on low-side F1, but the memory evaluator still improves QWK and low-side F1 over it.
This supports the need for a response-aware personalized evaluator rather than only matching each user's historical score distribution.

### Planned Paper Revision

- Add global mean/majority and user-history mean/median/CDF controls.
- Adjust claim from "best on all metrics against all baselines" to a more precise claim:
  - history-only controls are strong on score-scale correlation;
  - the proposed evaluator is strongest on QWK and low-side detection, and remains response-aware for counterfactual candidate scoring.

## Response to W3: Benchmark Judge Reliability and Calibration Fallback

### Confirmed Result

The submitted paper did not make the calibration coverage sufficiently clear.
For the revision, we can use the reference-based calibration protocol, which calibrates all 300 replay records with no identity fallback.

Benchmark sensitivity across scoring views:

| Model | Raw user macro | RefMS user macro | RefCDF user macro | RefCDF low-side rate |
|---|---:|---:|---:|---:|
| kimi-k2.6 | 4.8400 | 4.8535 | 4.8121 | 0.0133 |
| glm-5.1 | 4.8415 | 4.8273 | 4.8003 | 0.0267 |
| deepseek-v4-pro | 4.8024 | 4.8153 | 4.7911 | 0.0267 |
| gpt-5.5 | 4.7213 | 4.7578 | 4.6893 | 0.0567 |
| claude-opus-4-7 | 4.5553 | 4.6338 | 4.6361 | 0.0467 |
| gemini-3.1-pro-preview | 4.6310 | 4.6516 | 4.6283 | 0.0600 |
| minimax-m2.7 | 4.2057 | 4.3621 | 4.4256 | 0.1367 |

### Draft Response

We agree that the benchmark should be read cautiously because it uses an automatic judge.
In the revision, we will strengthen this scope statement and describe PersTurnBench as a screening benchmark.
We will also report raw and calibrated sensitivity analyses.
The exact ordering of close models changes slightly across scoring views, but the broad groups remain stable.
The raw vs. reference-CDF leaderboard has Kendall's tau 0.8095, reference mean-shift vs. reference-CDF has tau 0.9048, and reference-transfer-CDF vs. reference-CDF has tau 0.9048.

Regarding calibration fallback, we will revise the paper to use the reference-based calibration protocol.
This protocol uses the full original-response prediction set as reference context and calibrates all 300 replay records, so the reported benchmark scores do not mix calibrated records with identity-fallback records.

### TODO

- [ ] 在论文 Section 5 或 appendix 中加入 reference-based calibration coverage 描述。
- [ ] 明确 CI overlap 后不要过度解释 top group 内部细微差异。

## Response to Additional Comment: Table 1 Direct Feedback

### Draft Response

Thank you for noting the ambiguity in Table 1.
We will revise the comparison table to distinguish direct original-user labels used for evaluator verification from automatic replay scoring of counterfactual candidate responses.
PersTurnBench uses direct user labels to verify the evaluator, but the replayed candidate responses are scored by the frozen evaluator rather than by the original users.
This change will prevent the table from suggesting that every benchmarked model response receives new direct user feedback.

## Response to Typo / Contribution Wording

### Draft Response

We will reword the second contribution bullet to make the meta-evaluation claim clearer.
