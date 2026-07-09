# ARR Rebuttal: SPUR-Style Baseline Clarification

本文档整理 Reviewer fUbw 关于 SPUR-style baseline 的问题和可直接用于 rebuttal / revision 的澄清内容。
当前 induced rubric 已经存在于本地输出中。
本次补充先新增了一个不重新调用 LLM 的 SPUR 输出映射诊断版本，并进一步实现了完整的 `DSAT/NEUTRAL/SAT` 三分类 SPUR personalized pipeline。
结论是这部分可以直接处理：在 rebuttal 中澄清原已报告结果是基于本文训练集重新诱导 rubric 的 binary boundary-oriented SPUR-style adaptation，并在附录或 artifact 中给出 induced rubric。
如果时间允许，正式 rebuttal 应优先报告完整 3-level SPUR 结果，而不是仅报告输出映射诊断。

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

## 3. Full 3-Level SPUR Implementation

The personalized SPUR runner now supports `label_schema=trinary`.
This is the setting closest to the reviewer's interpretation of the original SPUR-style setup:

- `score 1--2 -> DSAT`
- `score 3 -> NEUTRAL`
- `score 4--5 -> SAT`

In this mode, Phase 1 extracts separate DSAT, NEUTRAL, and SAT rubric candidates from the personalized training split.
Phase 2 summarizes three rubric sets.
Phase 3 predicts one of `DSAT`, `NEUTRAL`, and `SAT` for each test turn.
The compatible JSONL maps predictions to `pred_score=2/3/4` so the existing 1--5 evaluator can still be used for auxiliary ordinal metrics.

Full run command:

```bash
cd /data/wangzhefan/RecLLMSim
conda activate chat
model='Qwen/Qwen3-8B' \
base_url='http://localhost:8001/v1' \
api_key='EMPTY' \
variant='direct' \
label_schema='trinary' \
score_mapping='trinary_24' \
max_extract_per_label=150 \
max_workers=4 \
output_dir='outputs/spur_personalized/qwen3_8b_trinary_direct' \
output_jsonl='outputs/personalized/spur_trinary_direct_qwen3_8b_personalized_test.jsonl' \
metrics_json='outputs/personalized/spur_trinary_direct_qwen3_8b_personalized_test_metrics.json' \
bash detection/scripts/run_personalized_spur.sh
```

Evaluation command:

```bash
result_file='outputs/personalized/spur_trinary_direct_qwen3_8b_personalized_test.jsonl' \
output_json='outputs/personalized/spur_trinary_direct_qwen3_8b_personalized_test_eval.json' \
bash detection/scripts/eval_personalized.sh
```

Current implementation checks:

- Python compile check passed for the updated SPUR modules.
- Trinary split check produced train labels: `SAT=1261`, `NEUTRAL=108`, `DSAT=44`.
- Trinary split check produced test labels: `SAT=5367`, `NEUTRAL=733`, `DSAT=374`.
- Cached binary SPUR smoke test passed with the original Phase-3 cache and reproduced the existing binary metrics (`Accuracy=0.7135`, `F1-DSAT=0.2955`).

The full LLM run has not been started in this session because the same local vLLM endpoint may already be occupied by the repeated-run stability experiment.

## 4. Diagnostic Trinary Mapping Control

To check whether the original `DSAT -> score 3` mapping disadvantages SPUR under the reviewer-suggested trinary schema, we added a second output-only mapping:

- `boundary_34`: `SAT -> score 4`, `DSAT -> score 3`.
- `trinary_24`: `SAT -> score 4`, `DSAT -> score 2`.

This change reuses the same induced rubrics and the same cached Phase-3 SPUR decisions.
It does not re-call the LLM or change the SPUR decision boundary; it only changes how the binary SPUR output is converted into the 1--5-compatible record format.

Command:

```bash
model='Qwen/Qwen3-8B' \
output_dir='outputs/spur_personalized/qwen3_8b_direct' \
output_jsonl='outputs/personalized/spur_direct_qwen3_8b_personalized_test_trinary24.jsonl' \
metrics_json='outputs/personalized/spur_direct_qwen3_8b_personalized_test_trinary24_metrics.json' \
skip_phase1=1 \
skip_phase2=1 \
score_mapping='trinary_24' \
conda run -n chat bash detection/scripts/run_personalized_spur.sh
```

Output files:

- `detection/outputs/personalized/spur_direct_qwen3_8b_personalized_test_trinary24.jsonl`
- `detection/outputs/personalized/spur_direct_qwen3_8b_personalized_test_trinary24_metrics.json`
- `detection/outputs/personalized/spur_direct_qwen3_8b_personalized_test_trinary24_eval.json`
- `detection/outputs/personalized/spur_trinary_mapping_comparison_metrics.json`

Under the trinary schema (`1--2=DSAT`, `3=Neutral`, `4--5=SAT`), the two SPUR mappings behave differently:

| SPUR mapping | Acc | Macro-F1 | Weighted-F1 | F1-DSAT | F1-Neutral | F1-SAT | Pred DSAT/Neu/SAT |
|---|---:|---:|---:|---:|---:|---:|---:|
| `boundary_34` | 0.6894 | 0.3422 | 0.7033 | 0.0000 | 0.2063 | 0.8202 | 0/1526/4948 |
| `trinary_24` | 0.6775 | 0.3281 | 0.6894 | 0.1642 | 0.0000 | 0.8202 | 1526/0/4948 |

The `trinary_24` mapping confirms that SPUR's low-side predictions do contain some severe-dissatisfaction signal, raising F1-DSAT from `0.0000` to `0.1642`.
However, because this SPUR adaptation is still binary, it cannot identify score-3 Neutral turns under this mapping.
This is why the full memory evaluator remains stronger in the trinary view: it can produce all three regions and obtains F1-DSAT `0.2820`, F1-Neutral `0.2456`, and Macro-F1 `0.4680`.

## 5. Induced Rubrics

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

## 6. Recommended Paper Revision

Main Table 2 footnote:

> The SPUR-style row is a binary boundary-oriented adaptation: rubrics are re-induced from the personalized training split, and SAT/low-side predictions are mapped to scores 4/3 for ordinal metrics.

Additional rebuttal / appendix wording for the new control:

> We additionally checked an alternative trinary-compatible output mapping for the same SPUR decisions, where SAT is mapped to score 4 and DSAT is mapped to score 2.
> This raises SPUR's trinary DSAT F1 but removes Neutral predictions, confirming that the limitation is the binary SPUR adaptation rather than the specific 3/4 score mapping.

Preferred rebuttal wording after full trinary SPUR finishes:

> We additionally ran a full three-level SPUR-style adaptation using the same personalized split, with scores 1--2, 3, and 4--5 used to induce DSAT, NEUTRAL, and SAT rubrics respectively.
> This directly addresses the neutral-label concern while keeping the rubric induction source controlled.

Appendix baseline details:

> We do not claim to reproduce the full neutral-label SPUR pipeline.
> Instead, we adapt SPUR-style rubric induction as a supervised boundary baseline under our personalized split.
> The induced rubrics are learned from the training users and are not borrowed from the SPUR paper.

## 7. Rebuttal-Ready Wording

> We agree that the SPUR-style row should be described more carefully.
> Our implementation is a binary SPUR-style rubric-induction adaptation over the personalized training split, not a full reproduction of the original SPUR neutral-label pipeline.
> The rubrics are re-induced from our training users rather than borrowed from the SPUR paper.
> We will provide the induced rubrics in the appendix and revise the footnote to interpret this row as a boundary-oriented rubric baseline, with SAT/low-side predictions mapped to scores 4/3 for ordinal metrics.
> To address the mapping concern directly, we also evaluated the same SPUR decisions with an alternative trinary-compatible mapping, SAT -> 4 and DSAT -> 2.
> This improves SPUR's DSAT F1 under the trinary view but leaves Neutral F1 at 0, so the main limitation is that this adaptation is binary rather than a full trinary or 1--5 evaluator.
