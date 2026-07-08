# ARR Rebuttal: Table 2 User-Level Bootstrap Confidence Intervals

This report computes user-level bootstrap confidence intervals for the evaluator verification metrics in Table 2.
Each bootstrap sample draws 90 test users with replacement and includes all target turns for each sampled user.
We use 1,000 bootstrap resamples with random seed 42.

The calculation was run in the `chat` conda environment with `scipy` and `sklearn`.
No persistent analysis code was added.

## Source Files

- `BERT scorer`: `detection/outputs/personalized/bert_supervised_dsat_w3_t06_sel_f1dsat_test.jsonl`
- `BERT ordinal scorer`: `detection/outputs/personalized/bert_supervised_ordinal_dsat_w3_t06_sel_f1dsat_test.jsonl`
- `Nearest-history score`: `detection/outputs/personalized/qwen3_8b_episodic_rag_nearest_k6_full.jsonl`
- `RAG prompted scorer`: `detection/outputs/personalized/qwen3_8b_episodic_rag_boundary_first_k4_full.jsonl`
- `SPUR-style evaluator`: `detection/outputs/personalized/spur_direct_qwen3_8b_personalized_test.jsonl`
- `Zero-shot judge`: `detection/outputs/personalized/generic_judge_qwen3_8b_zero_shot_test.jsonl`
- `Few-shot judge`: `detection/outputs/personalized/generic_judge_qwen3_8b_few_shot_global_test.jsonl`
- `Task-rubric judge`: `detection/outputs/personalized/generic_judge_qwen3_8b_task_rubric_test.jsonl`
- `Prometheus-rubric judge`: `detection/outputs/personalized/generic_judge_qwen3_8b_prometheus_rubric_test.jsonl`
- `User-memory evaluator`: `detection/outputs/personalized/Qwen_Qwen3-8B_test_none_calCDF.jsonl`

## Point Estimates and 95% CIs

| Method | Pearson | Spearman | QWK | Low-side F1 |
|---|---:|---:|---:|---:|
| BERT scorer | 0.0126 [-0.0604, 0.0869] | 0.0192 [-0.0567, 0.0932] | 0.0125 [-0.0599, 0.0857] | 0.2268 [0.1732, 0.2814] |
| BERT ordinal scorer | 0.0363 [-0.0733, 0.1444] | 0.0625 [-0.0497, 0.1682] | 0.0322 [-0.0634, 0.1258] | 0.1984 [0.0993, 0.2829] |
| Nearest-history score | 0.3281 [0.2603, 0.3878] | 0.3529 [0.2772, 0.4211] | 0.2992 [0.2350, 0.3557] | 0.2361 [0.1738, 0.2993] |
| RAG prompted scorer | 0.1360 [0.0686, 0.2016] | 0.0776 [0.0395, 0.1156] | 0.0282 [0.0046, 0.0635] | 0.0214 [0.0056, 0.0443] |
| SPUR-style evaluator | 0.1314 [0.0873, 0.1723] | 0.1095 [0.0667, 0.1511] | 0.0842 [0.0556, 0.1112] | 0.2955 [0.2551, 0.3314] |
| Zero-shot judge | 0.1865 [0.1412, 0.2257] | 0.1274 [0.0942, 0.1600] | 0.1753 [0.1333, 0.2122] | 0.1732 [0.1439, 0.2021] |
| Few-shot judge | 0.1924 [0.1432, 0.2420] | 0.1578 [0.1169, 0.2022] | 0.1867 [0.1390, 0.2330] | 0.2680 [0.2325, 0.3010] |
| Task-rubric judge | 0.1600 [0.1212, 0.2006] | 0.1210 [0.0899, 0.1542] | 0.1511 [0.1148, 0.1912] | 0.2559 [0.2224, 0.2897] |
| Prometheus-rubric judge | 0.2066 [0.1611, 0.2486] | 0.1435 [0.1092, 0.1794] | 0.2005 [0.1559, 0.2398] | 0.2207 [0.1862, 0.2528] |
| User-memory evaluator | 0.3601 [0.3003, 0.4103] | 0.3716 [0.3148, 0.4214] | 0.3595 [0.2996, 0.4094] | 0.3655 [0.3074, 0.4183] |

## Interpretation

The confidence intervals are user-level intervals, not turn-level intervals, so they account for dependence among turns from the same user.
The memory evaluator has the strongest point estimates on all four Table 2 metrics, but its correlation intervals overlap with the strongest retrieval-style baseline.
The low-side F1 interval is more clearly separated from most generic LLM judges, supporting the claim that personalized memory is most useful for detecting turns below the minimum satisfaction boundary.
These intervals should be used as uncertainty estimates for fixed-output evaluator runs; they do not measure human test-retest reliability or LLM decoding variability.

## Rebuttal-Ready Wording

> We add user-level bootstrap confidence intervals for Table 2, resampling users rather than turns to account for within-user dependence.
> The memory evaluator obtains Pearson 0.3601 [0.3003, 0.4103], Spearman 0.3716 [0.3148, 0.4214], QWK 0.3595 [0.2996, 0.4094], and low-side F1 0.3655 [0.3074, 0.4183].
> These intervals show that the evaluator is consistently stronger than generic LLM judges, while the correlation intervals overlap with the strongest retrieval-style control.
> We will therefore present the results as evidence that personalized memory improves low-side detection and ordinal agreement, while avoiding claims of a human-level reliability ceiling.
