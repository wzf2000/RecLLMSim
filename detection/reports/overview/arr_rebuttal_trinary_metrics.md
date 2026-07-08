# ARR Rebuttal: SAT/DSAT Terminology and Trinary Metrics

本文档整理针对 Reviewer fUbw 的 SAT/DSAT terminology 和 trinary SAT-Neutral-DSAT schema 的补充分析。
本次分析没有新增持久代码文件，也没有重新调用 LLM。
指标由现有 evaluator/baseline JSONL 输出一次性统计得到。

## 1. Motivation

Reviewer fUbw 指出，论文原稿中将 score $\leq 3$ 统一称为 DSAT 容易误导，因为数据标注说明中 score 3 是 neutral。
更准确的做法是将 score $\leq 3$ 作为 3/4 minimum-satisfaction boundary 的 low side，而不是全部称为 dissatisfied。
因此正文中的 `DSAT detection`、`DSAT rate` 等应改为 `low-side detection`、`low-side rate`，并说明 low side means scores at most 3.

同时，为了回应 reviewer 对三分类 schema 的建议，可以补充一个 trinary view:

- DSAT: scores 1--2.
- Neutral: score 3.
- SAT: scores 4--5.

## 2. Source Files

本次统计使用以下现有输出：

- `detection/outputs/personalized/Qwen_Qwen3-8B_test_none_calCDF.jsonl`
- `detection/outputs/personalized/history_baselines/*.jsonl`
- `detection/outputs/personalized/spur_direct_qwen3_8b_personalized_test.jsonl`
- `detection/outputs/personalized/generic_judge_qwen3_8b_*_test.jsonl`
- `detection/outputs/personalized/bert_supervised_*_test.jsonl`

Gold distribution on the evaluator verification test split:

| Class | Score range | Count |
|---|---:|---:|
| DSAT | 1--2 | 374 |
| Neutral | 3 | 733 |
| SAT | 4--5 | 5,367 |

The test set is highly skewed toward SAT, so weighted-F1 and accuracy can be high even for methods that rarely identify DSAT or Neutral.
Macro-F1 and class-wise F1 are more informative for the trinary analysis.

## 3. Trinary Results

| Method | Acc | Macro-F1 | Weighted-F1 | F1-DSAT | F1-Neutral | F1-SAT | Pred DSAT/Neu/SAT |
|---|---:|---:|---:|---:|---:|---:|---:|
| Supervised BERT | 0.6495 | 0.3232 | 0.6726 | 0.0103 | 0.1721 | 0.7872 | 15/1673/4786 |
| Supervised ordinal BERT | 0.8282 | 0.3020 | 0.7511 | 0.0000 | 0.0000 | 0.9060 | 0/5/6469 |
| Global mean | 0.8290 | 0.3022 | 0.7515 | 0.0000 | 0.0000 | 0.9065 | 0/0/6474 |
| User-history mean | 0.8092 | 0.3517 | 0.7632 | 0.0000 | 0.1559 | 0.8993 | 0/383/6091 |
| User-history median | 0.8177 | 0.3487 | 0.7629 | 0.0301 | 0.1134 | 0.9026 | 25/219/6230 |
| Nearest-history turn | 0.7561 | 0.4144 | 0.7508 | 0.1729 | 0.2045 | 0.8656 | 320/685/5469 |
| Nearest-history turn k=3 | 0.7722 | 0.3873 | 0.7544 | 0.0975 | 0.1869 | 0.8777 | 98/712/5664 |
| SPUR-style evaluator | 0.6894 | 0.3422 | 0.7033 | 0.0000 | 0.2063 | 0.8202 | 0/1526/4948 |
| Zero-shot judge | 0.8034 | 0.3770 | 0.7587 | 0.1612 | 0.0765 | 0.8935 | 110/261/6103 |
| Few-shot judge | 0.7484 | 0.3944 | 0.7402 | 0.1809 | 0.1414 | 0.8610 | 378/582/5514 |
| Task-rubric judge | 0.7266 | 0.3849 | 0.7273 | 0.1550 | 0.1542 | 0.8454 | 271/849/5354 |
| Prometheus-rubric judge | 0.7825 | 0.3946 | 0.7540 | 0.1937 | 0.1088 | 0.8812 | 163/443/5868 |
| Memory evaluator + CDF | 0.7754 | 0.4680 | 0.7706 | 0.2820 | 0.2456 | 0.8763 | 321/684/5469 |

## 4. Interpretation

The trinary view confirms that overall accuracy is not sufficient because the test set is dominated by SAT examples.
For example, global mean obtains accuracy 0.8290 and weighted-F1 0.7515 by predicting all turns as SAT, but its DSAT and Neutral F1 are both 0.

The memory evaluator obtains the best macro-F1 among the compared methods.
Its strongest gains come from the minority classes, with F1-DSAT 0.2820 and F1-Neutral 0.2456.
This supports the main claim that personalized memory helps identify lower-satisfaction and neutral turns rather than simply matching the dominant SAT class.

The SPUR-style baseline should be described carefully.
The implemented adaptation is a binary SAT/low-side baseline that maps its two outputs to scores 4 and 3 for the 1--5 table.
Under the trinary view, it cannot predict the DSAT class because it never outputs scores 1--2.
Therefore, it is useful as a boundary-oriented comparison, but it is not a full trinary satisfaction model.

## 5. Recommended Rebuttal Wording

> We agree that score 3 should not be described as dissatisfied.
> We will revise the terminology for the 3/4 boundary from DSAT detection to low-satisfaction/neutral-side detection, where the low side means scores at most 3.
> We also add a trinary SAT-Neutral-DSAT analysis with scores 1--2 as DSAT, score 3 as Neutral, and scores 4--5 as SAT.
> In this view, the memory evaluator achieves the best macro-F1 among the compared methods and improves the minority-class F1 scores for both DSAT and Neutral turns.
> We will also clarify that our SPUR-style row is a binary boundary-oriented adaptation rather than a full trinary predictor.
