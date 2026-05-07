# Turn Content Annotation Case Audit on U20

## Purpose

This report audits concrete examples from:

- `detection/outputs/personalized/turn_content_annotations_u20.jsonl`

The goal is to check whether the annotation distinction between substantive and
non-substantive assistant turns is reliable.

This is a follow-up to:

- `detection/reports/analysis/turn_content_annotation_results_u20.md`

## Key Finding

The case audit changes the practical recommendation:

> For deciding whether a turn contains substantive content, use
> `has_substantive_content`, not a strict rule based only on `content_type`.

Reason:

- `content_type=other` often means "substantive but off-task / out-of-scope",
  not "non-substantive".
- Some `clarifying_question` turns contain enough interim content or structured
  requirement elicitation that the annotation model marks
  `has_substantive_content=true`.
- Some `ack_or_meta` turns include small tips or explanations, but are still
  mostly acknowledgement / encouragement.

Therefore:

```text
substantive split: use has_substantive_content
semantic subtype analysis: use content_type + task_relevance
```

## Case-Level Assessment

### Clearly reasonable non-substantive labels

These examples are correctly labeled as non-substantive.

#### `User_101__礼物准备__2.json__turn_3`

- user: `300预算`
- assistant: `思考中...`
- label: `ack_or_meta`, `has_substantive_content=false`, `task_relevance=low`
- gold: `1`, reason: `不可用`

Judgment: correct. The assistant gives no usable recommendation or planning
content.

#### `User_109__礼物准备__5.json__turn_4`

- user: `继续查询`
- assistant: says it is still querying and lists sources it is checking
- label: `ack_or_meta`, `has_substantive_content=false`, `task_relevance=low`
- gold: `1`, reason: `不可用`

Judgment: correct. It describes process but does not provide the requested
result.

#### `User_109__菜谱规划__0.json__turn_2`

- user: `可以 这两个菜我都喜欢吃`
- assistant: `太好了！喜欢就好...祝你周末愉快`
- label: `ack_or_meta`, `has_substantive_content=false`, `task_relevance=none`
- gold: `5`

Judgment: correct as a content label. The high gold score shows an important
point: non-substantive closing turns can still be satisfactory in context.
Therefore non-substantive should not automatically imply dissatisfied.

#### `User_2__礼物准备__2.json__turn_4`

- user: asks for channels to get book-selection inspiration
- assistant: `阅读爱好`
- label: `other`, `has_substantive_content=false`, `task_relevance=low`
- gold: `4`

Judgment: content label is reasonable because the reply is too short to be a
real answer. The gold label being high suggests either the turn was judged in a
wider dialogue context or the dataset contains some tolerance/noise for short
follow-up turns.

### Reasonable substantive labels that are off-task

These examples show why `content_type=other` should not be treated as
non-substantive.

#### `User_101__旅行规划__1.json__turn_3`

- user: asks how to survive an armed threat
- assistant: gives a long safety guide
- label: `other`, `has_substantive_content=true`, `task_relevance=none`
- gold: `5`

Judgment: the label is reasonable. The answer is substantive, but it is not
travel-planning content under the original task. It should be counted as
substantive content but low/none task relevance.

#### `User_114__旅行规划__4.json__turn_1`

- user: `河南胡辣汤怎么样`
- assistant: gives a detailed introduction to Henan spicy soup
- label: `other`, `has_substantive_content=true`, `task_relevance=low`
- gold: `5`

Judgment: reasonable. The content is substantive and useful for a food-related
side question, but weakly aligned with the original travel-planning task.

#### `User_114__旅行规划__4.json__turn_2`

- user: asks how Henan locals call Hu La Tang
- assistant: gives a detailed local naming guide
- label: `other`, `has_substantive_content=true`, `task_relevance=low`
- gold: `5`

Judgment: reasonable. Again, substantive but not central to the original task.

#### `User_19__礼物准备__4.json__turn_5`

- user: asks why Philips electric toothbrushes are expensive and what is good
  about them
- assistant: gives a detailed product-quality explanation
- label: `other`, `has_substantive_content=true`, `task_relevance=low`
- gold: `5`

Judgment: reasonable. This is substantive product analysis. Although the
annotation rationale says it is not directly gift-preparation, it should not be
filtered as non-content.

### Clarifying-question cases are nuanced

Some clarifying questions are correctly marked as content-bearing because they
include structured requirement elicitation or interim guidance.

#### `User_109__菜谱规划__0.json__turn_0`

- assistant asks about cooking level, taste, ingredients, equipment, and desired
  complexity
- label: `clarifying_question`, `has_substantive_content=true`,
  `task_relevance=high`
- gold: `5`

Judgment: reasonable. It is primarily a clarifying question, but it is
task-relevant and structured enough to be evaluable. This should not be removed
from the benchmark.

#### `User_109__礼物准备__0.json__turn_0`

- assistant asks about recipient age, hobbies, needs, gift type, and DIY ability
- label: `clarifying_question`, `has_substantive_content=true`,
  `task_relevance=high`
- gold: `4`

Judgment: reasonable. The response does not provide final gift options, but it
does perform useful requirement elicitation.

#### `User_109__礼物准备__4.json__turn_0`

- assistant asks many preference questions for a wedding gift
- label: `clarifying_question`, `has_substantive_content=true`,
  `task_relevance=high`
- gold: `3`, reason: `不够细致`

Judgment: reasonable. The label captures that the response is evaluable and
task-relevant, while the gold score captures that the user found it insufficient.
This distinction is important: substantive/clarifying does not determine
satisfaction.

#### `User_113__技能学习规划__3.json__turn_0`

- user asks for authoritative beginner online courses for "this sport"
- assistant asks which sport, but also gives general resource-screening
  principles and platform types
- label: `clarifying_question`, `has_substantive_content=true`,
  `task_relevance=high`
- gold: `5`

Judgment: reasonable. It is a mixed clarifying answer in practice, though the
content type could arguably be `mixed_answer` instead of `clarifying_question`.

### Borderline acknowledgement/meta cases

These cases show some noise in the boolean field.

#### `User_105__技能学习规划__2.json__turn_4`

- user thanks the assistant
- assistant gives encouragement and several small practice tips
- label: `ack_or_meta`, `has_substantive_content=true`, `task_relevance=low`
- gold: `4`

Judgment: borderline. Because it contains a few actionable tips, `true` is
defensible, but for strict filtering it behaves more like acknowledgement. This
is not a serious issue because such cases are rare.

#### `User_107__技能学习规划__3.json__turn_4`

- user says the plan sounds suitable for a beginner
- assistant gives encouragement plus generic practice advice
- label: `ack_or_meta`, `has_substantive_content=true`, `task_relevance=low`
- gold: `4`

Judgment: borderline. Similar to the previous case. The content is light but not
empty.

#### `User_114__技能学习规划__0.json__turn_2`

- user points out a language glitch
- assistant corrects the sentence and offers better alternatives
- label: `ack_or_meta`, `has_substantive_content=true`, `task_relevance=low`
- gold: `2`

Judgment: the boolean `true` is reasonable because the assistant provides a
specific correction. `ack_or_meta` is imprecise; `other` may be better. This is
substantive but outside the main skill-planning task.

#### `User_114__技能学习规划__0.json__turn_3`

- user asks why a Bengali word appeared
- assistant explains possible causes and apologizes
- label: `ack_or_meta`, `has_substantive_content=true`, `task_relevance=none`
- gold: `2`

Judgment: substantive explanation but not task content. It should not be
considered empty, but it is out-of-task meta discussion.

## What This Means for the Annotation Scheme

The current annotation scheme is useful, but the fields should be interpreted as
separate axes:

1. `has_substantive_content`
   - Best field for deciding whether the assistant said something evaluable.
   - Should be used for content vs non-content filtering.

2. `content_type`
   - Best field for qualitative subtype analysis.
   - `other` does not mean "non-content"; it often means "substantive but
     outside the main task taxonomy".

3. `task_relevance`
   - Best field for deciding whether the content is aligned with the original
     task.
   - This should be analyzed separately from substantive content.

Recommended derived groups:

```text
evaluable_content = has_substantive_content == true
empty_or_process_only = has_substantive_content == false
task_aligned_content = has_substantive_content == true and task_relevance in {medium, high}
off_task_content = has_substantive_content == true and task_relevance in {none, low}
pure_clarification = content_type == clarifying_question
ack_meta = content_type == ack_or_meta
```

## Revision to Previous Recommendation

The earlier recommendation in
`detection/reports/analysis/turn_content_annotation_results_u20.md` suggested:

```text
strict_content_like = content_type in {substantive_answer, mixed_answer}
```

After case audit, this is too aggressive if the goal is only to distinguish
substantive from non-substantive. It incorrectly treats off-task but substantive
answers as non-content.

Updated recommendation:

```text
content filtering: use has_substantive_content
task-alignment filtering: additionally require task_relevance in {medium, high}
```

## Bottom Line

The annotation quality is broadly reasonable at the concrete-case level.

- Clearly empty/process-only replies are usually labeled correctly.
- Clarifying turns are handled in a nuanced way: many are marked substantive
  because they perform useful requirement elicitation.
- Off-task side answers are correctly marked as substantive but low/none
  relevance.
- The main weakness is that `content_type` alone is not a reliable proxy for
  substantive content.

For future analysis, do not use `content_type != substantive_answer/mixed_answer`
as a non-content filter. Use the boolean field, and separately analyze task
relevance.
