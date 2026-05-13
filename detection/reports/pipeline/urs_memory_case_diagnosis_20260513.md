# URS Memory Case Diagnosis

## Context

Recent URS Qwen3-8B predictor comparison showed:

- `mem_calibrated` has the best global metrics.
- `no_mem_v2` has better user-aware metrics.
- Memory-based runs still show negative user-aware correlations.

This note inspects concrete cases to understand why the current URS memory is
not a reliable personalization signal.

## Finding 1: Thin History Produces Overconfident User Profiles

Example:

- `en_163__retrieval__urs::en::00293.json__turn_0`
- Gold: `4`
- `mem_calibrated`: `2`
- `no_mem_calibrated`: `4`

The target session asks:

> What is demand pull inflation?

The assistant gives a relevant definition. The no-memory model correctly rates
it as satisfactory. The memory model rates it as `2` because the user's memory
contains only one historical 3-point leisure session about travel/family
activities. The memory summary incorrectly treats this as stable user-specific
requirements:

- detailed tourist attractions
- transportation suggestions
- family-friendly activity recommendations

These requirements are irrelevant to an economics definition question. This is
a cross-intent transfer failure caused by very thin history.

## Finding 2: Memory Can Override The Actual Session Content

Example:

- `zh_569__retrieval__urs::zh::00395.json__turn_0`
- Gold: `1`
- `mem_calibrated`: `4`
- `no_mem_calibrated`: `3`

The user asks for today's weather. The assistant says it cannot provide
real-time weather and suggests using a weather app or website.

The memory says the user is generally lenient and often gives `4-5`, with an
average satisfaction score of `4.33`. The memory-based predictor accepts the
fallback suggestion as satisfying the core need and predicts `4`, despite the
gold label being very dissatisfied.

This shows that memory-level prior can dominate concrete failure evidence in
the target session.

## Finding 3: All-Positive Histories Inflate Scores

Example:

- `en_26__leisure__urs::en::00049.json__turn_0`
- Gold: `2`
- `mem_v2`: `5`
- `no_mem_v2`: `2`

The user asks for a hedgehog joke. The assistant gives a weak joke:

> Why did the hedgehog cross the road? To see his flat mate!

The no-memory model identifies weak humor and predicts `2`. The memory model
predicts `5` because the user's available history contains only 5-point
sessions. The memory summary states that the user is lenient and tends to give
full marks.

This is a classic positive-history overfitting failure. With no negative
counterexamples, the memory prior becomes too strong.

## Finding 4: Memory Sometimes Helps By Capturing User Leniency

Example:

- `zh_515__creative__urs::zh::00224.json__turn_0`
- Gold: `5`
- `mem_calibrated`: `5`
- `no_mem_calibrated`: `2`

The conversation shifts between interview tips, a climate-related passage, and
a vocabulary explanation. The no-memory model penalizes the session as
off-topic, while the memory model observes that this user has all-positive
history and tends to reward concise, accurate answers. It predicts `5`, matching
gold.

This indicates that memory is not useless. It can help when user-level scoring
style is real and the target session would otherwise look odd from a generic
quality perspective.

## Finding 5: Some Gold Labels Are Hard For A Generic Judge To Infer

Example:

- `zh_33__professional__urs::zh::00508.json__turn_0`
- Gold: `5`
- `mem_calibrated`: `2`
- `no_mem_calibrated`: `4`

The user asks a multiple-choice statistics question. The assistant answers
`-2.00`. The gold label is `5`.

The memory model treats the answer as incorrect and gives `2`; the no-memory
model assumes it is correct and gives `4`. This case is difficult because the
judge needs domain correctness. If the judge's own calculation differs from the
dataset's expected answer, memory cannot fix the issue.

## Main Diagnosis

The current URS memory is not reliable because:

1. It is a summary memory, not retrieval-grounded evidence.
2. Many user-intent blocks have very thin history.
3. Summary fields often overgeneralize intent-specific needs into global user
   requirements.
4. The predictor gives the memory prior too much authority over target-session
   content.
5. All-positive or all-neutral histories create extreme priors.
6. URS session-level labels can reflect user idiosyncrasy or hidden context
   that is hard to infer from text alone.

## Recommendations

For URS benchmark judging:

- Use `mem_calibrated` as the main global-score judge only with caveats.
- Keep `no_mem_v2` or `no_mem_calibrated` as a robustness check.
- Report that current URS memory does not yet demonstrate reliable
  personalization.

For improving memory:

- Add memory confidence fields based on history count and score diversity.
- Downweight memory when history has fewer than 3 sessions or only one score
  bucket.
- Separate intent-specific observations from global user requirements more
  strictly.
- Prefer retrieval evidence over compressed summary when scoring individual
  sessions.
- Add prompt instructions that target-session evidence overrides memory priors
  when there is a direct contradiction.
