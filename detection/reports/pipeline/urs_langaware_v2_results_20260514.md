# URS Language-Aware V2 Results

## Setup

Compared three Qwen3-8B URS memory-based predictor runs:

- `cal`: `outputs/urs/Qwen_Qwen3-8B_test_none_urs_v2_calibrated.jsonl`
- `old_lang`: `outputs/urs/Qwen_Qwen3-8B_test_none_urs_v2_calibrated_langaware.jsonl`
- `new_lang`: `outputs/urs/Qwen_Qwen3-8B_test_none_urs_v2_calibrated_langaware_v2.jsonl`

The new language-aware prompt keeps the English branch but adds a main-task-first policy:

- Do not downgrade below `4` solely because an answer is short or lightly truncated after giving the core answer.
- For factual short-answer / definition tasks, prioritize correctness and directness.
- Treat personalized memory as weak evidence and ignore unrelated intent preferences.

Evaluation output:

- `outputs/urs/qwen3_8b_urs_langaware_v2_comparison.json`

## Overall Metrics

| run | MAE | RMSE | Pearson | Spearman | QWK | Acc | F1-DSAT | F1-SAT | pred SAT / DSAT |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `cal` | 0.7260 | 1.0353 | 0.2358 | 0.2495 | 0.2073 | 0.6370 | 0.4592 | 0.7268 | 397 / 187 |
| `old_lang` | 0.7825 | 1.0799 | 0.1702 | 0.1843 | 0.1496 | 0.5942 | 0.4601 | 0.6749 | 350 / 234 |
| `new_lang` | 0.7243 | 1.0558 | 0.1835 | 0.1961 | 0.1572 | 0.6336 | 0.4022 | 0.7358 | 431 / 153 |

## Language Breakdown

### English

| run | MAE | RMSE | Pearson | Spearman | QWK | Acc | F1-DSAT | F1-SAT | pred dist |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `cal` | 0.7818 | 1.0787 | 0.1961 | 0.2394 | 0.1750 | 0.6545 | 0.4412 | 0.7500 | 1:1, 2:7, 3:67, 4:133, 5:12 |
| `old_lang` | 0.8909 | 1.1755 | -0.0021 | 0.0583 | -0.0017 | 0.5591 | 0.3899 | 0.6548 | 2:3, 3:95, 4:115, 5:7 |
| `new_lang` | 0.7636 | 1.0829 | 0.0734 | 0.0916 | 0.0627 | 0.6455 | 0.3036 | 0.7622 | 2:3, 3:48, 4:155, 5:14 |

### Chinese

| run | MAE | RMSE | Pearson | Spearman | QWK | Acc | F1-DSAT | F1-SAT | pred dist |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `cal` | 0.6923 | 1.0082 | 0.2779 | 0.2699 | 0.2347 | 0.6264 | 0.4688 | 0.7119 | 2:4, 3:108, 4:235, 5:17 |
| `old_lang` | 0.7170 | 1.0177 | 0.2823 | 0.2857 | 0.2515 | 0.6154 | 0.5000 | 0.6875 | 1:2, 2:3, 3:131, 4:209, 5:19 |
| `new_lang` | 0.7005 | 1.0391 | 0.2375 | 0.2497 | 0.2024 | 0.6264 | 0.4472 | 0.7178 | 1:1, 2:4, 3:97, 4:242, 5:20 |

Chinese prompt content did not intentionally change in V2; Chinese metric differences should be treated as rerun stochasticity unless reproduced with deterministic decoding.

## Delta Analysis

English `new_lang` vs `old_lang`:

- `50 / 220` English samples improved by absolute error.
- `22 / 220` worsened.
- `148 / 220` unchanged.
- Score deltas: `+1` for `63` samples, `0` for `148`, `-1` for `9`.

English `new_lang` vs `cal`:

- `34 / 220` improved.
- `33 / 220` worsened.
- `153 / 220` unchanged.

Chinese `new_lang` vs `old_lang`:

- `46 / 364` improved.
- `41 / 364` worsened.
- `277 / 364` unchanged.

## Case Checks

The intended failure modes were partly fixed:

- `en_142__creative__urs::en::00250.json__turn_0`: gold `5`, `old_lang=3`, `new_lang=4`.
  V2 no longer treats a late truncation as enough to make the answer unsatisfactory.
- `en_12__retrieval__urs::en::00020.json__turn_0`: gold `5`, `old_lang=3`, `new_lang=4`.
  V2 recognizes that a definition answer can be satisfactory without step-by-step guidance.
- `en_133__retrieval__urs::en::00241.json__turn_0`: gold `5`, `old_lang=3`, `new_lang=4`.
  V2 ignores unrelated generic memory preferences for a factual bus-route lookup.
- `en_74__retrieval__urs::en::00151.json__turn_0`: gold `2`, `old_lang=4`, `new_lang=3`.
  V2 fixes an over-satisfied prediction when the answer misses the critical manufacturing-origin requirement.

Remaining issue:

- `en_163__retrieval__urs::en::00297.json__turn_0`: gold `2`, `old_lang=4`, `new_lang=4`.
  The answer appears semantically relevant to the query, but the gold label is dissatisfied.
  This suggests the main-task-first policy can still over-credit sensitive or low-quality answers when the surface task is addressed.
- `en_56__retrieval__urs::en::00111.json__turn_0`: gold `1`, `old_lang=4`, `new_lang=3`.
  V2 improves but remains too lenient because it treats the model's "essay may not exist" response as partially useful rather than nearly unusable.

## Conclusion

The V2 adjustment successfully fixes the old English language-aware prompt's most obvious over-penalization of short/truncated high-score answers.
However, it overcorrects toward SAT predictions:

- English predicted DSAT decreases from `98` in `old_lang` to `51` in `new_lang`.
- Overall predicted SAT increases from `350` to `431`, above both `cal` (`397`) and gold SAT (`379`).
- English F1-SAT improves, but English F1-DSAT drops from `0.3899` to `0.3036`.

Therefore, `new_lang` is better than `old_lang` for MAE and high-score recovery, but it is not a better default than `cal`.
The best current single-run URS evaluator remains `urs_v2_calibrated`.

If continuing this line, the next revision should keep the short-answer/truncation robustness rules but add a stronger DSAT guard:

- If the user asks for a specific artifact, source, product, or exact constraint and the answer does not provide it, do not assign `4`.
- If the response only says the requested item may not exist and gives generic background, cap at `3`, and often `2`.
- For sensitive-topic prompts, require not only topical relevance but also direct usefulness and appropriateness before assigning `4`.
