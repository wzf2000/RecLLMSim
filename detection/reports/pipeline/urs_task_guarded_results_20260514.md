# URS Task-Guarded Prompt Results

## Setup

Evaluated two new Qwen3-8B URS runs:

- Memory: `outputs/urs/Qwen_Qwen3-8B_test_none_urs_v2_calibrated_task_guarded.jsonl`
- No memory: `outputs/urs/Qwen_Qwen3-8B_test_no_memory_urs_v2_calibrated_task_guarded.jsonl`

Compared against:

- `urs_v2_calibrated`
- `urs_v2_memory_guarded`
- `urs_v2_calibrated_langaware_v2`
- no-memory `urs_v2_calibrated`

Evaluation output:

- `outputs/urs/qwen3_8b_urs_task_guarded_comparison.json`

## Overall Metrics

| run | MAE | RMSE | Pearson | Spearman | QWK | Acc | F1-DSAT | F1-SAT | FalseSAT | FalseDSAT |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `cal` | 0.7260 | 1.0353 | 0.2358 | 0.2495 | 0.2073 | 0.6370 | 0.4592 | 0.7268 | 0.5610 | 0.2559 |
| `mem_guard` | 0.7483 | 1.0558 | 0.1851 | 0.1928 | 0.1590 | 0.6164 | 0.4563 | 0.7037 | 0.5415 | 0.2982 |
| `lang_v2` | 0.7243 | 1.0558 | 0.1835 | 0.1961 | 0.1572 | 0.6336 | 0.4022 | 0.7358 | 0.6488 | 0.2137 |
| `task_guard` | 0.6952 | 1.0119 | 0.2760 | 0.2883 | 0.2424 | 0.6627 | 0.5207 | 0.7398 | 0.4780 | 0.2612 |
| `no_cal` | 0.7928 | 1.0925 | 0.1621 | 0.1527 | 0.1411 | 0.6301 | 0.3415 | 0.7429 | 0.7268 | 0.1768 |
| `no_task_guard` | 0.8921 | 1.1842 | 0.0915 | 0.0848 | 0.0837 | 0.6216 | 0.3912 | 0.7255 | 0.6537 | 0.2296 |

## Main Findings

`task_guard` is the strongest single URS run so far.
It improves all main global metrics over `cal`:

- MAE: `0.7260 -> 0.6952`
- Pearson: `0.2358 -> 0.2760`
- Spearman: `0.2495 -> 0.2883`
- QWK: `0.2073 -> 0.2424`

It also improves the 3/4 boundary:

- Accuracy: `0.6370 -> 0.6627`
- F1-DSAT: `0.4592 -> 0.5207`
- F1-SAT: `0.7268 -> 0.7398`
- FalseSAT: `0.5610 -> 0.4780`

The no-memory task-guarded variant is much worse than memory `task_guard`.
This indicates that the task guard is useful when paired with memory/context, but as a no-memory prompt it pushes the model into noisier and less calibrated judgments.
Do not use `no_task_guard` as the default URS evaluator.

## Language Breakdown

| run | lang | MAE | Pearson | Spearman | QWK | Acc | F1-DSAT | F1-SAT |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `cal` | en | 0.7818 | 0.1961 | 0.2394 | 0.1750 | 0.6545 | 0.4412 | 0.7500 |
| `task_guard` | en | 0.7227 | 0.2505 | 0.2733 | 0.2117 | 0.6955 | 0.5109 | 0.7789 |
| `cal` | zh | 0.6923 | 0.2779 | 0.2699 | 0.2347 | 0.6264 | 0.4688 | 0.7119 |
| `task_guard` | zh | 0.6786 | 0.2956 | 0.2971 | 0.2612 | 0.6429 | 0.5255 | 0.7137 |

Unlike the language-aware branch, task guarding improves both English and Chinese.
This supports the previous hypothesis that URS needs task/dataset calibration more than language-specific prompt branches.

## Task Breakdown

Compared with `cal`, `task_guard` improves most important intents:

| task | cal MAE | task_guard MAE | cal F1-DSAT | task_guard F1-DSAT | note |
|---|---:|---:|---:|---:|---|
| `advice` | 0.7396 | 0.6250 | 0.4800 | 0.5789 | clear improvement |
| `creative` | 0.7500 | 0.7031 | 0.3256 | 0.4091 | improved DSAT |
| `leisure` | 0.7708 | 0.7083 | 0.4615 | 0.4848 | modest improvement |
| `professional` | 0.7949 | 0.7778 | 0.4156 | 0.5417 | large boundary improvement |
| `retrieval` | 0.7030 | 0.6606 | 0.4950 | 0.5714 | strongest target-task gain |
| `text` | 0.6279 | 0.7093 | 0.5385 | 0.4561 | regression |
| `other` | 0.6250 | 0.7500 | 0.4000 | 0.2857 | small n=8, unstable |

The main regression is `text`.
The text-task guard may be too strict about "directly usable target text" and may over-penalize responses that URS gold treats as satisfactory.

## Delta Analysis vs Calibrated

Relative to `cal`:

- `81 / 584` samples improved by absolute error.
- `63 / 584` worsened.
- `440 / 584` unchanged.
- Score deltas: `0` for `439`, `-1` for `77`, `+1` for `63`, `+2` for `4`, `-2` for `1`.

Boundary behavior:

- Fixed `32` cases where `cal` predicted SAT for gold DSAT and `task_guard` predicted DSAT.
- Introduced `32` cases where `cal` predicted SAT for gold SAT and `task_guard` predicted DSAT.

The net boundary gain is therefore not just from predicting more DSAT.
It comes from better placement across the score scale and fewer false-SAT cases overall.

Examples of useful fixes:

- `en_30__retrieval__urs::en::00060.json__turn_0`: gold `2`, `cal=4`, `task_guard=3`.
- `en_55__retrieval__urs::en::00106.json__turn_0`: gold `2`, `cal=4`, `task_guard=3`.
- `en_79__advice__urs::en::00155.json__turn_0`: gold `2`, `cal=4`, `task_guard=3`.
- `zh_1__professional__urs::zh::00000.json__turn_0`: gold `2`, `cal=4`, `task_guard=3`.
- `zh_575__professional__urs::zh::00427.json__turn_0`: gold `2`, `cal=4`, `task_guard=3`.

Examples of remaining / new harms:

- `en_163__retrieval__urs::en::00297.json__turn_0`: gold `2`, `cal=2`, `task_guard=4`.
  This remains a hard case where the response is topically aligned but gold is dissatisfied.
- `en_111__professional__urs::en::00194.json__turn_0`: gold `5`, `cal=4`, `task_guard=3`.
  The professional guard appears too strict.
- `en_132__advice__urs::en::00234.json__turn_0`: gold `5`, `cal=4`, `task_guard=3`.
  Advice guard may over-penalize simple but accepted suggestions.
- `zh_39__text__urs::zh::00157.json__turn_0`: gold `4`, `cal=4`, `task_guard=2`.
  Text guard can be too harsh.

## Conclusion

`urs_v2_calibrated_task_guarded` with memory should become the leading single-run URS evaluator candidate.
It improves global ordinal metrics, SAT/DSAT boundary metrics, and both language subsets.

The no-memory task-guarded variant should not be used as the main evaluator.
Its large drop suggests that task guards alone are not enough; the memory/context branch is helping stabilize the judgment.

Recommended next step:

- Keep memory `task_guard` as the main candidate.
- Inspect and soften the `text` guard.
- Possibly make `professional` and `advice` guards less aggressive for simple but accepted answers.
- Re-evaluate after a small guard refinement rather than returning to language-specific prompting.
