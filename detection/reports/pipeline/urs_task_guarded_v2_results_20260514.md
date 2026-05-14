# URS Task-Guarded V2 Results

## Setup

Evaluated:

- Memory V2: `outputs/urs/Qwen_Qwen3-8B_test_none_urs_v2_calibrated_task_guarded_v2.jsonl`
- No-memory V2: `outputs/urs/Qwen_Qwen3-8B_test_no_memory_urs_v2_calibrated_task_guarded_v2.jsonl`

Compared against:

- `cal`
- memory `task_guarded` V1
- no-memory `cal`
- no-memory `task_guarded` V1

Evaluation output:

- `outputs/urs/qwen3_8b_urs_task_guarded_v2_comparison.json`

## Overall Metrics

| run | MAE | RMSE | Pearson | Spearman | QWK | Acc | F1-DSAT | F1-SAT | FalseSAT | FalseDSAT |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `cal` | 0.7260 | 1.0353 | 0.2358 | 0.2495 | 0.2073 | 0.6370 | 0.4592 | 0.7268 | 0.5610 | 0.2559 |
| `task_v1` | 0.6952 | 1.0119 | 0.2760 | 0.2883 | 0.2424 | 0.6627 | 0.5207 | 0.7398 | 0.4780 | 0.2612 |
| `task_v2` | 0.7209 | 1.0378 | 0.2058 | 0.2224 | 0.1746 | 0.6353 | 0.4409 | 0.7294 | 0.5902 | 0.2427 |
| `no_cal` | 0.7928 | 1.0925 | 0.1621 | 0.1527 | 0.1411 | 0.6301 | 0.3415 | 0.7429 | 0.7268 | 0.1768 |
| `no_task_v1` | 0.8921 | 1.1842 | 0.0915 | 0.0848 | 0.0837 | 0.6216 | 0.3912 | 0.7255 | 0.6537 | 0.2296 |
| `no_task_v2` | 0.8305 | 1.1623 | 0.1181 | 0.1043 | 0.1040 | 0.6610 | 0.3963 | 0.7643 | 0.6829 | 0.1530 |

## Main Result

Memory `task_v2` is worse than memory `task_v1` on every main metric.
It also falls below the original `cal` on Pearson, Spearman, QWK, F1-DSAT, and boundary accuracy.

The intended softening did recover some high-gold cases, but it overcorrected toward SAT predictions:

- V1 predicted SAT/DSAT: `378 / 206`, almost exactly matching gold `379 / 205`.
- V2 predicted SAT/DSAT: `413 / 171`.
- FalseSAT increased from `0.4780` to `0.5902`.
- F1-DSAT dropped from `0.5207` to `0.4409`.

Conclusion: V2 should not replace V1.

## Language Breakdown

| run | lang | MAE | Pearson | Spearman | QWK | Acc | F1-DSAT | F1-SAT |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `task_v1` | en | 0.7227 | 0.2505 | 0.2733 | 0.2117 | 0.6955 | 0.5109 | 0.7789 |
| `task_v2` | en | 0.7455 | 0.1675 | 0.1707 | 0.1394 | 0.6545 | 0.3770 | 0.7610 |
| `task_v1` | zh | 0.6786 | 0.2956 | 0.2971 | 0.2612 | 0.6429 | 0.5255 | 0.7137 |
| `task_v2` | zh | 0.7060 | 0.2247 | 0.2472 | 0.1909 | 0.6236 | 0.4710 | 0.7079 |

V2 hurts both English and Chinese.
This confirms that the regression is not language-specific.

## Task Breakdown

| task | V1 MAE | V2 MAE | V1 F1-DSAT | V2 F1-DSAT | interpretation |
|---|---:|---:|---:|---:|---|
| `advice` | 0.6250 | 0.6667 | 0.5789 | 0.4545 | softened too much |
| `creative` | 0.7031 | 0.6719 | 0.4091 | 0.3000 | MAE improves, boundary worsens |
| `leisure` | 0.7083 | 0.7917 | 0.4848 | 0.4615 | worse |
| `professional` | 0.7778 | 0.7692 | 0.5417 | 0.4324 | MAE slight gain, DSAT collapse |
| `retrieval` | 0.6606 | 0.7030 | 0.5714 | 0.5098 | worse despite unchanged retrieval text |
| `text` | 0.7093 | 0.7442 | 0.4561 | 0.4074 | not fixed |

The original goal was to improve `text` and reduce over-strictness in `professional/advice`.
V2 did not achieve this.
It made `professional/advice` less strict, but the main effect was to convert many correct DSAT predictions into false-SAT.

## Delta Analysis

Relative to memory V1:

- `64 / 584` samples improved by absolute error.
- `77 / 584` worsened.
- `443 / 584` unchanged.
- Score deltas: `+1` for `82`, `-1` for `57`, `+2` for `1`, `-2` for `1`, `0` for `443`.

Key harmful pattern:

- V1 correct DSAT -> V2 false-SAT occurred in many cases.
- Examples include:
  - `en_130__professional__urs::en::00228.json__turn_0`: gold `2`, V1 `3`, V2 `4`.
  - `en_55__retrieval__urs::en::00106.json__turn_0`: gold `2`, V1 `3`, V2 `4`.
  - `en_79__advice__urs::en::00155.json__turn_0`: gold `2`, V1 `3`, V2 `4`.
  - `zh_18__professional__urs::zh::00065.json__turn_0`: gold `1`, V1 `2`, V2 `4`.
  - `zh_575__professional__urs::zh::00427.json__turn_0`: gold `2`, V1 `3`, V2 `4`.

Useful V2 fixes:

- Some high-gold false-DSAT cases were recovered:
  - `en_111__professional__urs::en::00194.json__turn_0`: gold `5`, V1 `3`, V2 `4`.
  - `en_132__advice__urs::en::00234.json__turn_0`: gold `5`, V1 `3`, V2 `4`.
  - `en_142__text__urs::en::00247.json__turn_0`: gold `5`, V1 `3`, V2 `4`.
  - `en_79__professional__urs::en::00156.json__turn_0`: gold `5`, V1 `3`, V2 `4`.

These improvements are real, but fewer and less important than the DSAT losses for the current evaluator goal.

## No-Memory Result

No-memory V2 improves over no-memory V1 but is still weak:

- MAE: `0.8921 -> 0.8305`
- QWK: `0.0837 -> 0.1040`
- F1-DSAT: `0.3912 -> 0.3963`

It remains worse than memory V1 and the original memory `cal`.
No-memory task-guard variants should not be used as primary URS evaluators.

## Conclusion

Use memory `urs_v2_calibrated_task_guarded` V1 as the current best single-run URS evaluator.
Do not use V2 as the default.

The V2 experiment is still informative:

- Softening task guards globally is too blunt.
- The V1 gains come from a carefully balanced DSAT guard; relaxing it quickly increases false-SAT.
- Future refinement should be narrower than V2, likely case-level or task-specific post-processing rather than broad prompt softening.

Recommended next step:

- Keep V1.
- If further tuning is needed, use targeted post-hoc calibration or an ensemble combining V1 with `cal`, rather than prompt-softening V1 into V2.
