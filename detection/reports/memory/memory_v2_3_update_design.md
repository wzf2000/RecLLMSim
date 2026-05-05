# Memory V2.3 Update Design

## Motivation

`v2.1` worked because it kept:

- raw local turn evidence
- field-level patching
- deterministic code-side calibration updates

`v2.2` regressed because it likely over-corrected:

- evidence was over-compressed into a structured bundle
- harder gates removed useful signal together with noisy signal

So `v2.3` is intentionally a **small-step refinement of `v2.1`**, not a new redesign.

## Core Idea

Keep the successful parts of `v2.1`:

- same patch schema (`MemoryUpdatePatchV2_1`)
- same raw turn-level update examples
- same deterministic score-distribution / average-score update

Only add two lightweight changes:

1. a short boundary summary block in the update prompt
2. a light program-side boundary gate in merge

## Prompt Changes

`build_memory_update_prompt_v2_3()` stays close to `v2.1`, but adds:

- whether the session contains both `3` and `4`
- whether the session contains both `4` and `5`
- counts of `<=3`, `4`, `5`
- an explicit instruction:
  - if there is no adjacent-score evidence, do not rewrite long-term boundary rules

Importantly, `v2.3` still shows the raw low / mid / high examples directly, instead of replacing them with a heavily abstracted evidence bundle.

## Merge Changes

`merge_memory_v2_3_patch()` differs from `v2.1` only in two ways:

1. boundary fields are lightly gated
   - `three_vs_four_distinction` updates only if the new session has both `3` and `4`
   - `four_vs_five_distinction` updates only if the new session has both `4` and `5`

2. generic requirements are filtered
   - examples such as `更详细 / 更具体 / 更清晰 / 更结构化 / 更实用` are not added as long-term requirements

Everything else stays intentionally close to `v2.1`.

## Expected Outcome

Relative to `v2.1`, `v2.3` is expected to:

- preserve the useful local evidence that made `v2.1` work
- slightly reduce noisy boundary rewrites
- avoid the heavy regression seen in `v2.2`

This is a conservative experiment. It is not expected to produce a dramatic jump, but it should be much less risky than `v2.2`.

## Recommended Comparison

Minimum comparison set:

1. `Qwen none`
2. `Qwen per_session + v2.1`
3. `Qwen per_session + v2.3`

Optional:

4. `Qwen per_session_oracle + v2.3`

## Success Criteria

`v2.3` should be considered promising if it can beat `v2.1` on at least one of:

- `MAE / RMSE`
- `Pearson / QWK`
- `F1-DSAT`
- `false_sat_rate`
- user-aware boundary metrics

without clearly regressing on the others.
