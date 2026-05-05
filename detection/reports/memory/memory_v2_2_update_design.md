# Memory V2.2 Update Design

## Motivation

`memory_update_prompt_version=v2_1` already proved that Qwen3 update can be made useful:

- `per_session_oracle + v2.1` clearly beats `Qwen none`
- `per_session + v2.1` also improves `MAE / RMSE`

But the gap between oracle and non-oracle is still large.

Observed pattern:

- oracle `v2.1` improves global + boundary metrics together
- non-oracle `v2.1` mainly improves absolute error
- non-oracle `v2.1` still behaves more like a SAT-leaning calibration adjustment than a strong boundary improvement

This suggests that the remaining bottleneck is not the patch idea itself, but the **quality of update evidence** under noisy non-oracle labels.

## Core Idea

`v2.2` keeps the `v2.1` patch framework, but changes two things:

1. **input**: from raw session text to a structured evidence bundle
2. **merge policy**: from relatively permissive field patching to stronger program-side gates

The goal is to let non-oracle update benefit more from stable local patterns while reducing noisy boundary rewrites.

## Design

### 1. Structured evidence bundle

Instead of sending the whole new session as free-form text, `v2.2` first builds a compact `Evidence Bundle` containing:

- `session_summary`
  - number of turns
  - score sequence
  - mean / min / max score
  - task context excerpt
  - whether oracle labels are used

- `current_memory_summary`
  - current average score
  - current score distribution
  - current scoring style

- `boundary_evidence`
  - number of `<=3`, `4`, `5`
  - whether `3&4` both appear
  - whether `4&5` both appear
  - compact low / mid / high examples

- `uncertainty_signals`
  - whether this is prediction-only update
  - borderline-analysis mention count
  - `3/4` score-flip count
  - `4/5` score-flip count
  - dominant score and its ratio

The intention is to make update explicitly aware of:

- whether there is actually enough boundary evidence
- whether this session is noisy / unstable
- whether the update should primarily affect calibration or rules

### 2. Richer patch schema

`v2.2` patch output is more explicit than `v2.1`:

- each boundary field now has:
  - `update_*`
  - new text
  - `confidence`
  - `evidence_count`

- candidate requirements now carry:
  - `requirement`
  - `confidence`
  - `support_count`

- task observations now carry:
  - `task_name`
  - `observation`
  - `confidence`
  - `support_count`

This lets the program decide not only **what** the model wants to change, but also **how strongly supported** that change is.

### 3. Harder program-side gates

`v2.2` makes merge behavior stricter:

- `three_vs_four_distinction` updates only if:
  - session has both `3` and `4`
  - confidence is `medium/high`
  - evidence count >= 2

- `four_vs_five_distinction` updates only if:
  - session has both `4` and `5`
  - confidence is `medium/high`
  - evidence count >= 2

- `preferred_response_format` updates only if:
  - confidence is `high`
  - support count >= 2

- requirement additions are filtered if:
  - confidence is `low`
  - support count < 2
  - requirement text looks generic (`详细 / 具体 / 清晰 / 结构化 / 实用 ...`)

Statistics (`avg_satisfaction_score`, `score_distribution`) are still updated deterministically in code.

## Expected Benefits

Compared with `v2.1`, `v2.2` should ideally:

1. reduce noisy non-oracle boundary rewrites
2. preserve the global error gains already achieved by `v2.1`
3. recover more of the oracle-style gains on:
   - `Pearson / Spearman / QWK`
   - `F1-DSAT`
   - `false_sat_rate`
   - user-aware boundary metrics

## Experiments to Run

Minimum comparison set:

1. `Qwen none`
2. `Qwen per_session + v2.1`
3. `Qwen per_session + v2.2`

Optional upper bound:

4. `Qwen per_session_oracle + v2.2`

## Success Criteria

`v2.2` should be considered successful if it can improve over `per_session + v2.1` on at least one of these dimensions without clear regression elsewhere:

- lower `MAE`
- higher `Pearson / Spearman / QWK`
- higher `F1-DSAT`
- lower `false_sat_rate`
- higher `PU-bin F1-DSAT / PU-bin Kappa`

## Implementation Scope

`v2.2` is intentionally limited to:

- `memory_version=v2`
- update stage only

It does **not** modify:

- initial `v2` memory cache building
- turn-eval prompts
- memory v3 / two-stage pipelines

This keeps the experiment clean and makes it directly comparable with `v2.1`.
