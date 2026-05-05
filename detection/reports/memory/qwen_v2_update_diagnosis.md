# Why Qwen3 v2 memory update is not helping

## Question

For `Qwen/Qwen3-8B` with `memory_v2`, why do `per_session` / `per_session_oracle` / `per_turn` fail to produce meaningful gains over `none`, even though update looked promising conceptually?

This note focuses on:

- what the outputs actually change
- whether update is truly a no-op
- what in the update content makes it ineffective

## Raw metric fact

From `detection/reports/overview/personalized_satisfaction_results.md`:

| method | MAE | Pearson | Spearman | QWK |
|---|---:|---:|---:|---:|
| `Qwen none` | `0.7110` | `0.2967` | `0.2820` | `0.2815` |
| `Qwen per_session` | `0.7133` | `0.2812` | `0.2619` | `0.2694` |
| `Qwen per_session_oracle` | `0.7141` | `0.2737` | `0.2563` | `0.2632` |
| `Qwen per_turn` | `0.7087` | `0.2794` | `0.2641` | `0.2694` |

So the high-level observation is correct:

- update does not produce a meaningful gain
- oracle update also does not help
- per-turn is not clearly better either

## First key finding: update is not a no-op

Comparing the full JSONL outputs turn by turn:

| comparison | score changed | reason changed |
|---|---:|---:|
| `none` vs `per_session` | `1145 / 6474 = 17.69%` | `939 / 6474 = 14.50%` |
| `none` vs `oracle` | `1183 / 6474 = 18.27%` | `909 / 6474 = 14.04%` |
| `none` vs `per_turn` | `1190 / 6474 = 18.38%` | `1000 / 6474 = 15.45%` |

So update definitely changes downstream predictions.

The problem is not “memory update has zero effect”.

The problem is:

- it changes a non-trivial number of turns
- but the changes are almost as often harmful as helpful

## Second key finding: changed turns are almost evenly split between better and worse

Using `none` as reference, for turns whose score changed:

### `per_session`

- better: `564`
- worse: `581`
- net: `-17`

### `oracle`

- better: `580`
- worse: `602`
- tie in absolute error: `1`
- net: `-22`

### `per_turn`

- better: `602`
- worse: `588`
- net: `+14`

Even if we only look at later sessions inside each block (where update should matter most), the net effect is still not positive:

| mode | later-session better | later-session worse | net |
|---|---:|---:|---:|
| `per_session` | `452` | `477` | `-25` |
| `oracle` | `466` | `494` | `-28` |
| `per_turn` | `457` | `468` | `-11` |

This is the clearest diagnosis:

**update is active, but the update signal is too noisy and too weakly targeted to improve net accuracy.**

## Third key finding: most changes are shallow 3↔4 and 4↔5 nudges

Top transition types relative to `none`:

### `per_session`

- `4 -> 5`: `360`
- `3 -> 4`: `264`
- `5 -> 4`: `233`
- `4 -> 3`: `199`

### `oracle`

- `4 -> 5`: `407`
- `3 -> 4`: `281`
- `5 -> 4`: `227`
- `4 -> 3`: `170`

### `per_turn`

- `4 -> 5`: `467`
- `3 -> 4`: `278`
- `5 -> 4`: `215`
- `4 -> 3`: `139`

This means update mostly changes:

- the local `3/4` boundary
- the local `4/5` boundary

But it rarely repairs the deeper low-score structure:

- `1` and `2` remain rare
- severe DSAT is still not being modeled cleanly

So the update is nudging the score scale, not teaching the model a sharper user-specific failure boundary.

## Concrete example: `User_84 / 技能学习规划`

This block is one of the strongest oracle-update cases in terms of later-session changed turns.

Its initial cached `v2` memory says:

- `avg_satisfaction_score = 2.91`
- `scoring_style = 严格`
- `three_vs_four_distinction` emphasizes missing special needs / lack of diversity
- `four_vs_five_distinction` emphasizes concrete names, structure, and detail

That memory is already quite specific.

### What the oracle update sees

For the first target session (`0.json`), the oracle update prompt contains four turns with mixed labels:

- `4（满意）`
- `3（不可用）`
- `5（满意）`
- `2（不满足需求）`

But the prompt only provides:

- the existing memory JSON
- the raw dialogue snippets
- the turn-level gold score and gold reason

It does **not** provide:

- a structured explanation of why a turn was `2` rather than `4`
- a concise extracted delta like “this reply failed because it did not answer the user’s last question”
- contrastive summaries of adjacent examples (`3 vs 4`, `4 vs 5`)

### What happens downstream

On this same block, later-session predictions often move, but not in a reliably helpful way.

Examples:

- `2.json turn 0`, gold `1`
  - `none -> 3`
  - `per_session -> 5`
  - `oracle -> 4`
  - `per_turn -> 3`

- `3.json turn 0`, gold `2`
  - `none -> 3`
  - `per_session -> 4`
  - `oracle -> 4`
  - `per_turn -> 4`

- `3.json turn 2`, gold `2`
  - `none -> 3`
  - `per_session -> 4`
  - `oracle -> 4`
  - `per_turn -> 4`

So even oracle update, after seeing more labels, still pushes several clearly unsatisfied turns upward into SAT.

This is not consistent with “the model learned a sharper dissatisfaction boundary”.

It is more consistent with:

- the update absorbing a vague signal like “this user values detailed structured plans”
- but failing to learn exactly what makes a reply fall below the satisfaction line

## Why the update content is structurally weak

Inspecting the actual `build_memory_update_prompt()` content shows four main issues.

### 1. The supervision is coarse

The update prompt only exposes:

- raw assistant reply text
- gold score
- gold reason

For example, a turn may be labeled:

- `★2（不满足需求）`

But the update prompt does not explicitly say:

- which user requirement was missed
- whether the failure was about core-task mismatch, missing detail, format, or unusability
- whether this should update `three_vs_four_distinction`, `four_vs_five_distinction`, or only calibration

So the LLM must infer the update delta by itself from long raw text.

### 2. The prompt is explicitly conservative

The update prompt instructs:

- keep memory unchanged if the new session is broadly consistent
- only modify distinction fields when there is explicit contradiction

This is a good defense against noise, but for Qwen it likely over-suppresses the only fields that actually matter at inference:

- `three_vs_four_distinction`
- `four_vs_five_distinction`
- `user_specific_requirements`

As a result, many updates probably collapse to:

- small distribution/statistics shifts
- slight paraphrases

rather than meaningful boundary revision.

### 3. The update rewrites the whole memory at once

The model is asked to regenerate the entire memory JSON.

That means it must simultaneously decide whether to change:

- calibration fields
- 3/4 boundary
- 4/5 boundary
- user-specific requirements
- preferred format
- task-specific observations

This encourages diffuse rewriting instead of targeted patching.

For Qwen, that likely causes:

- some random local wording changes
- some shallow score-scale nudges
- little stable improvement in the exact fields used for scoring

### 4. The turn-eval prompt does not strongly exploit the fields most safely updated

In `v2` turn evaluation, the memory rubric prominently uses:

- `three_vs_four_distinction`
- `four_vs_five_distinction`
- `user_specific_requirements`

while the updated statistics:

- `avg_satisfaction_score`
- `score_distribution`

are present but do not dominate the decision process.

So even if update correctly refreshes the numeric distribution, that benefit may not propagate strongly into the final prediction.

Meanwhile, if the verbal distinction fields do change, they change noisily.

This is a bad combination:

- safe updates affect weakly used fields
- strongly used fields are hard to update reliably

## Bottom-line explanation

For `Qwen3 + memory_v2`, update is ineffective because it sits in the worst possible regime:

1. it is **not strong enough** to reliably rewrite the true `3/4` or `4/5` boundary
2. it is **not precise enough** to only change the parts that need to change
3. it still changes predictions often enough to introduce noise

So the empirical behavior becomes:

- update changes about 18% of turns
- but mostly through shallow boundary nudges
- and these nudges are nearly evenly split between helpful and harmful

That is exactly why the final metrics barely move.

## Practical implication

The current result does **not** mean:

- “user adaptation after observing new sessions is useless”

It means:

- **the current `v2` update interface is too weak and too noisy**

If you want update to become useful, the most promising changes are:

### 1. Update calibration and rules separately

Do not ask the model to rewrite the whole memory JSON.

Instead:

- update `avg_satisfaction_score` / `score_distribution` deterministically in code
- only ask the LLM to revise boundary fields when there is explicit evidence

### 2. Feed structured delta evidence, not just raw session text

For each turn, provide something like:

- gold score
- gold reason
- whether the core question was answered
- what key requirement was missed
- whether the failure is ordinary “not detailed enough” or true below-threshold dissatisfaction

This would give the update stage the missing “what exactly changed” supervision.

### 3. Only update boundary rules from adjacent-score evidence

For example:

- update `three_vs_four_distinction` only when a session contains useful `3/4` evidence
- update `four_vs_five_distinction` only when there is useful `4/5` evidence

Otherwise keep those fields unchanged.

### 4. Make update field-specific instead of full-memory rewrite

Return a patch-style output such as:

- `update_calibration: yes/no`
- `update_34_boundary: yes/no + revised text`
- `update_45_boundary: yes/no + revised text`
- `update_requirements: append/remove/unchanged`

This would make the update mechanism much easier to control and analyze.

## Final conclusion

For Qwen3 `memory_v2`, update “looks useless” not because it does nothing, but because:

- it changes predictions noticeably
- yet the content of the update is too coarse to sharpen the user boundary
- and too conservative to produce a reliable net gain

So the more accurate statement is:

**Qwen3 v2 update is active, but its current content and interface make it a noisy boundary nudge rather than a useful user-model refinement mechanism.**
