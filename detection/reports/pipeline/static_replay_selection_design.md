# Static Replay Selection Design

Date: 2026-05-10

This note proposes turn-selection rules for Static Replay Evaluation. The goal is
to reduce replay generation cost while keeping the selected turns informative for
user-specific satisfaction benchmarking.

## Motivation

The current static replay pipeline can replay every historical assistant turn.
This is expensive and not always informative:

- Many original assistant turns are already highly satisfactory, so replaying
  them mostly tests whether a candidate model can preserve easy wins.
- Some turns are low-value for evaluation, such as very short acknowledgements,
  transitional turns, or turns whose satisfaction is mostly determined by later
  dialogue context.
- Replaying every turn in the same dialogue overweights long sessions and makes
  model comparison cost scale poorly.

The selection policy should therefore prefer turns where an alternative response
could plausibly improve or damage user satisfaction, while maintaining coverage
across users and tasks.

## Recommended Selection Policy

Use a two-stage policy:

1. Candidate filtering: remove turns that are unlikely to be useful evaluation
   points.
2. Budgeted sampling: rank the remaining turns and sample a capped number per
   user/task/session.

This avoids relying on a single heuristic and makes the evaluation budget
explicit.

## Candidate Filtering Rules

### 1. Prefer Low- or Borderline-Satisfaction Turns

Select turns whose original satisfaction score is low or near the 3/4 boundary:

- Always include turns with original score `<= 3` when budget permits.
- Include score `4` turns only when they are uncertain or boundary-like.
- Downsample score `5` turns heavily, unless they are needed as positive anchors.

Rationale: the benchmark should test whether a candidate model can improve weak
responses and avoid crossing the satisfied/dissatisfied boundary. Score `5`
turns are often less diagnostic and can dominate the replay budget.

### 2. Use Dissatisfied Reason as a Strong Signal

For turns with score `<= 3`, prioritize those with a meaningful dissatisfied
reason label:

- High priority: specific, actionable reason labels such as missing constraint,
  wrong assumption, insufficient detail, poor personalization, or failure to
  follow user preference.
- Lower priority: generic or ambiguous dissatisfaction labels if no other signal
  supports the turn.

Rationale: reason-labeled dissatisfied turns are better for diagnosing whether a
candidate model fixes the original failure mode.

### 3. Filter Non-Substantive Assistant Turns

Exclude turns that are unlikely to represent a meaningful response-quality
decision:

- pure acknowledgement or transition
- asking a clarification question without substantive answer, unless the task is
  specifically about clarification behavior
- boilerplate safety/disclaimer text with no task progress
- very short replies under a token/character threshold
- turns where the user prompt is also empty, malformed, or only a continuation
  marker

Rationale: these turns inflate cost but produce noisy satisfaction judgments.

### 4. Prefer Turns With Enough Local Context

Keep turns where the candidate model can reasonably answer from replayed context:

- user request is explicit enough in the local/session context
- required constraints are present before the target turn
- no hidden information from future turns is needed to judge response quality

Exclude turns that depend heavily on future corrections or unavailable external
state.

Rationale: static replay should be fair to the replayed model. The model should
not be penalized for missing information that was not available at that point.

### 5. Keep Some High-Satisfaction Positive Controls

Reserve a small budget for original score `5` turns:

- include only high-substance turns
- prefer tasks/users where the model previously had low-score selected turns
- cap them aggressively, e.g. at 10-20% of selected turns

Rationale: all-negative selection can make the benchmark measure only repair
ability. Positive controls test whether the candidate model can preserve good
answers and help detect over-refusal or over-correction.

## Budgeting Rules

### Per-Session Cap

Do not replay every turn from the same dialogue. Recommended default:

- select at most 1-2 replay turns per session
- if a session has multiple low-score turns, choose the earliest severe failure
  and optionally one later boundary turn

Rationale: multiple turns in one session are highly correlated. A per-session cap
improves diversity and cost efficiency.

### Per-User/Task Cap

Use a fixed cap per `(user, target_task)` block:

- small smoke benchmark: 2-4 turns per block
- medium benchmark: 4-6 turns per block
- full benchmark: 6-8 turns per block

Rationale: this prevents active users or long dialogues from dominating the
benchmark and preserves comparability across users/tasks.

### Stratified Sampling

Within each `(user, task)` block, target a mix like:

- 50-60% low-score turns (`<= 3`)
- 20-30% boundary or uncertain score `4` turns
- 10-20% strong positive score `5` controls

If a block lacks enough dissatisfied turns, fill with boundary-like score `4`
turns rather than forcing low-quality positives.

## Ranking Score

A practical ranking score for candidate turns:

```text
selection_score =
  3.0 * is_score_le_3
+ 2.0 * is_score_3_or_4_boundary
+ 1.5 * has_specific_dsat_reason
+ 1.0 * is_substantive_answer
+ 1.0 * has_clear_user_request
+ 0.5 * predictor_uncertainty
- 1.5 * is_non_substantive
- 1.0 * same_session_already_selected
- 0.5 * is_score_5
```

`predictor_uncertainty` can be estimated from disagreement between available
predictors, calibration confidence, boundary confidence, or model self-reported
uncertainty if present. This term is optional.

## Suggested Default for Current Project

For the current personalized satisfaction benchmark, use this conceptual default:

- Per session: select at most 1 target turn.
- Per `(user, task)` block: select up to 4 target turns.
- Include all available score `<= 3` substantive turns first.
- If budget remains, add score `4` turns near the 3/4 boundary.
- Reserve at most one score `5` positive-control turn per block.
- Exclude non-substantive turns using the existing turn-content annotation where
  available; otherwise use a lightweight heuristic based on response length and
  answer type.

This gives a cheaper benchmark that focuses on user-specific improvement and
boundary sensitivity without making the replay set purely negative.

The implemented `hard` / `filter` collection mode uses a conservative default
budget to stay below 1000 turns on the current test split:

- `selection_mode=hard` or `selection_mode=filter`
- `hard_max_per_session=1`
- `hard_max_per_block=3`
- `hard_global_budget=300`
- `hard_positive_controls_per_block=1`
- `hard_min_turn_idx=1`, which skips the first assistant turn in each session
- `hard_score_quota=1:25,2:50,4:75`, which reserves some examples for severe
  failures and boundary-satisfied turns before filling the rest by selection
  score
- `hard_min_per_user=1`, which first reserves one selected turn for every user
  that has an eligible mid-dialogue candidate before filling the remaining
  budget

With the 2026-05-10 test split (`90` users, `356` user-task blocks, `6474` full
target turns), this selects `300` turns before replay generation and focuses on
mid-dialogue turns starting from the second assistant reply.

## Reporting Requirements

Any static replay result collected under a selection policy should report:

- selection policy name and version
- total selected turns and selected-turn ratio
- distribution by gold score
- distribution by task and user
- number of selected turns per session
- fraction of non-substantive turns filtered out
- original-response satisfaction distribution before candidate replay

These statistics are needed to interpret benchmark scores and compare different
selection policies fairly.
