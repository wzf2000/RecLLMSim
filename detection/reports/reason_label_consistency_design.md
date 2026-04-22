# Reason Label Consistency Design

## Problem

In the satisfaction data, dissatisfaction reasons are only meaningful when the
gold satisfaction score is `<= 3`. For turns with score `>= 4`, the gold reason
is always `满意`.

Before this update, multiple prompts only told the model to choose a reason from
the full label set, which allowed invalid combinations such as:

- `classification=4`, `reason=不满足需求`
- `classification=5`, `reason=不够细致`
- `classification=3`, `reason=满意`

This made the predicted reason field semantically inconsistent with the ground
truth definition.

## Design Decision

The project now treats reason/score consistency as an explicit validity rule:

- If `classification >= 4`, `reason` must be `满意`.
- Only if `classification <= 3`, `reason` may come from dissatisfaction labels.
- Outputs violating this rule are considered invalid and are normalized before
  being written to traces.

## Prompt Changes

The legality rule is now written explicitly into:

- `build_turn_eval_prompt(...)`
- `build_turn_eval_refute_followup_prompt(...)`
- `build_turn_eval_prompt_no_memory(...)`
- `trace/collect_api.py::build_prompt(...)`

Each prompt now includes:

- the valid mapping between score and reason
- a statement that inconsistent outputs are invalid
- a JSON schema description that encodes the same rule

## Runtime Normalization

To keep outputs aligned with the dataset semantics, code-side normalization was
added through `lib.satisfaction_constants.normalize_reason_for_score(...)`.

Behavior:

- predicted score `>= 4` always maps to reason `满意`
- predicted score `<= 3` cannot keep reason `满意`; invalid labels fall back to
  a dissatisfaction default (currently `其它`)
- invalid raw pairs are logged with a warning for diagnosis

This normalization is applied in:

- `trace/collect_personalized.py`
- `trace/collect_api.py`

## Selective Boundary Side Effect

`boundary_34_selective_refute_v2/v3/v4` previously used non-satisfied reason
labels on `classification=4` examples as part of the second-pass trigger gate.

After enforcing the correct semantics, that signal is no longer available.
Therefore the gate was updated:

- for first-pass `classification=3`, keep the old reason-based filter
- for first-pass `classification=4`, rely on `needs_refute_review=true`

This keeps the selective-refute logic compatible with the corrected reason
definition.
