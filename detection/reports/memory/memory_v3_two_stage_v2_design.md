# memory_v3_two_stage_v2 design

## Motivation

`detection/reports/memory/memory_v3_two_stage_subset20_results.md` showed that:

- the two-stage idea is correct
- the main bottleneck is still the first-stage `3/4` SAT gate
- the second-stage `4/5` and `1/2/3` refiners are not the dominant failure source

At the same time:

- `memory_v3` carries stronger calibration information than the boundary prompts
- `boundary_34_selective_refute_v2` has a better `3/4` boundary decision style than the original `v3_two_stage` gate

So `v3_two_stage_v2` is designed as a clean hybrid:

- keep `memory_version=v3`
- replace the first-stage gate with a `selective-refute v2` style gate
- keep the existing second-stage SAT/DSAT refiners unchanged

## Pipeline

### Stage 1: selective SAT gate

The first stage now uses a new prompt version:

- `v3_two_stage_v2_gate`

It is still a `3/4` gate, but its judgment style is closer to `boundary_34_selective_refute_v2`:

- first decide whether the reply answered the core question and satisfied key constraints
- then decide whether it crosses the user’s minimum satisfaction line
- only mark `needs_refute_review=true` for a small set of genuinely ambiguous boundary cases

Important constraints:

- `calibration_summary` and `avg_satisfaction_score` are background only
- they cannot override clear evidence that the reply missed the core question or key requirements
- if `3/4` evidence is weak, the model should rely more on core utility, key requirements, and concrete usability rather than defaulting to SAT

### Stage 1.5: optional gate follow-up

If the first pass marks a high-uncertainty boundary case, the pipeline runs:

- `v3_two_stage_v2_gate_followup`

This follow-up keeps the same principle as `selective_refute_v2`:

- default to preserving the first-pass gate result
- only revise the gate if there is explicit counter-evidence

This means the new gate is:

- stronger than the original one-pass `v3_two_stage` gate
- still much lighter than running a full second fullscale router

### Stage 2: unchanged branch refinement

After the gate is finalized:

- if gate output is `4`, run existing SAT refiner (`4/5`)
- if gate output is `3`, run existing DSAT refiner (`1/2/3`)

No branch logic is changed in this version. The experiment is intended to isolate whether a better first-stage gate is enough to improve the full pipeline.

## Additional output fields

Besides the existing two-stage fields, `v3_two_stage_v2` writes:

- `two_stage_gate_model_flag`
- `two_stage_gate_triggered`
- `two_stage_gate_refute_applied`
- `analysis_gate_first_pass`
- `analysis_gate_followup`

These fields are intended to diagnose:

- how often the first-pass gate requests review
- how often review is actually triggered
- whether the gain comes from better first-pass judgment or from gate follow-up corrections

## Recommended experiment

```bash
cd detection
model=Qwen/Qwen3-8B \
memory_update_mode=none \
memory_version=v3 \
turn_eval_prompt_version=v3_two_stage_v2 \
limit_users=20 \
output_jsonl=outputs/personalized/Qwen_Qwen3-8B_test_none_memv3_v3_two_stage_v2_u20.jsonl \
bash scripts/collect_personalized_vllm.sh
```

## Expected comparison target

This version should be compared first against:

- `memv3_v3`
- `memv3_v3_1`
- `memv3_two_stage`
- same-subset `Qwen none`

The most important metrics are:

- fullscale: `MAE / Pearson / QWK`
- boundary: `F1-DSAT / false_sat_rate / false_dsat_rate`
- gate-specific: trigger rate and whether final boundary accuracy improves beyond the original `v3_two_stage`
