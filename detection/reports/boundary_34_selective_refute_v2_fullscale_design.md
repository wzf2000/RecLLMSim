# boundary_34_selective_refute_v2_fullscale Design

## Goal

Build a full `1-5` personalized satisfaction pipeline without discarding the current best boundary system.

The design keeps `boundary_34_selective_refute_v2` as the first-stage router, because it is currently the most stable `3/4` boundary variant. Instead of asking one prompt to directly predict `1-5`, the new pipeline decomposes the task into:

1. `3/4` boundary routing
2. `4/5` refinement for SAT cases
3. `1/2/3` refinement for DSAT cases

This is intended to preserve the strongest part of the current system, namely user-specific satisfaction-boundary modeling, while restoring full-scale output.

## Pipeline

### Stage 1: Boundary Router

The first stage reuses the existing `boundary_34_selective_refute_v2` pipeline unchanged.

- Output space: `3` or `4`
- Memory usage: same `UserMemory v2` as current boundary runs
- Selective refute: same gate and second-pass logic as `boundary_34_selective_refute_v2`

Interpretation:

- router output `4` means: the reply has passed the user-specific satisfaction threshold
- router output `3` means: the reply has not passed the threshold

This first-stage output is treated as a routing decision, not the final score.

### Stage 2A: SAT Refinement

If the router predicts SAT (`>=4` side), a second prompt refines only between `4` and `5`.

Design choices:

- output space is restricted to `4` or `5`
- `reason` must be `满意`
- the prompt explicitly uses `four_vs_five_distinction`
- `three_vs_four_distinction` is still shown as background, but is no longer the decision focus
- anchor examples, if enabled, are instructed to support `4/5` comparison rather than generic similarity

Operational rule:

- default to `4`
- only upgrade to `5` when there is clear evidence that the response reaches the user-specific high-satisfaction bar

### Stage 2B: DSAT Refinement

If the router predicts DSAT (`<=3` side), a second prompt refines only between `1`, `2`, and `3`.

Design choices:

- output space is restricted to `1/2/3`
- `reason` must be a dissatisfaction reason
- the prompt explicitly distinguishes:
  - `3`: below the threshold but still partially helpful
  - `2`: major failure, weak usefulness
  - `1`: severe failure, almost unusable, clearly wrong, or badly misaligned
- anchor examples, if enabled, are instructed to support `1/2/3` severity comparison

Operational rule:

- do not over-compress all DSAT cases into `3`
- also do not overuse `1`
- reserve `1` and `2` for clearly stronger failures than a standard not-satisfied case

## Memory Usage

This pipeline keeps the same memory source as the current personalized boundary pipeline.

Most important memory fields:

- `three_vs_four_distinction`
- `four_vs_five_distinction`
- `user_specific_requirements`
- `preferred_response_format`
- `task_specific_observations`

Usage by stage:

- Stage 1 mainly uses `three_vs_four_distinction`
- Stage 2A mainly uses `four_vs_five_distinction`
- Stage 2B uses the user-specific requirements and task observations to estimate DSAT severity

## Output Semantics

The final output is still a standard turn-level prediction record, but the JSONL now also stores intermediate router signals for later analysis.

Additional fields written by this fullscale pipeline:

- `analysis_router`
- `analysis_sat_refine`
- `analysis_dsat_refine`
- `fullscale_router_score`
- `fullscale_router_reason`
- `fullscale_router_analysis`
- `fullscale_router_triggered`
- `fullscale_router_applied`
- `fullscale_router_initial_score`
- `fullscale_router_initial_reason`
- `fullscale_router_model_flag`
- `fullscale_branch`
- `fullscale_refine_applied`

This makes it possible to separately diagnose:

- whether the router is the bottleneck
- whether the SAT branch or DSAT branch is weaker
- whether the selective-refute stage inside the router is actually affecting downstream refinement

## Reason Consistency

This fullscale version follows the current global reason consistency rule:

- if final score `>=4`, `reason` must be `满意`
- only scores `<=3` may use dissatisfaction reasons

The same normalization logic in the collection pipeline is applied after branch refinement.

## Current Command

Recommended first experiment:

```bash
cd detection
model=Qwen/Qwen3-8B \
memory_update_mode=none \
turn_eval_prompt_version=boundary_34_selective_refute_v2_fullscale \
limit_users=20 \
output_jsonl=outputs/personalized/Qwen_Qwen3-8B_test_none_boundary_34_selective_refute_v2_fullscale_u20.jsonl \
bash scripts/collect_personalized_vllm.sh
```

## Expected Failure Modes

The most likely risks are:

1. the router remains the dominant bottleneck, so downstream refinement cannot help much
2. the SAT branch may collapse to mostly `4`
3. the DSAT branch may collapse to mostly `3`
4. anchors may help one branch but add latency or instability to the other

Because of this, the first evaluation should separately report:

- overall `1-5` metrics
- `3/4` boundary metrics derived from final predictions
- predicted distribution over `1/2/3/4/5`
- branch usage counts: `sat_45` vs `dsat_123`
