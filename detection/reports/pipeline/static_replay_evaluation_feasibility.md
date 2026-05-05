# Static Replay Evaluation Feasibility

## Goal

Use the best available user-specific satisfaction predictor as an automatic judge for LLM response benchmarking.

The proposed pipeline:

1. take existing user dialogue histories
2. replay each target assistant turn as a prompt to a candidate LLM
3. let the candidate LLM generate one response
4. score that response with the user-specific satisfaction predictor
5. aggregate predicted satisfaction into model-level benchmark scores

## Feasibility Summary

The current project design can support this idea, but it needs a thin new pipeline layer.

Already reusable:

- `PersonalizedSample` and `SessionData` provide user profile, task context, history sessions, and target dialogue prefixes.
- `build_user_memory(...)` can build a user-specific memory from source tasks.
- `build_turn_eval_prompt(...)` and `_call_predict_turn(...)` can score an arbitrary assistant response if we provide:
  - memory
  - profile
  - task context
  - dialogue prefix
  - candidate assistant reply
- `eval/personalized.py` already has most aggregation metrics for 1-5, boundary, user-aware, and task-level views.

Missing:

- a candidate-response generation pipeline
- a candidate-response JSONL schema
- a scoring pipeline that scores generated responses without requiring gold satisfaction labels
- benchmark-specific aggregation reports

## Recommended Evaluation Mode

Use **single-turn static replay** as the first version.

For each original assistant turn:

1. keep only the dialogue prefix before that assistant turn
2. ask the candidate model to answer the current user turn
3. score the candidate response using the frozen satisfaction predictor
4. do not roll the generated response into later turns

This is the cleanest setting because:

- all candidate models receive the same prefix
- future human turns remain irrelevant to the current score
- model outputs do not change the replay trajectory
- aggregation is straightforward

Avoid multi-turn generated replay for the first version. Once candidate responses become part of the future context, each model creates a different trajectory, and scores become much harder to compare.

## Predictor Choice

The current best practical base is still the Qwen3 memory-v2 line.

For benchmark scoring, the most defensible first judge is:

- `memory_version=v2`
- `memory_update_mode=none`
- `turn_eval_prompt_version=v2`
- optional post-hoc calibration as a separate reported variant

Reason:

- It is the most stable general predictor across global, user-aware, and boundary metrics.
- Update-based predictors can be useful for prediction experiments, but benchmark scoring should start from a frozen judge to avoid model-dependent memory drift.

`per_session + v2.1` is worth evaluating later as an adaptive-judge variant, but it should not be the default static benchmark judge.

## Why Frozen Memory Matters

If memory is updated using predicted scores from each candidate model, then the judge state becomes model-dependent:

- a strong model may push memory in one direction
- a weak model may push memory in another direction
- later scores are no longer produced by the same judge state

That can be meaningful for an "adaptive user simulation" benchmark, but it is not ideal for the first static replay benchmark.

Recommended first track:

- build memory once from source-task history
- keep memory fixed
- score every candidate response with the same user/task memory

Optional second track:

- adaptive replay with `per_session + v2.1`
- report separately

## Suggested Output Schema

Candidate generation file:

```json
{
  "sample_id": "User_0__礼物准备__0.json__turn_0",
  "user": "User_0",
  "target_task": "礼物准备",
  "target_file": "0.json",
  "turn_idx": 0,
  "candidate_model": "model-name",
  "profile": "...optional or omitted...",
  "task_context": "...",
  "dialogue_prefix": [
    {"role": "user", "content": "..."}
  ],
  "candidate_response": "...",
  "source_chat_model": "original-data-model"
}
```

Scored benchmark file:

```json
{
  "sample_id": "User_0__礼物准备__0.json__turn_0",
  "candidate_model": "model-name",
  "judge_model": "Qwen/Qwen3-8B",
  "judge_config": "memory_v2_none",
  "pred_score": 4,
  "reason_prediction": "满意",
  "analysis": "...",
  "user": "User_0",
  "target_task": "礼物准备",
  "target_file": "0.json",
  "turn_idx": 0
}
```

Gold fields can be kept separately for diagnostic correlation against the original assistant, but candidate model benchmarking should not require gold labels.

## Aggregation

Report at least four model-level scores:

1. Micro average:
   - mean predicted satisfaction over all turns
   - sensitive to users/tasks with more turns

2. User macro average:
   - average per-user mean satisfaction
   - treats users equally

3. Task macro average:
   - average per-task mean satisfaction
   - reveals task-specific strengths

4. User-task macro average:
   - average over `(user, target_task)` blocks
   - closest to current personalized block structure

Also report:

- SAT rate: fraction of predicted scores `>=4`
- DSAT rate: fraction of predicted scores `<=3`
- predicted score distribution
- bootstrap confidence intervals over users

## Main Risks

### 1. Predictor noise

Current best predictors are useful but not perfect. The judge should be treated as a noisy learned evaluator, not ground truth.

Mitigation:

- report confidence intervals
- include sanity checks on original human responses
- keep calibration variants separate

### 2. Judge-model bias

If Qwen3 is used as the satisfaction predictor and Qwen-family candidate models are evaluated, there may be style or family bias.

Mitigation:

- later run a second judge model, such as GPT-based memory v2
- report judge agreement
- avoid overclaiming absolute ranking from a single judge

### 3. Static replay mismatch

A candidate response may be incompatible with later human turns, but single-turn static replay intentionally ignores later turns.

Mitigation:

- define benchmark as "one-turn response quality under fixed user context"
- do not claim it measures full interactive dialogue quality

### 4. Calibration drift

Post-hoc calibration improved RecLLMSim prediction, but for candidate model benchmarking it changes score distributions mechanically.

Mitigation:

- report raw judge scores as the primary score
- report calibrated scores as a secondary view
- do not mix raw and calibrated scores in the same leaderboard

## Minimal Implementation Plan

1. Add `trace/collect_static_replay.py`
   - iterate over `PersonalizedSample`
   - build dialogue prefix for each target assistant turn
   - call candidate model to generate `candidate_response`
   - write generated JSONL

2. Add `trace/score_static_replay.py`
   - load generated JSONL
   - build/fetch user memory
   - call the selected satisfaction predictor prompt on `candidate_response`
   - write scored JSONL

3. Add `eval/static_replay.py`
   - aggregate by model, user, task, and user-task block
   - compute score distribution and SAT/DSAT rate
   - bootstrap CIs over users

4. Add scripts:
   - `scripts/collect_static_replay.sh`
   - `scripts/score_static_replay.sh`
   - `scripts/eval_static_replay.sh`

## Conclusion

The current design can support Static Replay Evaluation, but it should be treated as a new benchmark layer above the existing predictor.

The recommended first version is:

- single-turn static replay
- frozen `memory_v2 none` judge
- raw predicted satisfaction as the primary score
- calibrated scores and adaptive-memory judges as secondary analyses

This keeps model comparison clean while still leveraging the user-specific memory predictor developed in the project.
