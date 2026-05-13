# URS As An Auxiliary Static Replay Benchmark

## Goal

The current static replay benchmark is built on the project's own turn-level
personalized satisfaction dataset. The question is whether URS can be used as a
secondary evaluation dataset for the same benchmark idea:

1. Let a target LLM generate responses under fixed historical contexts.
2. Use a personalized satisfaction predictor to score the generated responses.
3. Compare LLMs by predicted user-specific satisfaction.

## Current URS Pipeline Status

The repository already contains a URS personalized satisfaction prediction
pipeline:

- `detection/lib/urs_data.py`
  - Loads `detection/data/urs/chinese_merged.json` and
    `detection/data/urs/english_processed.json`.
  - Maps zh/en intent labels into canonical intent slugs.
  - Maps URS user satisfaction labels to 1-5 scores.
  - Builds `PersonalizedSample` objects using cross-intent split.
- `detection/lib/urs_memory.py`
  - Builds session-level memory and session-level scoring prompts.
  - Scores the whole dialogue once, not each assistant turn.
- `detection/trace/collect_urs.py`
  - Runs URS session-level personalized prediction.
- `detection/scripts/collect_urs.sh`
  - Shell entrypoint for URS collection.
- `detection/scripts/eval_urs.sh`
  - Reuses `eval/personalized.py` for metrics.

The existing URS predictor run with `Qwen/Qwen3-8B`, memory update mode `none`,
has:

- `584` test sessions.
- `116` users.
- `339` user-intent blocks.
- Moderate global signal: Pearson `0.2830`, Spearman `0.2781`, QWK `0.2498`.
- Boundary F1-DSAT `0.5411`.
- Weak user-aware metrics, including negative user-aware correlations.

This means the URS predictor is usable as a noisy auxiliary judge, but it is
less reliable than the main personalized benchmark.

## Key Difference From The Main Benchmark

The main dataset is turn-level:

- Each assistant turn has one satisfaction score.
- Static replay naturally replaces one assistant response at a selected turn.
- The satisfaction predictor evaluates that single response in context.

URS is session-level:

- Each full conversation has one satisfaction score.
- The current loader stores the score as `SessionData.satisfaction_scores=[score]`
  only for compatibility.
- There is no gold turn-level score.

Therefore, URS cannot support the exact same turn-level replay semantics without
creating pseudo labels. The more defensible design is session-level replay.

## Feasibility Assessment

### Feasible

URS can support a session-level static replay benchmark:

1. Use the original URS conversation prefix or user prompt as the fixed input.
2. Let the target LLM generate an assistant response or a full assistant-side
   continuation.
3. Construct a replayed session.
4. Score the replayed session with the URS session-level satisfaction predictor.
5. Compare models by predicted session satisfaction.

This is compatible with the existing URS `SessionData` abstraction and the
existing `build_session_eval_prompt` / `build_session_eval_prompt_no_memory`
scoring prompts.

### Partially Feasible

Memory-augmented candidate generation is also feasible:

- URS has multiple sessions per user and multiple intents.
- For each target intent, `history_sessions` already contains other-intent
  sessions from the same user.
- These can be used as unlabeled dialogue memory, analogous to the new
  `dialogue_memory_tfidf` / `dialogue_memory_diverse` modes in the main static
  replay pipeline.

However, URS has no explicit profile and no turn-level satisfaction annotation,
so memory should only include raw dialogue history, not labels.

### Not Recommended

Turn-level replay on URS is not recommended as the main setting:

- It would require choosing one assistant turn inside a multi-turn URS session.
- The only gold label is for the whole session.
- Replacing one turn while keeping later original turns creates inconsistent
  dialogue state.
- Replaying every assistant turn would be expensive and hard to interpret.

If turn-level replay is needed for analysis, it should be described as an
exploratory pseudo-turn setting, not as the primary URS benchmark.

## Proposed URS Auxiliary Benchmark Design

### Unit

Use one URS session as one benchmark item.

Each item contains:

- `user`
- `target_intent`
- `title` / `task_context`
- original conversation history
- original session-level satisfaction score
- target model generated response or generated continuation
- judge-predicted replay satisfaction score

### Replay Input Modes

Use three candidate-visible modes:

1. `raw`
   - Candidate model sees only the session's initial user request or selected
     dialogue prefix.
   - This is the vanilla LLM baseline.

2. `dialogue_memory_tfidf`
   - Candidate model sees retrieved raw dialogue memories from the same user and
     other intents.
   - Retrieval is based on lexical/semantic similarity between the current
     session context and historical sessions.

3. `dialogue_memory_diverse`
   - Candidate model sees same-user historical sessions selected with intent
     diversity.
   - This tests whether broad user history helps more than nearest-neighbor
     similarity.

The memory modes should exclude:

- satisfaction score
- dissatisfaction reason
- user profile
- any gold annotation

### Replay Granularity

Recommended first version:

**single-response replay from the first user request**.

For each URS session:

1. Extract the first user message.
2. Let the target LLM produce one assistant response.
3. Build a synthetic two-turn session:
   - original first user message
   - generated assistant response
4. Score this synthetic session with the URS session-level predictor.

This is cheaper and cleaner than multi-turn continuation. It evaluates whether
the model can produce a satisfying initial answer for a real user request.

Potential second version:

**prefix replay at the last user turn**.

For each URS session:

1. Keep the original dialogue prefix up to the final user message.
2. Replace only the final assistant response with the candidate response.
3. Score the replayed full session.

This is closer to the main static replay setup, but it has higher risk:
session-level gold and predictor judgments may be dominated by earlier turns
that are unchanged.

### Selection Strategy

Do not replay all URS sessions initially. Use a balanced hard subset:

- Include all scores `1/2/3` up to a cap.
- Include score `4` boundary cases.
- Include a smaller number of score `5` positive controls.
- Balance by language (`zh`, `en`) and intent.
- Ensure each selected user contributes at most a small number of sessions.

Recommended initial scale:

- `100-150` URS sessions for smoke evaluation.
- `300-400` sessions for a stable auxiliary benchmark.

This is appropriate because URS test only has `584` sessions.

### Scoring

Use the existing URS session-level predictor:

- Primary judge: `Qwen/Qwen3-8B`, memory update mode `none`.
- Do not use URS CDF calibration as the default, because current calibration
  coverage is poor and degrades global metrics.
- Report raw predicted scores first.

If calibration is needed, use a URS-specific user-level or language-level
calibration rather than the current block-level CDF.

### Metrics

For each target LLM:

- Mean predicted satisfaction.
- User-macro mean predicted satisfaction.
- Intent-macro mean predicted satisfaction.
- SAT rate: predicted score `>=4`.
- DSAT rate: predicted score `<=3`.
- Pairwise win/tie/lose versus a reference model.

Because replayed responses do not have gold satisfaction labels, these metrics
are benchmark scores rather than predictor accuracy metrics.

For validating the judge itself, continue reporting:

- Pearson
- Spearman
- QWK
- F1-DSAT

on the original URS gold-labeled sessions.

## Recommended Implementation Plan

### Step 1: Add URS Static Replay Collection

Create a separate script rather than overloading the current turn-level static
replay script:

- `detection/trace/collect_urs_static_replay.py`
- `detection/scripts/collect_urs_static_replay.sh`

The script should:

1. Load `build_urs_personalized_samples`.
2. Select sessions according to `selection_mode`.
3. Build candidate messages using `raw` or dialogue memory mode.
4. Generate candidate response.
5. Write replay JSONL.

Suggested output path:

- `outputs/urs_static_replay/{model}_{split}_{selection}_{context}_responses.jsonl`

### Step 2: Add URS Replay Scoring

Create:

- `detection/trace/score_urs_static_replay.py`
- `detection/scripts/score_urs_static_replay.sh`

The scorer should:

1. Read URS replay responses.
2. Reconstruct a synthetic session.
3. Build or load URS user memory.
4. Call `evaluate_urs_session`.
5. Write scored JSONL compatible with static replay evaluation.

### Step 3: Add URS Replay Evaluation

Create:

- `detection/eval/urs_static_replay.py`
- `detection/scripts/eval_urs_static_replay.sh`

The first evaluator can reuse the main static replay summary logic, but should
label metrics as judge-predicted benchmark scores rather than gold accuracy.

### Step 4: Add Pairwise Comparison

Reuse or lightly adapt:

- `detection/eval/static_replay_pairwise.py`

Use the same selected URS item ids for all candidate models.

## Main Risks

1. The current URS predictor is much noisier than the main predictor.
   It should be treated as an auxiliary judge only.
2. URS session-level labels make turn-level replay hard to justify.
   Session-level replay is cleaner.
3. URS calibration currently does not transfer well.
   Raw judge scores should be the default baseline.
4. URS has no explicit user profile and weaker user-aware signal.
   Memory-augmented candidate generation should use raw dialogue memories only.
5. Multi-turn session replay may require rolling generation and can become
   expensive. Start with single-response replay.

## Recommendation

Use URS as a secondary benchmark to test cross-dataset robustness of the static
replay evaluation idea, not as the primary evidence for personalized
satisfaction prediction.

The most defensible first setting is:

- session-level URS static replay
- first-user-message single-response generation
- `raw`, `dialogue_memory_tfidf`, and `dialogue_memory_diverse` context modes
- Qwen3-8B URS predictor with memory update mode `none`
- raw judge scores without CDF calibration
- pairwise win/tie/lose as the main model-comparison view
