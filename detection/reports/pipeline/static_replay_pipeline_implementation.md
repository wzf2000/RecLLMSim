# Static Replay Pipeline Implementation

## Goal

This report records the initial coding base for Static Replay Evaluation.

The pipeline evaluates candidate LLM responses under fixed historical dialogue
prefixes, then scores those responses with a user-specific satisfaction judge.
Both the replay candidate model and the satisfaction judge can be either an
OpenAI API model or an OpenAI-compatible vLLM endpoint.

## Implemented Components

### Candidate response collection

File: `detection/trace/collect_static_replay.py`

The collector iterates over `PersonalizedSample.target_sessions`. For each
original assistant turn, it keeps the dialogue prefix before that turn, asks a
candidate model to generate one assistant reply, and writes one JSONL record.

The generated reply is not rolled into later turns. This is a single-turn static
replay setting, so all candidate models are evaluated on the same original
dialogue prefixes.

Output fields include:

- `sample_id`
- `user`
- `target_task`
- `target_file`
- `turn_idx`
- `candidate_model`
- `task_context`
- `dialogue_prefix`
- `candidate_response`
- `source_assistant_reply`
- `source_chat_model`
- `gold_score`
- `gold_reason`

`gold_score` and `gold_reason` are kept only as diagnostics against the original
human-labeled turn. They are not required for benchmark scoring.

### Satisfaction judge scoring

File: `detection/trace/score_static_replay.py`

The scorer loads the generated JSONL records, rebuilds the corresponding
personalized samples, builds or loads user memory, and calls the selected
satisfaction predictor on each candidate response.

The first recommended judge configuration is:

- `memory_version=v2`
- `turn_eval_prompt_version=v2`
- no memory update during scoring

This keeps the judge state frozen for every candidate model. Adaptive update
variants can be added later, but they should be reported as separate benchmark
tracks because the judge state would become model-dependent.

Output fields preserve the generation record and add:

- `judge_model`
- `judge_config`
- `judge_memory_version`
- `judge_turn_eval_prompt_version`
- `pred_score`
- `reason_prediction`
- `analysis`

### Benchmark aggregation

File: `detection/eval/static_replay.py`

The evaluator groups scored JSONL records by `candidate_model` and reports:

- `micro_mean`
- `user_macro_mean`
- `task_macro_mean`
- `user_task_macro_mean`
- `sat_rate`
- `dsat_rate`
- `score_distribution`
- `task_means`
- user-level bootstrap confidence interval

If `gold_score` is available, it also reports
`diagnostic_mae_vs_original_gold`. This is only a diagnostic view because
candidate responses do not have direct human labels.

## Shell Scripts

The following wrappers were added:

- `detection/scripts/collect_static_replay.sh`
- `detection/scripts/score_static_replay.sh`
- `detection/scripts/eval_static_replay.sh`

They use `python` from the active environment and do not hardcode local Python
paths.

## Backend Usage

### Candidate collection with vLLM

Run from `detection/`:

```bash
model=Qwen/Qwen3-8B \
base_url=http://localhost:8000/v1 \
api_key=EMPTY \
limit_users=1 \
output_jsonl=outputs/static_replay/qwen3_8b_test_responses_u1.jsonl \
bash scripts/collect_static_replay.sh
```

### Candidate collection with OpenAI API

Run from `detection/`:

```bash
model=gpt-4o-mini \
limit_users=1 \
output_jsonl=outputs/static_replay/gpt4o_mini_test_responses_u1.jsonl \
bash scripts/collect_static_replay.sh
```

This uses the default project OpenAI client configuration from `lib.llm`.

### Judge scoring with vLLM

Run from `detection/`:

```bash
input_jsonl=outputs/static_replay/qwen3_8b_test_responses_u1.jsonl \
judge_model=Qwen/Qwen3-8B \
judge_base_url=http://localhost:8000/v1 \
judge_api_key=EMPTY \
judge_config=qwen3_memv2_none \
memory_version=v2 \
turn_eval_prompt_version=v2 \
output_jsonl=outputs/static_replay/qwen3_8b_test_scored_by_qwen3_memv2.jsonl \
bash scripts/score_static_replay.sh
```

### Judge scoring with OpenAI API

Run from `detection/`:

```bash
input_jsonl=outputs/static_replay/gpt4o_mini_test_responses_u1.jsonl \
judge_model=gpt-4o-mini \
judge_config=gpt4o_mini_memv2_none \
memory_version=v2 \
turn_eval_prompt_version=v2 \
output_jsonl=outputs/static_replay/gpt4o_mini_test_scored_by_gpt4o_mini_memv2.jsonl \
bash scripts/score_static_replay.sh
```

### Aggregation

Run from `detection/`:

```bash
input_jsonl=outputs/static_replay/qwen3_8b_test_scored_by_qwen3_memv2.jsonl \
output_json=outputs/static_replay/qwen3_8b_static_replay_eval.json \
bash scripts/eval_static_replay.sh
```

## Current Limitations

The initial scorer is intended for direct turn-evaluation prompt versions such
as `v2`. More complex high-level flows, such as multi-stage refinement wrappers,
should be integrated explicitly before being used as static replay judges.

The initial benchmark score is the raw predicted satisfaction score. Calibration
or multi-judge ensembling should be added as separate post-processing layers so
the raw judge behavior remains auditable.

No full model smoke test is recorded in this report. The implemented scripts
were validated with Python compilation and CLI help checks.
