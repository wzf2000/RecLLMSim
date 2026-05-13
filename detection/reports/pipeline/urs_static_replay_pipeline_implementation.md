# URS Static Replay Pipeline Implementation

## Scope

This implementation adds an auxiliary URS static replay pipeline alongside the
main turn-level static replay benchmark. URS is treated as a session-level
dataset, so each replay item is one URS session rather than one assistant turn.

## Added Entry Points

### Satisfaction Predictor Evaluation

Use the existing URS predictor outputs and evaluate them with the shared
personalized metrics:

```bash
cd detection
result_file=outputs/urs/Qwen_Qwen3-8B_test_none_rerun.jsonl \
output_json=outputs/urs/Qwen_Qwen3-8B_test_none_rerun_eval.json \
bash scripts/eval_urs_predictor.sh
```

This evaluates predictor quality against URS gold labels. The four most useful
diagnostic metrics are Pearson, Spearman, QWK, and F1-DSAT.

### URS Static Replay Collection

Collect target-model responses for selected URS sessions:

```bash
cd detection
model=gpt-5.4-nano \
selection_mode=hard \
replay_context_mode=raw \
replay_granularity=first_user \
hard_global_budget=300 \
max_tokens=4096 \
max_workers=4 \
output_jsonl=outputs/urs_static_replay/gpt-5.4-nano_test_hard_raw_first_user_responses.jsonl \
bash scripts/collect_urs_static_replay.sh
```

Memory-augmented collection:

```bash
model=gpt-5.4-nano \
selection_mode=hard \
replay_context_mode=dialogue_memory_tfidf \
replay_granularity=first_user \
dialogue_memory_top_k=4 \
hard_global_budget=300 \
max_tokens=4096 \
max_workers=4 \
output_jsonl=outputs/urs_static_replay/gpt-5.4-nano_test_hard_dialogue_memory_tfidf_first_user_responses.jsonl \
bash scripts/collect_urs_static_replay.sh
```

Available replay context modes:

- `raw`
- `dialogue_memory_tfidf`
- `dialogue_memory_diverse`

Available replay granularities:

- `first_user`: use only the first user request and generate one assistant
  response. This is the recommended first setting.
- `last_user`: keep the prefix before the original final assistant response and
  regenerate that response. This is closer to turn-level replay but less clean
  under URS session-level labels.

### URS Static Replay Scoring

Score collected replay responses with the URS session-level predictor:

```bash
input_jsonl=outputs/urs_static_replay/gpt-5.4-nano_test_hard_raw_first_user_responses.jsonl \
judge_model=Qwen/Qwen3-8B \
judge_base_url=http://localhost:8000/v1 \
judge_api_key=EMPTY \
memory_cache_dir=outputs/urs/memory_cache \
max_workers=4 \
output_jsonl=outputs/urs_static_replay/gpt-5.4-nano_test_hard_raw_first_user_scored_by_qwen3_8b.jsonl \
bash scripts/score_urs_static_replay.sh
```

The scorer reconstructs a synthetic URS session from:

- `dialogue_prefix`
- `candidate_response`

and then calls the existing URS `evaluate_urs_session` logic.

### URS Static Replay Benchmark Evaluation

Evaluate scored replay outputs as benchmark scores:

```bash
input_jsonl=outputs/urs_static_replay/gpt-5.4-nano_test_hard_raw_first_user_scored_by_qwen3_8b.jsonl \
output_json=outputs/urs_static_replay/gpt-5.4-nano_test_hard_raw_first_user_eval.json \
bash scripts/eval_urs_static_replay.sh
```

The output reports:

- micro mean predicted satisfaction
- user-macro mean
- task/intent-macro mean
- user-intent block macro mean
- SAT / DSAT rate
- score distribution
- language means
- replay context / granularity means

These are judge-predicted benchmark scores, not gold accuracy metrics.

## Implementation Files

- `detection/trace/collect_urs_static_replay.py`
- `detection/trace/score_urs_static_replay.py`
- `detection/eval/urs_static_replay.py`
- `detection/scripts/collect_urs_static_replay.sh`
- `detection/scripts/score_urs_static_replay.sh`
- `detection/scripts/eval_urs_static_replay.sh`
- `detection/scripts/eval_urs_predictor.sh`

## Design Notes

The replay collector uses the same URS cross-intent split as
`collect_urs.sh`. For memory context modes, it retrieves raw historical
dialogue episodes from the same user's other intents using
`detection/lib/dialogue_memory.py`. Satisfaction labels, dissatisfaction
reasons, and user profile fields are not exposed to the candidate model.

The recommended first benchmark configuration is:

- `selection_mode=hard`
- `hard_global_budget=300`
- `replay_granularity=first_user`
- `replay_context_mode=raw`, then compare against
  `dialogue_memory_tfidf` and `dialogue_memory_diverse`
- judge with `Qwen/Qwen3-8B`
- no post-hoc calibration by default

## Caveats

The current URS predictor is noisier than the main personalized predictor.
Existing URS Qwen3-8B `none` results show moderate global signal but weak
user-aware metrics. URS static replay should therefore be treated as auxiliary
evidence for cross-dataset robustness rather than the primary benchmark.
