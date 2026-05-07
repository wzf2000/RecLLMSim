# Detection Python Refactor Plan

## Purpose

This note audits the Python scripts under `detection/` and proposes a
logic-preserving refactor plan.  The main constraint is backward compatibility:
existing commands such as `python trace/collect_personalized.py ...` and shell
wrappers under `detection/scripts/` must keep working.

## Current Size Hotspots

Line-count scan on 2026-05-07:

| file | lines | role |
|---|---:|---|
| `detection/lib/memory.py` | 3547 | memory schemas, memory prompts, update prompts, merge logic, turn-eval prompts |
| `detection/trace/collect_personalized.py` | 2705 before refactor | personalized inference pipeline CLI |
| `detection/eval/spur.py` | 952 | SPUR-style rubric extraction, scoring, embedding classifier |
| `detection/trace/sft.py` | 686 | SFT data selection, prompt formatting, training |
| `detection/eval/personalized.py` | 667 | personalized result metrics and comparisons |
| `detection/tools/annotation_app.py` | 581 | annotation UI |
| `detection/eval/turn_content_filter.py` | 561 | LLM turn-content annotation and grouped analysis |
| `detection/eval/analysis.py` | 557 | static replay / satisfaction result diagnostics |
| `detection/trace/collect_urs.py` | 551 | URS personalized pipeline |
| `detection/trace/collect_api.py` | 516 | API collection pipeline |

The two highest-value targets are `lib/memory.py` and
`trace/collect_personalized.py`.  Other files are long but mostly single-purpose
CLIs; splitting them is useful only after the shared modules stabilize.

## Dependency Constraints

`detection/lib/memory.py` is a public internal API.  It is imported by:

- `detection/trace/collect_personalized.py`
- `detection/trace/score_static_replay.py`
- `detection/trace/collect_urs.py`
- `detection/lib/urs_memory.py`

Therefore `lib.memory` should remain importable as a compatibility facade even
if implementation moves to smaller modules.

`detection/trace/collect_personalized.py` is also reused as a module, not only
as a script:

- `detection/trace/collect_urs.py` imports `StructuredOutputError`,
  `TurnPrediction`, and `_structured_parse`.
- `detection/trace/collect_uss.py` imports `load_finished_ids` and
  `run_agent_on_sample`.
- `detection/trace/score_static_replay.py` imports the module as
  `trace.collect_personalized` and mutates its global client for vLLM mode.

So `trace.collect_personalized` must also remain as the backward-compatible
entry point.

## Recommended Module Layout

Suggested target layout:

```text
detection/
  lib/
    memory.py                         # compatibility facade / public imports
    memory_schema.py                  # ScoreDistribution, UserMemory*, patch models
    memory_formatting.py              # shared truncation/formatting helpers
    memory_build_prompts.py           # build_memory_prompt, build_memory_prompt_v3
    memory_update_prompts.py          # build_memory_update_prompt*
    memory_update_merge.py            # merge_memory_v2_*_patch
    memory_eval_prompts.py            # build_turn_eval_prompt* and no-memory prompts
  trace/
    collect_personalized.py           # CLI + orchestration facade
    structured_output.py              # shared structured-output parsing
    personalized_predictions.py       # response schemas and reconstruction logic
    personalized_memory.py            # build/update memory calls and cache handling
    personalized_turn_eval.py         # turn prediction flows and session evaluation
  eval/
    turn_content/
      __init__.py
      annotation.py                   # annotation prompt/call/load/save
      analysis.py                     # grouped metric analysis
      cli.py                          # argparse commands
```

The compatibility files should re-export old names so existing imports and
commands continue to work.

## Refactor Order

### Step 1: Extract structured-output parsing

Status: implemented in this session.

Moved the common parse/recovery helpers from
`detection/trace/collect_personalized.py` to
`detection/trace/structured_output.py`.

Backward compatibility preserved by keeping wrapper functions in
`trace.collect_personalized`:

- `_structured_parse(...)`
- `_structured_parse_from_raw_text(...)`
- `StructuredOutputError`

This is low risk because it does not change prompts, schemas, API parameters,
retry behavior, output fields, or CLI arguments.

Verification:

```bash
PYTHONPYCACHEPREFIX=/tmp/rec_pycache python -m py_compile \
  detection/trace/structured_output.py \
  detection/trace/collect_personalized.py \
  detection/trace/collect_urs.py \
  detection/trace/collect_uss.py \
  detection/trace/score_static_replay.py

cd detection
python -c "from trace.collect_personalized import StructuredOutputError, TurnPrediction, _structured_parse; from trace import collect_personalized as base; from trace.structured_output import structured_parse; print(StructuredOutputError.__name__, TurnPrediction.__name__, callable(_structured_parse), callable(structured_parse), hasattr(base, 'client'))"
```

### Step 2: Extract personalized prediction schemas

Status: implemented in this session.

Moved the following from `trace.collect_personalized` to
`trace/personalized_predictions.py`:

- `TurnPrediction`
- `BoundaryTurnPrediction`
- `SelectiveBoundaryTurnPrediction`
- `HistoryPriorDeltaPrediction`
- `HistoryPriorDeltaV2Prediction`
- `SatRefinementPrediction`
- `DsatRefinementPrediction`
- `_normalize_pred_reason`
- `_clip_score`
- `_reconstruct_history_prior_delta_score`
- `_reconstruct_history_prior_delta_v2_score`
- `_history_prior_delta_v3_dsat_votes`
- `_reconstruct_history_prior_delta_v3_score`
- `_reconstruct_history_prior_delta_v3_1_score`
- `_retrieve_anchor_turns`
- `_anchor_metadata`

Keep these names imported into `trace.collect_personalized` for callers such as
`collect_urs.py`.

Backward compatibility is preserved by importing the same names back into
`trace.collect_personalized`, so old imports such as
`from trace.collect_personalized import TurnPrediction` still work.

Verification:

```bash
PYTHONPYCACHEPREFIX=/tmp/rec_pycache python -m py_compile \
  detection/trace/personalized_predictions.py \
  detection/trace/structured_output.py \
  detection/trace/collect_personalized.py \
  detection/trace/collect_urs.py \
  detection/trace/collect_uss.py \
  detection/trace/score_static_replay.py

cd detection
python -c "from trace.collect_personalized import TurnPrediction, BoundaryTurnPrediction, HistoryPriorDeltaV2Prediction, _normalize_pred_reason, _history_prior_delta_v3_dsat_votes, _retrieve_anchor_turns, _structured_parse; from trace.personalized_predictions import TurnPrediction as TP; print(TurnPrediction is TP, BoundaryTurnPrediction.__name__, HistoryPriorDeltaV2Prediction.__name__, callable(_normalize_pred_reason), callable(_history_prior_delta_v3_dsat_votes), callable(_retrieve_anchor_turns), callable(_structured_parse))"

python trace/collect_personalized.py --help
```

Another `collect_personalized.py` subsplit was completed for turn evaluation:

Moved turn-level evaluation flow to `trace/personalized_turn_eval.py`:

- `_call_predict_turn`
- `_should_trigger_selective_refute`
- `_predict_turn_with_optional_selective_refute`
- `_predict_turn_fullscale_from_boundary_v2`
- `_predict_turn_v3_two_stage`
- `_predict_turn_v3_two_stage_v2`
- `evaluate_session`

`trace.collect_personalized` keeps wrappers with the old names and signatures.
The wrappers inject `_structured_parse` and `_structured_parse_from_raw_text`,
so module-level `client` replacement for vLLM remains compatible.

Verification:

```bash
PYTHONPYCACHEPREFIX=/tmp/rec_pycache python -m py_compile \
  detection/trace/personalized_turn_eval.py \
  detection/trace/personalized_memory.py \
  detection/trace/personalized_predictions.py \
  detection/trace/structured_output.py \
  detection/trace/collect_personalized.py \
  detection/trace/collect_urs.py \
  detection/trace/collect_uss.py \
  detection/trace/score_static_replay.py

cd detection
python -c "from trace.collect_personalized import _call_predict_turn, _predict_turn_with_optional_selective_refute, _predict_turn_v3_two_stage, evaluate_session, TurnPrediction; from trace import collect_personalized as base; from trace import personalized_turn_eval as te; print(callable(_call_predict_turn), callable(_predict_turn_with_optional_selective_refute), callable(_predict_turn_v3_two_stage), callable(evaluate_session), hasattr(base, 'client'), callable(te.evaluate_session), TurnPrediction.__name__)"

python trace/collect_personalized.py --help
```

### Step 3: Split `lib.memory` behind a facade

Before starting on `lib.memory`, an additional `collect_personalized.py`
subsplit was completed:

Moved memory build/update helpers to `trace/personalized_memory.py`:

- `_call_build_memory`
- `build_user_memory`
- `_call_update_memory`
- `_resolve_memory_update_prompt_version`
- `update_memory`

`trace.collect_personalized` still exposes wrapper functions with the old
names/signatures, and the wrappers pass the module-level `_structured_parse`
function into `personalized_memory.py`.  This preserves the existing behavior
where other scripts mutate `trace.collect_personalized.client` for vLLM mode.

Verification:

```bash
PYTHONPYCACHEPREFIX=/tmp/rec_pycache python -m py_compile \
  detection/trace/personalized_memory.py \
  detection/trace/personalized_predictions.py \
  detection/trace/structured_output.py \
  detection/trace/collect_personalized.py \
  detection/trace/collect_urs.py \
  detection/trace/collect_uss.py \
  detection/trace/score_static_replay.py

cd detection
python -c "from trace.collect_personalized import build_user_memory, update_memory, _call_build_memory, _call_update_memory; from trace import collect_personalized as base; from trace import personalized_memory as mem; print(callable(build_user_memory), callable(update_memory), callable(_call_build_memory), callable(_call_update_memory), hasattr(base, 'client'), callable(mem.build_user_memory))"

python trace/collect_personalized.py --help
```

The safest approach is to create new modules first and keep `lib/memory.py` as
the public facade.  Do not change callers in the same commit.

Recommended split:

- `memory_schema.py`: all Pydantic schemas and small schema-only helpers.
- `memory_formatting.py`: `_truncate`, score-group formatting, reason-rule
  formatting, update evidence formatting.
- `memory_build_prompts.py`: initial memory building prompts.
- `memory_update_prompts.py`: update prompt builders.
- `memory_update_merge.py`: patch merge functions.
- `memory_eval_prompts.py`: all turn-eval prompt builders.

Then `lib/memory.py` should import and re-export the same public symbols.  This
preserves `from lib.memory import ...` while making future edits localized.

### Step 4: Split secondary CLIs only after shared modules settle

Good candidates:

- `eval/turn_content_filter.py` into `eval/turn_content/{annotation,analysis,cli}.py`
- `eval/spur.py` into rubric extraction, scoring, embeddings/classifier, CLI
- `trace/sft.py` into prompt parsing, dataset building, trainer wrapper, CLI

These are lower priority because they are less reused by other modules.

## Risks and Guardrails

- Do not rename current CLI entry files.
- Do not change existing argparse option names or defaults.
- Do not change output JSONL field names.
- For `collect_personalized.py`, preserve module-level `client` and `_is_vllm`
  behavior because other scripts mutate that module state.
- For `lib.memory`, preserve all existing imports from `lib.memory` until every
  downstream caller is migrated and verified.
- Prefer one mechanical extraction per commit, followed by `py_compile` and
  import smoke tests.

## Suggested Commit Batches

1. `[trace] refactor: extract structured output parsing`
2. `[trace] refactor: extract personalized prediction models`
3. `[lib/memory] refactor: split memory schemas`
4. `[lib/memory] refactor: split memory update logic`
5. `[lib/memory] refactor: split turn eval prompts`
6. `[eval] refactor: split turn content filter`
