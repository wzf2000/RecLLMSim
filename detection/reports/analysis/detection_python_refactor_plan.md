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

The turn-evaluation facade was then split further:

- `personalized_turn_eval.py`: compatibility facade
- `personalized_turn_api.py`: structured prediction API call and parse-failure dumping
- `personalized_turn_selective.py`: normal turn prediction and selective refute
- `personalized_turn_two_stage.py`: fullscale and two-stage prediction flows
- `personalized_session_eval.py`: session-level assistant-turn loop

Verification:

```bash
PYTHONPYCACHEPREFIX=/tmp/rec_pycache python -m py_compile \
  detection/trace/personalized_turn_eval.py \
  detection/trace/personalized_turn_api.py \
  detection/trace/personalized_turn_selective.py \
  detection/trace/personalized_turn_two_stage.py \
  detection/trace/personalized_session_eval.py \
  detection/trace/collect_personalized.py \
  detection/trace/collect_urs.py \
  detection/trace/collect_uss.py \
  detection/trace/score_static_replay.py

cd detection
python -c "from trace.personalized_turn_eval import call_predict_turn, predict_turn_with_optional_selective_refute, predict_turn_v3_two_stage, evaluate_session; from trace.collect_personalized import _call_predict_turn, evaluate_session as eval2; print(callable(call_predict_turn), callable(predict_turn_with_optional_selective_refute), callable(predict_turn_v3_two_stage), callable(evaluate_session), callable(_call_predict_turn), callable(eval2))"

python trace/collect_personalized.py --help
```

### Step 3: Split `lib.memory` behind a facade

Status: implemented in this session.

`detection/lib/memory.py` is now a compatibility facade.  It re-exports the
same public symbols used by downstream callers, while implementation lives in
smaller modules:

- `memory_schema.py`: all Pydantic schemas and small schema-only helpers.
- `memory_formatting.py`: `_truncate`, score-group formatting, reason-rule
  formatting, profile formatting, and session score grouping helpers.
- `memory_build_prompts.py`: initial memory building prompts.
- `memory_update_prompts.py`: update prompt builders and update evidence
  bundle helpers.
- `memory_update_merge.py`: patch merge functions.
- `memory_eval_prompts.py`: all turn-eval prompt builders.

This preserves imports such as:

- `from lib.memory import UserMemory`
- `from lib.memory import build_memory_prompt`
- `from lib.memory import build_turn_eval_prompt`
- `from lib.memory import merge_memory_v2_5_patch`

Line-count after split:

| file | lines |
|---|---:|
| `detection/lib/memory.py` | 78 |
| `detection/lib/memory_schema.py` | 398 |
| `detection/lib/memory_formatting.py` | 110 |
| `detection/lib/memory_build_prompts.py` | 209 |
| `detection/lib/memory_update_prompts.py` | 691 |
| `detection/lib/memory_update_merge.py` | 453 |
| `detection/lib/memory_eval_prompts.py` | 1717 |

Verification:

```bash
PYTHONPYCACHEPREFIX=/tmp/rec_pycache python -m py_compile \
  detection/lib/memory_schema.py \
  detection/lib/memory_formatting.py \
  detection/lib/memory_build_prompts.py \
  detection/lib/memory_update_prompts.py \
  detection/lib/memory_update_merge.py \
  detection/lib/memory_eval_prompts.py \
  detection/lib/memory.py

cd detection
python -c "from lib.memory import UserMemory, UserMemoryContent, UserMemoryV3, MemoryUpdatePatchV2_1, build_memory_prompt, build_memory_prompt_v3, build_memory_update_prompt_v2_5, merge_memory_v2_5_patch, build_turn_eval_prompt, build_turn_eval_prompt_no_memory; print(UserMemory.__name__, UserMemoryContent.__name__, UserMemoryV3.__name__, MemoryUpdatePatchV2_1.__name__, callable(build_memory_prompt), callable(build_memory_prompt_v3), callable(build_memory_update_prompt_v2_5), callable(merge_memory_v2_5_patch), callable(build_turn_eval_prompt), callable(build_turn_eval_prompt_no_memory))"

PYTHONPYCACHEPREFIX=/tmp/rec_pycache python -m py_compile \
  detection/trace/personalized_memory.py \
  detection/trace/personalized_turn_eval.py \
  detection/trace/collect_personalized.py \
  detection/trace/collect_urs.py \
  detection/trace/score_static_replay.py \
  detection/lib/urs_memory.py

cd detection
python trace/collect_personalized.py --help
```

### Step 4: Split turn content filter

Status: implemented after the shared `trace` and `lib.memory` modules were
stabilized.

Moved the turn-content annotation and analysis script into a focused package:

- `detection/eval/turn_content/io.py`: JSONL load/save helpers.
- `detection/eval/turn_content/annotation.py`: target turn extraction,
  annotation prompt building, LLM calls, resume handling, and annotate command.
- `detection/eval/turn_content/analysis.py`: metric calculation, named result
  parsing, grouped analysis, and analyze command.
- `detection/eval/turn_content/cli.py`: argparse definition and vLLM client
  setup.
- `detection/eval/turn_content_filter.py`: compatibility entry point and
  re-export facade.

Line counts after split:

| file | lines |
|---|---:|
| `detection/eval/turn_content_filter.py` | 80 |
| `detection/eval/turn_content/__init__.py` | 29 |
| `detection/eval/turn_content/io.py` | 21 |
| `detection/eval/turn_content/annotation.py` | 351 |
| `detection/eval/turn_content/analysis.py` | 135 |
| `detection/eval/turn_content/cli.py` | 62 |

Backward compatibility:

- `python eval/turn_content_filter.py annotate ...` and
  `python eval/turn_content_filter.py analyze ...` keep using the old entry
  file.
- Existing imports from `eval.turn_content_filter` keep working for the public
  functions and the previously local helper names.
- CLI argument names, defaults, prompts, output fields, retry behavior, and
  metrics logic were kept unchanged.

Verification:

```bash
PYTHONPYCACHEPREFIX=/tmp/rec_pycache python -m py_compile \
  detection/eval/turn_content_filter.py \
  detection/eval/turn_content/__init__.py \
  detection/eval/turn_content/io.py \
  detection/eval/turn_content/annotation.py \
  detection/eval/turn_content/analysis.py \
  detection/eval/turn_content/cli.py

cd detection
PYTHONPYCACHEPREFIX=/tmp/rec_pycache python eval/turn_content_filter.py --help

cd detection
PYTHONPYCACHEPREFIX=/tmp/rec_pycache python -c "from eval.turn_content_filter import TurnContentAnnotation, build_parser, command_annotate, command_analyze; print(TurnContentAnnotation.__name__, callable(build_parser), callable(command_annotate), callable(command_analyze))"

PYTHONPYCACHEPREFIX=/tmp/rec_pycache python -m py_compile $(find detection -name '*.py' -print)
```

### Step 5: Split SPUR workflow

Status: implemented after `turn_content_filter.py`.

Moved the SPUR baseline implementation into a focused package:

- `detection/eval/spur/constants.py`: SAT/DSAT label constants.
- `detection/eval/spur/data.py`: session-to-turn row preprocessing and
  conversation formatting.
- `detection/eval/spur/llm.py`: shared chat-completion helper.
- `detection/eval/spur/rubrics.py`: Phase 1 rubric extraction and Phase 2
  rubric summarization.
- `detection/eval/spur/scoring.py`: Phase 3 rubric-based direct scoring.
- `detection/eval/spur/embeddings.py`: Phase 4 embeddings, rubric features,
  and logistic regression classifier.
- `detection/eval/spur/metrics.py`: metric calculation and metric logging.
- `detection/eval/spur/cli.py`: argparse, cached phase orchestration, and the
  old `main(...)` wrapper.
- `detection/eval/spur.py`: compatibility script entry point and re-export
  facade.

Line counts after split:

| file | lines |
|---|---:|
| `detection/eval/spur.py` | 132 |
| `detection/eval/spur/__init__.py` | 71 |
| `detection/eval/spur/constants.py` | 2 |
| `detection/eval/spur/data.py` | 47 |
| `detection/eval/spur/llm.py` | 28 |
| `detection/eval/spur/rubrics.py` | 210 |
| `detection/eval/spur/scoring.py` | 157 |
| `detection/eval/spur/embeddings.py` | 193 |
| `detection/eval/spur/metrics.py` | 53 |
| `detection/eval/spur/cli.py` | 275 |

Backward compatibility:

- `bash scripts/run_spur.sh ...` still reaches `python eval/spur.py "$@"`.
- `python eval/spur.py --help` still exposes the same CLI options and defaults.
- Existing imports from `eval.spur` resolve through the new package re-exports.
- Prompts, cache filenames, output filenames, metric fields, argparse defaults,
  and phase control flags were kept unchanged.

Verification:

```bash
PYTHONPYCACHEPREFIX=/tmp/rec_pycache python -m py_compile \
  detection/eval/spur.py \
  detection/eval/spur/__init__.py \
  detection/eval/spur/constants.py \
  detection/eval/spur/data.py \
  detection/eval/spur/llm.py \
  detection/eval/spur/rubrics.py \
  detection/eval/spur/scoring.py \
  detection/eval/spur/metrics.py \
  detection/eval/spur/embeddings.py \
  detection/eval/spur/cli.py

cd detection
PYTHONPYCACHEPREFIX=/tmp/rec_pycache python eval/spur.py --help

cd detection
PYTHONPYCACHEPREFIX=/tmp/rec_pycache python -c "from eval.spur import preprocess_to_rows, parse_args, compute_metrics, extract_rubric_candidates; print(callable(preprocess_to_rows), callable(parse_args), callable(compute_metrics), callable(extract_rubric_candidates))"

PYTHONPYCACHEPREFIX=/tmp/rec_pycache python -m py_compile $(find detection -name '*.py' -print)
```

### Step 6: Split SFT workflow

Status: implemented after the SPUR split.

Moved the SFT data formatting and training script into a focused package:

- `detection/trace/sft/parsing.py`: Qwen think markers and generated JSON
  parsing helpers.
- `detection/trace/sft/targets.py`: assistant target formatting and reflection
  target selection.
- `detection/trace/sft/prompts.py`: history splitting, collect-compatible
  prompt construction, source text construction, and prompt budget resolution.
- `detection/trace/sft/selection.py`: JSONL loading and SFT trace filtering.
- `detection/trace/sft/dataset.py`: tokenization and grouped train/valid
  dataset construction.
- `detection/trace/sft/training.py`: LoRA model setup, trainer setup, and
  `train_sft(...)`.
- `detection/trace/sft/cli.py`: argparse and CLI dispatch.
- `detection/trace/sft.py`: compatibility script entry point and re-export
  facade.

Line counts after split:

| file | lines |
|---|---:|
| `detection/trace/sft.py` | 104 |
| `detection/trace/sft/__init__.py` | 63 |
| `detection/trace/sft/parsing.py` | 143 |
| `detection/trace/sft/targets.py` | 75 |
| `detection/trace/sft/prompts.py` | 134 |
| `detection/trace/sft/selection.py` | 66 |
| `detection/trace/sft/dataset.py` | 126 |
| `detection/trace/sft/training.py` | 121 |
| `detection/trace/sft/cli.py` | 58 |

Backward compatibility:

- `python trace/sft.py ...` keeps the old entry file and CLI options.
- `detection/scripts/sft.sh` still reaches the same command path.
- Existing imports from `trace.sft` used by self-distillation, GRPO, and
  evaluation scripts are re-exported from the new package.
- Tokenization, prompt reconstruction, trace filtering, parsing, trainer
  configuration, and defaults were kept unchanged.

Verification:

```bash
PYTHONPYCACHEPREFIX=/tmp/rec_pycache python -m py_compile \
  detection/trace/sft.py \
  detection/trace/sft/__init__.py \
  detection/trace/sft/parsing.py \
  detection/trace/sft/targets.py \
  detection/trace/sft/prompts.py \
  detection/trace/sft/selection.py \
  detection/trace/sft/dataset.py \
  detection/trace/sft/training.py \
  detection/trace/sft/cli.py

cd detection
PYTHONPYCACHEPREFIX=/tmp/rec_pycache python trace/sft.py --help

cd detection
PYTHONPYCACHEPREFIX=/tmp/rec_pycache python -c "from trace.sft import QWEN3_THINK_BEGIN, build_prompt_like_collect, build_source_text, load_jsonl, parse_model_json, resolve_prompt_for_row, split_history_turns; print(QWEN3_THINK_BEGIN, callable(build_prompt_like_collect), callable(build_source_text), callable(load_jsonl), callable(parse_model_json), callable(resolve_prompt_for_row), callable(split_history_turns))"

PYTHONPYCACHEPREFIX=/tmp/rec_pycache python -m py_compile $(find detection -name '*.py' -print)
```

## Risks and Guardrails

## Additional Refactor Pass: Personalized Runner

Status: implemented after the secondary CLI splits.

Moved the remaining block-level orchestration out of
`detection/trace/collect_personalized.py` while preserving the historical
entry file and imported function names:

- `detection/trace/personalized_runner.py`: `run_agent_on_sample(...)`,
  per-turn memory update evaluation, turn record formatting, optional metadata
  propagation, and per-session memory update orchestration.
- `detection/trace/personalized_collect.py`: resume ID loading and concurrent
  block collection / JSONL append logic.
- `detection/trace/collect_personalized.py`: compatibility facade for old
  imports plus structured parse wrappers, memory wrappers, turn-eval wrappers,
  argparse, vLLM client setup, sample loading, and CLI dispatch.

Line counts after split:

| file | lines |
|---|---:|
| `detection/trace/collect_personalized.py` | 785 |
| `detection/trace/personalized_runner.py` | 376 |
| `detection/trace/personalized_collect.py` | 95 |

Backward compatibility:

- `python trace/collect_personalized.py ...` keeps the old command path.
- Existing imports of `run_agent_on_sample`, `collect_all`,
  `load_finished_ids`, and `_evaluate_session_per_turn_update` from
  `trace.collect_personalized` still work.
- `trace.collect_personalized.client` and `_is_vllm` remain module-level state
  in the compatibility entry file.
- Output JSONL fields, optional metadata keys, resume semantics, CLI options,
  and memory update modes were kept unchanged.

Verification:

```bash
PYTHONPYCACHEPREFIX=/tmp/rec_pycache python -m py_compile \
  detection/trace/collect_personalized.py \
  detection/trace/personalized_runner.py \
  detection/trace/personalized_collect.py

cd detection
PYTHONPYCACHEPREFIX=/tmp/rec_pycache python trace/collect_personalized.py --help

cd detection
PYTHONPYCACHEPREFIX=/tmp/rec_pycache python -c "from trace.collect_personalized import run_agent_on_sample, collect_all, load_finished_ids, _evaluate_session_per_turn_update; print(callable(run_agent_on_sample), callable(collect_all), callable(load_finished_ids), callable(_evaluate_session_per_turn_update))"

PYTHONPYCACHEPREFIX=/tmp/rec_pycache python -m py_compile $(find detection -name '*.py' -print)
```

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
7. `[eval] refactor: split spur workflow`
8. `[trace] refactor: split sft workflow`
