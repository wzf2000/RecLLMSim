# PersTurnBench Software Release Packaging

## Artifact

- Release directory: `detection/assets/releases/PersTurnBench/`
- Compressed archive: `detection/assets/releases/PersTurnBench.tar.gz`
- SHA256: `90cb7d9088772a1ff0a45d8fb2a7fd5d3975bc065b9132c58491afb6013359a1`
- Archive size: 144 KB

## Scope

The software release is a curated subset of the repository code for paper-facing experiments.
It excludes data, checkpoints, intermediate outputs, paper drafts, URS-specific pipelines, training code, and exploratory analysis reports.

Included code areas:

- `detection/lib/`: shared data loading, memory construction, prompt builders, calibration/statistics helpers, and metrics.
- `detection/trace/`: personalized evaluator inference, static replay collection, and replay scoring.
- `detection/eval/`: personalized evaluator metrics, baseline evaluation, static replay metrics, and reference-CDF calibration.
- `detection/scripts/`: shell entrypoints for the main evaluator, baselines, and PersTurnBench replay evaluation.
- `detection/assets/`: plotting scripts for PersTurnBench figures.

## Paper-Facing Configuration

The primary evaluator configuration documented in `README.md` is:

- model: `Qwen/Qwen3-8B` through an OpenAI-compatible vLLM endpoint;
- memory version: `v2`;
- memory update mode: `none`;
- turn-evaluation prompt version: `v2`;
- split: `test`.

The release keeps some shared helper functions that support older or exploratory modes when they are required by imports, but the documented commands use the no-update memory setting.

## Data and Configuration

The package does not include data.
Users should extract `persturnbench_data_release_20260526.tar.gz` into the release root and move its `data/` directory next to `detection/`.
The dataset release is also linked from the software README: <https://zenodo.org/records/20391777>.

The package does not include real API credentials.
It provides `api_config.example.json`.
The release-local `detection/lib/llm.py` was adjusted so missing `api_config.json` does not prevent imports or local vLLM usage.
The default OpenAI client disables environment proxy inheritance to avoid import-time failures in chat environments with SOCKS proxy variables but no SOCKS transport package.

## Checks

- Verified that the release tree contains no `data`, `outputs`, `ckpts`, `paper-draft`, or `reports` directories.
- Verified that release scripts do not hard-code local Python interpreter paths.
- Installed `requirements.txt` into `/tmp/persturn_deps` in the chat environment and ran release-local checks with `PYTHONPATH=/tmp/persturn_deps:<release>/detection`.
- Ran `python -m py_compile` over the released Python files.
- Verified `trace/collect_personalized.py --help`, `trace/collect_static_replay.py --help`, and `trace/score_static_replay.py --help`.
- Verified the released data loader on the sanitized data: 356 test user-scenario blocks, 1470 target sessions, and 6474 target assistant turns.
- Verified `scripts/evaluate_main_evaluator.sh` with a small smoke JSONL file.
- Full LLM generation/scoring was not executed in the chat environment because it requires an available vLLM or OpenAI-compatible endpoint.

## Validation Fixes

The chat-environment validation exposed several release-only portability issues, which were fixed before rebuilding the archive:

- `detection/lib/utils.py` now treats `torch` as optional for seed setting, so the paper-facing evaluator scripts do not require PyTorch for non-training utilities.
- `detection/lib/personalized_data.py` now accepts sanitized release files without the removed `questionnaire` field.
- `detection/lib/llm.py` constructs the default OpenAI client with `trust_env=False` to avoid proxy-related import-time failures.
- Shell entrypoints preserve external `PYTHONPATH` instead of replacing it, so users can run against dependencies installed in custom locations.
