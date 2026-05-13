# URS V2 Calibrated Prompt Design

## Motivation

The latest URS `Qwen/Qwen3-8B` predictor run shows a strong score-3 collapse:

- Gold distribution: `1:19, 2:47, 3:139, 4:250, 5:129`
- Predicted distribution: `1:3, 2:9, 3:366, 4:185, 5:21`

This reduces absolute error in some cases but weakens rank/ordinal quality:

- Pearson: `0.2312`
- Spearman: `0.2354`
- QWK: `0.1868`
- F1-DSAT: `0.5386`

The model appears too conservative on session-level URS scoring. Many sessions
that are globally satisfactory are pushed down to `3`.

## Change

Added a new URS scoring prompt version:

- `urs_prompt_version=urs_v2_calibrated`

The default remains:

- `urs_prompt_version=v2`

The calibrated version adds an explicit URS session-level score calibration
block to both memory and no-memory scoring prompts.

## Calibration Rules

The new block clarifies:

- `4` means the session is overall satisfactory; minor omissions or wording
  issues should not automatically lower it to `3`.
- `3` means neutral/general; the session only partially satisfies the request or
  leaves a clear gap.
- `5` does not require perfection; complete, helpful, clearly tailored sessions
  can receive `5`.
- For the `3/4` boundary, if the main task is solved and there is no serious
  error, prefer `4`.
- For the `4/5` boundary, avoid being overly conservative.

## Files

- `detection/lib/urs_memory.py`
- `detection/trace/urs/session_eval.py`
- `detection/trace/urs/runner.py`
- `detection/trace/urs/collect.py`
- `detection/trace/urs/cli.py`
- `detection/scripts/collect_urs.sh`
- `detection/trace/score_urs_static_replay.py`
- `detection/scripts/score_urs_static_replay.sh`

## Recommended Comparison

Run four URS predictor settings:

1. `v2 + memory_update_mode=none`
2. `urs_v2_calibrated + memory_update_mode=none`
3. `v2 + no_memory`
4. `urs_v2_calibrated + no_memory`

Primary metrics:

- Pearson
- Spearman
- QWK
- F1-DSAT
- SAT/DSAT distribution

The main expected improvement is reduced score-3 collapse and better SAT/DSAT
balance. If QWK or F1-DSAT drops sharply, the calibration is too permissive.
