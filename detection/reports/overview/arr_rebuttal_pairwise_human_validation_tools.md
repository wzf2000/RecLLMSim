# Pairwise Human Validation Tools

## Purpose

This note records the implementation added for replay pairwise human validation.
The goal is to support a small independent human validation study for PersTurnBench replay results.
Even if the ARR rebuttal timeline is too tight, the same tools can be reused for a later revision or resubmission.

## Added Files

- `detection/tools/build_replay_pairwise_items.py`
  - Builds blinded pairwise annotation items from scored static-replay JSONL files.
  - Supports candidate-vs-candidate comparisons by default.
  - Supports candidate-vs-original-assistant comparisons with `--include_source_assistant`.
  - Reconstructs user profiles through `lib.personalized_data.build_personalized_samples()`.
  - Stratifies selected items by evaluator score gap: `large_gap`, `small_gap`, and `tie`.
  - Balances repeated exposure with `--max_per_sample`, `--max_per_user`, and `--max_per_model_pair`.

- `detection/tools/replay_pairwise_validation_app.py`
  - Streamlit annotation page for pairwise validation.
  - Shows task context, optional user profile, conversation prefix, current user request, and blinded Response A/B.
  - Hides model names, evaluator scores, gold labels, and score deltas by default.
  - Saves one append-only JSONL file per annotator under `outputs/human_validation/annotations/`.
  - Supports resume by loading the latest record per `item_id`.

## Item Schema

Each generated item contains:

- identifiers: `item_id`, `sample_id`, `user`, `target_task`, `target_file`, `turn_idx`;
- context: `task_context`, `profile`, `dialogue_prefix`, `current_user_request`;
- selection metadata: `selection_mode`, `selection_score`, `selection_reasons`;
- hidden evaluation metadata: `pair_kind`, `pair_bucket`, `evaluator_preference`, `evaluator_score_delta`, `model_pair`;
- blinded sides: `side_a` and `side_b`, each containing a response and hidden source metadata.

The annotation UI only exposes the context and response text during normal use.
The hidden metadata can be shown with the sidebar debug checkbox for internal inspection.

## Recommended Commands

Candidate-vs-candidate validation over the main replay models:

```bash
cd /data/wangzhefan/RecLLMSim
conda activate chat
python detection/tools/build_replay_pairwise_items.py \
  --candidate_files \
  kimi=detection/outputs/static_replay/kimi-k2.6_test_hard_scored_by_qwen3_8b_v2_refCDF.jsonl \
  glm=detection/outputs/static_replay/glm-5.1_test_hard_scored_by_qwen3_8b_v2_refCDF.jsonl \
  deepseek=detection/outputs/static_replay/deepseek-v4-pro_test_hard_scored_by_qwen3_8b_v2_refCDF.jsonl \
  gpt=detection/outputs/static_replay/gpt-5.5_test_hard_scored_by_qwen3_8b_v2_refCDF.jsonl \
  claude=detection/outputs/static_replay/claude-opus-4-7_test_hard_scored_by_qwen3_8b_v2_refCDF.jsonl \
  gemini=detection/outputs/static_replay/gemini-3.1-pro-preview_test_hard_scored_by_qwen3_8b_v2_refCDF.jsonl \
  minimax=detection/outputs/static_replay/minimax-m2.7_test_hard_scored_by_qwen3_8b_v2_refCDF.jsonl \
  --output_jsonl detection/outputs/human_validation/replay_pairwise_items.jsonl \
  --summary_json detection/outputs/human_validation/replay_pairwise_items_summary.json \
  --sample_size 120 \
  --max_per_sample 1 \
  --max_per_user 4 \
  --max_per_model_pair 20
```

Optional candidate-vs-original-assistant validation:

```bash
python detection/tools/build_replay_pairwise_items.py \
  --candidate_files \
  kimi=detection/outputs/static_replay/kimi-k2.6_test_hard_scored_by_qwen3_8b_v2_refCDF.jsonl \
  gpt=detection/outputs/static_replay/gpt-5.5_test_hard_scored_by_qwen3_8b_v2_refCDF.jsonl \
  minimax=detection/outputs/static_replay/minimax-m2.7_test_hard_scored_by_qwen3_8b_v2_refCDF.jsonl \
  --include_source_assistant \
  --output_jsonl detection/outputs/human_validation/replay_pairwise_items_with_source.jsonl \
  --summary_json detection/outputs/human_validation/replay_pairwise_items_with_source_summary.json \
  --sample_size 120
```

Start the annotation UI:

```bash
cd /data/wangzhefan/RecLLMSim/detection
conda activate chat
streamlit run tools/replay_pairwise_validation_app.py -- \
  --items_jsonl outputs/human_validation/replay_pairwise_items.jsonl \
  --output_dir outputs/human_validation/annotations
```

## Dry-Run Verification

The following command was run in the `chat` environment:

```bash
conda run -n chat python detection/tools/build_replay_pairwise_items.py \
  --candidate_files \
  kimi=detection/outputs/static_replay/kimi-k2.6_test_hard_scored_by_qwen3_8b_v2_refCDF.jsonl \
  gpt=detection/outputs/static_replay/gpt-5.5_test_hard_scored_by_qwen3_8b_v2_refCDF.jsonl \
  mini=detection/outputs/static_replay/minimax-m2.7_test_hard_scored_by_qwen3_8b_v2_refCDF.jsonl \
  --output_jsonl detection/outputs/human_validation/replay_pairwise_items_dryrun.jsonl \
  --summary_json detection/outputs/human_validation/replay_pairwise_items_dryrun_summary.json \
  --sample_size 12 \
  --max_per_user 2 \
  --max_per_model_pair 6 \
  --include_source_assistant
```

Dry-run summary:

```json
{
  "n_candidates": 1800,
  "n_selected": 12,
  "selected_pair_buckets": {
    "small_gap": 5,
    "tie": 2,
    "large_gap": 5
  },
  "selected_pair_kinds": {
    "candidate_source": 10,
    "candidate_candidate": 2
  },
  "selected_users": 10,
  "selected_samples": 12
}
```

Additional checks:

- `PYTHONPYCACHEPREFIX=/tmp/rec_pycache conda run -n chat python -m py_compile detection/tools/build_replay_pairwise_items.py detection/tools/replay_pairwise_validation_app.py`
- `conda run -n chat python -c "import streamlit; print(streamlit.__version__)"`

The installed Streamlit version in the `chat` environment is `1.55.0`.

## Suggested Rebuttal Use

For rebuttal, the lowest-cost version is to annotate around 60 candidate-vs-candidate pairs with two independent annotators.
A stronger version is 100--120 pairs with two or three annotators.
The main reported statistic should be agreement between majority human preference and the frozen evaluator preference, with a separate high-margin subset.
Close pairs and ties should be reported separately rather than used as a hard failure case.
