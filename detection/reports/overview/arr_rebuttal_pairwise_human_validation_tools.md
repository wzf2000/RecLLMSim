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
  - Adds source-history preference evidence for annotators: score distribution, template summary, low-side reason counts, and high/low historical anchor examples.
  - The preference summary is generated from raw source-history labels and examples, not copied from the evaluator memory.
  - Stratifies selected items by evaluator score gap: `large_gap`, `small_gap`, and `tie`.
  - Balances repeated exposure with `--max_per_sample`, `--max_per_user`, and `--max_per_model_pair`.

- `detection/tools/replay_pairwise_validation_app.py`
  - Streamlit annotation page for pairwise validation.
  - Shows task context, optional user profile, conversation prefix, current user request, and blinded Response A/B.
  - Shows an optional `User preference evidence` panel to help annotators make personalized pairwise judgments.
  - Hides model names, evaluator scores, gold labels, and score deltas by default.
  - Supports subset assignment in the sidebar: all items, the first half, or the second half.
  - For a 120-item file, this allows annotators to label all 120 items, items 1--60, or items 61--120.
  - Saves one append-only JSONL file per annotator under `outputs/human_validation/annotations/`.
  - Supports resume by loading the latest record per `item_id`.

## Item Schema

Each generated item contains:

- identifiers: `item_id`, `sample_id`, `user`, `target_task`, `target_file`, `turn_idx`;
- context: `task_context`, `profile`, `user_preference_evidence`, `dialogue_prefix`, `current_user_request`;
- selection metadata: `selection_mode`, `selection_score`, `selection_reasons`;
- hidden evaluation metadata: `pair_kind`, `pair_bucket`, `evaluator_preference`, `evaluator_score_delta`, `model_pair`;
- blinded sides: `side_a` and `side_b`, each containing a response and hidden source metadata.

The annotation UI only exposes the context, source-history preference evidence, and response text during normal use.
The hidden metadata can be shown with the sidebar debug checkbox for internal inspection.
When a subset is selected, progress and navigation are computed within that subset.
Saved records include `annotation_subset`, `subset_item_index`, `source_item_index`, and `source_item_total`, so later analysis can recover both the annotator assignment and the original item-file position.

`user_preference_evidence` contains:

- `score_distribution`: source-history score counts, mean score, SAT rate, neutral rate, and score-1/2 rate;
- `summary`: a short template-generated summary of user strictness and common low-side reasons;
- `low_side_reasons`: top low-side / neutral reason counts from history;
- `anchor_examples`: up to two high-score and two low/neutral historical examples from other scenarios.

The evidence is meant to support independent human labeling of which response better fits the original user's preferences.
It does not include current-turn gold labels, evaluator scores, model identities, or evaluator-generated memory text.

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
  --max_per_model_pair 20 \
  --max_anchor_examples_per_side 2 \
  --anchor_excerpt_chars 420
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

After adding preference evidence, a second dry-run generated `detection/outputs/human_validation/replay_pairwise_items_evidence_dryrun.jsonl` with the same 12-item selection setting.
The first inspected item contained:

- five-bucket historical score distribution over 67 source-history turns;
- a template summary with historical mean, SAT/Neutral/score-1/2 rates, and common low-side reasons;
- two high-score anchors and two low/neutral anchors.

## Suggested Rebuttal Use

For rebuttal, the lowest-cost version is to annotate around 60 candidate-vs-candidate pairs with two independent annotators.
A stronger version is 100--120 pairs with two or three annotators.
The main reported statistic should be agreement between majority human preference and the frozen evaluator preference, with a separate high-margin subset.
Close pairs and ties should be reported separately rather than used as a hard failure case.
