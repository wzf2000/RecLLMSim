# URS Post-Hoc Arbitration Dev Split Design

## Motivation

The current best URS post-hoc rule, `selected_boundary_aux`, was selected after inspecting test-set results.
For a cleaner experimental protocol, arbitration rules should be selected on a dev split and then fixed before test evaluation.

## Data Split

`build_urs_personalized_samples` now supports:

- `split=train`
- `split=dev`
- `split=test`
- `split=all`

The original URS split uses user-level `GroupShuffleSplit` with `train_ratio=0.2`.
The new `dev` split is carved from the original train user pool using another user-level split controlled by `dev_ratio`.

With default settings:

- `train_ratio=0.2`
- `dev_ratio=0.5`
- `split_seed=42`

Observed sizes:

| split | users | blocks | target sessions |
|---|---:|---:|---:|
| train | 29 | 87 | 147 |
| dev | 15 | 44 | 74 |
| test | 116 | 339 | 584 |

The dev set is small, so it should be used for coarse arbitration-rule selection rather than fine-grained tuning.

## Arbitration Grid Search

`eval/urs_posthoc_arbitration.py` now supports `--grid_search`.

It enumerates:

- arbitration strategies,
- task subsets,
- and a chosen selection metric.

The default balanced metric is:

```text
F1-DSAT + boundary accuracy + Spearman + QWK - 0.25 * MAE
```

This favors a rule that improves dissatisfaction detection without destroying ordinal/ranking quality.

## Recommended Protocol

Step 1: run base and auxiliary predictors on dev.

```bash
model=Qwen/Qwen3-8B \
vllm_base_url=http://localhost:8001/v1 \
vllm_api_key=EMPTY \
split=dev \
memory_update_mode=none \
urs_prompt_version=urs_v2_calibrated_task_guarded \
output_jsonl=outputs/urs/dev/Qwen_Qwen3-8B_dev_none_urs_v2_calibrated_task_guarded.jsonl \
max_workers=4 \
bash scripts/collect_urs.sh
```

```bash
model=Qwen/Qwen3-8B \
vllm_base_url=http://localhost:8001/v1 \
vllm_api_key=EMPTY \
split=dev \
memory_update_mode=none \
urs_prompt_version=urs_v2_calibrated_task_guarded_evidence_first \
output_jsonl=outputs/urs/dev/Qwen_Qwen3-8B_dev_none_urs_v2_calibrated_task_guarded_evidence_first.jsonl \
max_workers=4 \
bash scripts/collect_urs.sh
```

Step 2: select arbitration rule on dev.

```bash
base_jsonl=outputs/urs/dev/Qwen_Qwen3-8B_dev_none_urs_v2_calibrated_task_guarded.jsonl \
aux_jsonl=outputs/urs/dev/Qwen_Qwen3-8B_dev_none_urs_v2_calibrated_task_guarded_evidence_first.jsonl \
grid_search=1 \
strategies="selected_downgrade selected_upgrade selected_boundary_aux" \
max_tasks=4 \
search_metric=balanced \
output_json=outputs/urs/dev/qwen3_8b_urs_posthoc_arbitration_grid_dev.json \
bash scripts/posthoc_urs_arbitration.sh
```

Step 3: apply the selected rule to test.

Use the strategy and selected task list from the `best` field of:

- `outputs/urs/dev/qwen3_8b_urs_posthoc_arbitration_grid_dev.json`

Example if dev selects `selected_boundary_aux` with `leisure professional text other`:

```bash
base_jsonl=outputs/urs/Qwen_Qwen3-8B_test_none_urs_v2_calibrated_task_guarded.jsonl \
aux_jsonl=outputs/urs/Qwen_Qwen3-8B_test_none_urs_v2_calibrated_task_guarded_evidence_first.jsonl \
strategies="selected_boundary_aux" \
selected_tasks="leisure professional text other" \
output_dir=outputs/urs/posthoc_dev_selected \
output_json=outputs/urs/qwen3_8b_urs_posthoc_dev_selected_test.json \
bash scripts/posthoc_urs_arbitration.sh
```

Step 4: evaluate the fixed test output.

```bash
result_files="task_v1=outputs/urs/Qwen_Qwen3-8B_test_none_urs_v2_calibrated_task_guarded.jsonl posthoc_dev=outputs/urs/posthoc_dev_selected/urs_posthoc_selected_boundary_aux.jsonl" \
output_json=outputs/urs/qwen3_8b_urs_posthoc_dev_selected_test_compare.json \
bash scripts/eval_urs_predictor.sh
```

## Notes

The existing test-selected `selected_boundary_aux` result is still useful as an upper-bound diagnostic.
The dev-selected version should be used for cleaner reporting if it performs similarly on test.

