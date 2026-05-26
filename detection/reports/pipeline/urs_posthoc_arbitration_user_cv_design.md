# URS Post-Hoc Arbitration User-CV Design

## Motivation

The single dev split for URS has only 74 target sessions.
The previous grid search selected a rule that looked good on dev but became SAT-biased on test.
To reduce this small-dev overfitting, post-hoc arbitration now supports user-level cross-validation.

## Implementation

Updated files:

- `eval/urs_posthoc_arbitration.py`
- `scripts/posthoc_urs_arbitration.sh`

New mode:

- `cv_search=1`

The CV search workflow:

1. Load base and auxiliary prediction files on the same sample set.
2. Split users into `n_folds` folds.
3. For each fold, search the best arbitration rule on the training users.
4. Apply the selected rule to held-out users.
5. Report averaged held-out metrics for selected arbitration, raw base, and auxiliary predictions.

Search uses a lightweight evaluator with global and boundary metrics only.
The final fold evaluation still computes the full evaluator metrics.

## Recommended Clean Protocol

Run base and auxiliary predictors on the original URS train user pool:

```bash
model=Qwen/Qwen3-8B \
vllm_base_url=http://localhost:8001/v1 \
vllm_api_key=EMPTY \
split=train \
memory_update_mode=none \
urs_prompt_version=urs_v2_calibrated_task_guarded \
output_jsonl=outputs/urs/train/Qwen_Qwen3-8B_train_none_urs_v2_calibrated_task_guarded.jsonl \
max_workers=4 \
bash scripts/collect_urs.sh
```

```bash
model=Qwen/Qwen3-8B \
vllm_base_url=http://localhost:8001/v1 \
vllm_api_key=EMPTY \
split=train \
memory_update_mode=none \
urs_prompt_version=urs_v2_calibrated_task_guarded_evidence_first \
output_jsonl=outputs/urs/train/Qwen_Qwen3-8B_train_none_urs_v2_calibrated_task_guarded_evidence_first.jsonl \
max_workers=4 \
bash scripts/collect_urs.sh
```

Run user-level CV rule selection:

```bash
base_jsonl=outputs/urs/train/Qwen_Qwen3-8B_train_none_urs_v2_calibrated_task_guarded.jsonl \
aux_jsonl=outputs/urs/train/Qwen_Qwen3-8B_train_none_urs_v2_calibrated_task_guarded_evidence_first.jsonl \
cv_search=1 \
strategies="selected_downgrade selected_upgrade selected_boundary_aux" \
max_tasks=4 \
search_metric=balanced \
n_folds=5 \
cv_seed=42 \
output_json=outputs/urs/train/qwen3_8b_urs_posthoc_arbitration_user_cv.json \
bash scripts/posthoc_urs_arbitration.sh
```

Then inspect `summary.selected` versus `summary.base`.
If selected arbitration is not consistently better than base in CV, do not use post-hoc arbitration as a clean main result.

## Smoke Test

A small smoke test on existing test predictions with `n_folds=2` and `max_tasks=1` ran successfully:

| version | MAE | Spearman | QWK | Boundary Acc | F1-DSAT | False-SAT |
|---|---:|---:|---:|---:|---:|---:|
| selected CV | 0.6967 | 0.2797 | 0.2321 | 0.6611 | 0.5088 | 0.4776 |
| base CV | 0.6949 | 0.2856 | 0.2372 | 0.6628 | 0.5070 | 0.4868 |

This smoke test is only a code validation run because it uses test predictions.
The clean experiment should use `split=train` predictions.

## User-CV Results

The clean user-level CV experiment was run on `split=train` predictions.

Output:

- `outputs/urs/train/qwen3_8b_urs_posthoc_arbitration_user_cv.json`

Dataset:

- Records: 147
- Users: 29
- Folds: 5

Average held-out metrics:

| version | MAE | RMSE | Pearson | Spearman | QWK | Boundary Acc | F1-SAT | F1-DSAT | False-SAT | False-DSAT |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| selected arbitration | 0.7236 | 1.0119 | 0.0743 | 0.0674 | 0.0599 | 0.5864 | 0.6530 | 0.4421 | 0.4347 | 0.3983 |
| base task_guarded_v1 | 0.7053 | 0.9998 | 0.1113 | 0.1119 | 0.0878 | 0.6048 | 0.6548 | 0.4888 | 0.3611 | 0.4094 |
| auxiliary evidence_first | 0.7099 | 1.0193 | 0.1179 | 0.0806 | 0.0914 | 0.5809 | 0.6537 | 0.4382 | 0.3896 | 0.4116 |

Fold-level selected rules:

| fold | valid records | selected strategy | selected tasks | changed records | valid Acc | valid F1-DSAT | valid MAE |
|---:|---:|---|---|---:|---:|---:|---:|
| 0 | 24 | selected_upgrade | leisure, retrieval, text | 0 | 0.6667 | 0.5556 | 0.6667 |
| 1 | 32 | selected_upgrade | advice, retrieval, text | 1 | 0.5312 | 0.3478 | 0.8438 |
| 2 | 33 | selected_boundary_aux | leisure, professional | 2 | 0.3939 | 0.1667 | 1.0606 |
| 3 | 35 | selected_upgrade | leisure, retrieval, text | 2 | 0.5143 | 0.5405 | 0.7429 |
| 4 | 23 | selected_upgrade | leisure, retrieval, text | 0 | 0.8261 | 0.6000 | 0.3043 |

The selected rules are unstable across folds and often change very few held-out records.
More importantly, the selected arbitration underperforms the raw base predictor on MAE, RMSE, Spearman, QWK, boundary accuracy, F1-SAT, and F1-DSAT.
It only reduces False-DSAT slightly, while increasing False-SAT substantially.

## Conclusion

User-level CV does not support post-hoc arbitration as a clean default URS evaluator.

The earlier `selected_boundary_aux` result remains useful as a test-set diagnostic showing that auxiliary boundary disagreements contain signal.
However, with the current URS train split size, rule selection does not generalize reliably enough to be reported as the main method.

For clean reporting, use `urs_v2_calibrated_task_guarded` as the main URS evaluator.
Post-hoc arbitration should be described as exploratory unless a larger development set or a more stable cross-validation protocol shows consistent held-out gains.
