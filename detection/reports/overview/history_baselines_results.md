# History Baselines Implementation and Results

## Setup

This report records the first implementation of statistical/history baselines
for the RecLLMSim personalized satisfaction setting.

Implemented files:

- `detection/eval/history_baselines.py`
- `detection/scripts/run_history_baselines.sh`

Output directory:

- `detection/outputs/personalized/history_baselines/`

Evaluation file:

- `detection/outputs/personalized/history_baselines/history_baselines_comparison.json`

Data setting:

- split: `test`
- train stats source: train users only, using the same `GroupShuffleSplit`
  seed and ratio as `build_personalized_samples`
- test records: 90 users, 356 blocks, 6474 assistant turns
- per-user history baselines use only source-task `history_sessions` in each
  `(user, target_task)` block

## Implemented Baselines

| Baseline | Description |
|---|---|
| `global_mean` | rounded global mean score from train users |
| `global_majority` | majority score from train users |
| `task_mean` | rounded target-task mean from train users |
| `task_majority` | target-task majority score from train users |
| `user_history_mean` | rounded mean of source-task user history scores |
| `user_history_median` | median of source-task user history scores |
| `user_history_majority` | majority score in source-task user history |
| `user_history_cdf_hash` | empirical source-task user score distribution assigned to target turns by deterministic sample-id hash rank |
| `nearest_history_turn` | score of the top-1 TF-IDF nearest source-task history turn |
| `nearest_history_turn_k3` | rounded mean score of top-3 TF-IDF nearest source-task history turns |

All baselines emit JSONL records compatible with `eval/personalized.py`:

- `sample_id`
- `user`
- `target_task`
- `target_file`
- `turn_idx`
- `gold_score`
- `pred_score`
- `gold_reason`
- `reason_prediction`
- `model`
- `baseline_name`

## Reproduction

Generate all baseline files:

```bash
cd detection
bash scripts/run_history_baselines.sh
```

Evaluate all generated files:

```bash
cd detection
result_files="global_mean=outputs/personalized/history_baselines/global_mean_test.jsonl \
global_majority=outputs/personalized/history_baselines/global_majority_test.jsonl \
task_mean=outputs/personalized/history_baselines/task_mean_test.jsonl \
task_majority=outputs/personalized/history_baselines/task_majority_test.jsonl \
user_mean=outputs/personalized/history_baselines/user_history_mean_test.jsonl \
user_median=outputs/personalized/history_baselines/user_history_median_test.jsonl \
user_majority=outputs/personalized/history_baselines/user_history_majority_test.jsonl \
user_cdf=outputs/personalized/history_baselines/user_history_cdf_hash_test.jsonl \
nearest1=outputs/personalized/history_baselines/nearest_history_turn_test.jsonl \
nearest3=outputs/personalized/history_baselines/nearest_history_turn_k3_test.jsonl" \
output_json=outputs/personalized/history_baselines/history_baselines_comparison.json \
bash scripts/eval_personalized.sh
```

## Main Results

### Full 1-5 Metrics

| Baseline | MAE | RMSE | Pearson | Spearman | QWK |
|---|---:|---:|---:|---:|---:|
| `global_mean` | `0.6985` | `0.9434` | N/A | N/A | `0.0000` |
| `global_majority` | `0.7970` | `1.2182` | N/A | N/A | `0.0000` |
| `task_mean` | `0.6985` | `0.9434` | N/A | N/A | `0.0000` |
| `task_majority` | `0.7519` | `1.1349` | `0.0258` | `0.0348` | `0.0165` |
| `user_history_mean` | `0.5661` | `0.8870` | `0.3560` | `0.3917` | `0.3105` |
| `user_history_median` | **`0.5542`** | `0.9150` | `0.3526` | **`0.3972`** | `0.3075` |
| `user_history_majority` | `0.5820` | `0.9825` | `0.2895` | `0.3573` | `0.2454` |
| `user_history_cdf_hash` | `0.7399` | `1.1476` | `0.2167` | `0.2579` | `0.2167` |
| `nearest_history_turn` | `0.7079` | `1.1064` | `0.2550` | `0.2891` | `0.2543` |
| `nearest_history_turn_k3` | `0.6285` | `0.9798` | `0.3076` | `0.3406` | `0.2974` |

### SAT/DSAT Boundary Metrics

| Baseline | Acc | F1-DSAT | Kappa | AUC | False SAT | False DSAT |
|---|---:|---:|---:|---:|---:|---:|
| `global_mean` | `0.8290` | `0.0000` | `0.0000` | `0.5000` | `1.0000` | `0.0000` |
| `global_majority` | `0.8290` | `0.0000` | `0.0000` | `0.5000` | `1.0000` | `0.0000` |
| `task_mean` | `0.8290` | `0.0000` | `0.0000` | `0.5000` | `1.0000` | `0.0000` |
| `task_majority` | `0.8290` | `0.0000` | `0.0000` | `0.5135` | `1.0000` | `0.0000` |
| `user_history_mean` | `0.8217` | `0.2255` | `0.1509` | `0.6583` | `0.8482` | `0.0401` |
| `user_history_median` | `0.8256` | `0.1643` | `0.1093` | `0.6623` | `0.8997` | `0.0248` |
| `user_history_majority` | `0.8157` | `0.1286` | `0.0675` | `0.6306` | `0.9205` | `0.0324` |
| `user_history_cdf_hash` | `0.7535` | `0.2752` | `0.1267` | `0.6040` | `0.7263` | `0.1476` |
| `nearest_history_turn` | `0.7751` | **`0.3106`** | `0.1766` | `0.6406` | `0.7037` | `0.1261` |
| `nearest_history_turn_k3` | `0.7916` | `0.2963` | **`0.1774`** | `0.6610` | `0.7435` | `0.0980` |

## Interpretation

The strongest signal is that user history distribution alone is a very strong
full-score baseline:

- `user_history_median` reaches `MAE=0.5542`, `Spearman=0.3972`, `QWK=0.3075`.
- `user_history_mean` reaches `MAE=0.5661`, `Pearson=0.3560`, `QWK=0.3105`.

This means many global 1-5 improvements can be explained by cross-task user
strictness / leniency transfer, not necessarily by turn-level semantic judging.

The nearest-history baselines behave differently:

- `nearest_history_turn` is weaker on MAE but strongest among these baselines on
  `F1-DSAT=0.3106`.
- `nearest_history_turn_k3` improves MAE to `0.6285` and keeps boundary kappa
  near the top.

This suggests that simple lexical similarity to labeled source-task turns
already captures some DSAT boundary signal.

## Practical Use

For future Qwen V2 comparisons:

- Use `user_history_mean` / `user_history_median` as the required full 1-5
  history-only baselines.
- Use `nearest_history_turn` / `nearest_history_turn_k3` as non-LLM semantic-ish
  history retrieval baselines.
- Use `global_mean` as a sanity lower bound, but do not treat its high accuracy
  as meaningful for boundary because it predicts all samples as SAT.

The main open question is now stricter:

> Does a Qwen memory-agent improve user-aware / within-user ranking and DSAT
> detection beyond what can be obtained from user history distribution and
> lexical nearest-neighbor transfer alone?
