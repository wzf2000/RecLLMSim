# URS Self-Consistency Limit-120 Results

## Setup

This report evaluates a lightweight self-consistency experiment on URS.

Scope:

- Prompt: `urs_v2_calibrated_task_guarded`
- Model: local `Qwen/Qwen3-8B`
- Memory update: `none`
- Target intents: `leisure`, `professional`, `text`, `other`
- Limit: 120 blocks, resulting in 218 session-level records
- Runs: 3 independent collections

Outputs:

- `outputs/urs/self_consistency/qwen3_task_guarded_sc1_limit120.jsonl`
- `outputs/urs/self_consistency/qwen3_task_guarded_sc2_limit120.jsonl`
- `outputs/urs/self_consistency/qwen3_task_guarded_sc3_limit120.jsonl`
- `outputs/urs/self_consistency/qwen3_task_guarded_sc_limit120_eval.json`
- `outputs/urs/self_consistency/qwen3_task_guarded_sc_limit120_ensemble.jsonl`
- `outputs/urs/self_consistency/qwen3_task_guarded_sc_limit120_single_vs_ensemble.json`

The ensemble file contains all requested ensemble strategies.
The meaningful per-strategy metrics are from `qwen3_task_guarded_sc_limit120_eval.json`.

## Single Runs vs Majority

| version | n | MAE | RMSE | Pearson | Spearman | QWK | Boundary Acc | F1-SAT | F1-DSAT | False-SAT | False-DSAT |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| sc1 | 218 | 0.7477 | 1.0730 | 0.2034 | 0.2106 | 0.1768 | 0.6147 | 0.6957 | 0.4750 | 0.5250 | 0.3043 |
| sc2 | 218 | 0.7569 | 1.0601 | 0.2546 | 0.2403 | 0.2298 | 0.6101 | 0.6863 | 0.4848 | 0.5000 | 0.3261 |
| sc3 | 218 | 0.7523 | 1.0752 | 0.2193 | 0.2264 | 0.1954 | 0.6147 | 0.6912 | 0.4878 | 0.5000 | 0.3188 |
| majority_sat | 218 | 0.7431 | 1.0623 | 0.2372 | 0.2388 | 0.2106 | 0.6239 | 0.6963 | 0.5060 | 0.4750 | 0.3188 |

The majority ensemble improves F1-DSAT over all three single runs and lowers False-SAT compared with each single run.
It also slightly improves MAE over the best single run.
However, the improvements are modest.

## Ensemble Strategies

`majority_sat`, `median`, and `mean` produce nearly identical predictions in this run.

| strategy | n | MAE | Pearson | Spearman | QWK | Boundary Acc | F1-DSAT | False-SAT | confidence |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| majority_sat | 218 | 0.7431 | 0.2372 | 0.2388 | 0.2106 | 0.6239 | 0.5060 | 0.4750 | high=159, medium=59 |
| median | 218 | 0.7431 | 0.2343 | 0.2403 | 0.2052 | 0.6239 | 0.5060 | 0.4750 | high=159, medium=59 |
| mean | 218 | 0.7431 | 0.2343 | 0.2403 | 0.2052 | 0.6239 | 0.5060 | 0.4750 | high=159, medium=59 |

No low-confidence cases were produced by the current three-run setup.
This indicates that repeated sampling mostly changes scores within a narrow range and does not create a strong uncertainty signal.

## Interpretation

Self-consistency is mildly useful on this subset, but it is not as promising as boundary arbitration.

The majority ensemble improves the subset F1-DSAT from the best single-run value of 0.4878 to 0.5060.
This shows that repeated prediction can reduce some boundary noise.
But the absolute metrics are still not strong, and the gains are smaller than the earlier `selected_boundary_aux` post-hoc result on the full URS test set.

The current evidence suggests:

- Self-consistency is not worth running as a full default evaluator because it triples inference cost for modest gains.
- It may still be useful as a targeted diagnostic on boundary-disagreement samples.
- The better practical direction remains post-hoc boundary arbitration using `task_guarded_v1` and `evidence_first`.

## Recommendation

Do not prioritize full self-consistency.

If self-consistency is revisited, run it only on uncertain samples:

- samples where `task_guarded_v1` and `evidence_first` cross the 3/4 boundary,
- samples where the base prediction is 3 or 4,
- and selected tasks where arbitration already showed benefit.

