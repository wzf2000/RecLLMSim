# GPT-5.4 Mini V2 None Limit-10 Results

## Setup

This report compares a small no-memory V2 prompt run with stronger API model
capacity against earlier no-memory baselines.

New run:

- model: `gpt-5.4-mini-2026-03-17`
- turn eval prompt: `v2`
- memory update mode: `none`
- limit users: `10`
- output:
  `detection/outputs/personalized/gpt-5.4-mini-2026-03-17_test_v2_none_limit10.jsonl`
- records: `828` turns from `10` users

For fair comparison, existing full outputs were filtered to the exact same
`sample_id` set:

- `detection/outputs/personalized/gpt54mini_limit10_compare/qwen3_none.jsonl`
- `detection/outputs/personalized/gpt54mini_limit10_compare/gpt4o_mini_none_v2.jsonl`
- `detection/outputs/personalized/gpt54mini_limit10_compare/gpt4o_mini_none_old.jsonl`

Evaluation files:

- `detection/outputs/personalized/gpt54mini_limit10_compare/eval_comparison.json`
- `detection/outputs/personalized/gpt54mini_limit10_compare/eval_with_history_baselines.json`

## Model Comparison

| Method | MAE | RMSE | Pearson | Spearman | QWK |
|---|---:|---:|---:|---:|---:|
| `gpt54mini_v2_none` | `0.6461` | **`0.9267`** | **`0.3667`** | **`0.3410`** | **`0.3340`** |
| `qwen3_none` | `0.6727` | `0.9587` | `0.2710` | `0.2795` | `0.2551` |
| `gpt4o_mini_none_v2` | **`0.6473`** | `0.9479` | `0.2697` | `0.2707` | `0.2574` |
| `gpt4o_mini_none_old` | `0.7488` | `1.0564` | `0.2233` | `0.2149` | `0.2192` |

`gpt-5.4-mini` is clearly better than Qwen3 none on the same subset:

- MAE improves from `0.6727` to `0.6461`.
- Pearson improves from `0.2710` to `0.3667`.
- Spearman improves from `0.2795` to `0.3410`.
- QWK improves from `0.2551` to `0.3340`.

Compared with `gpt4o_mini_none_v2`, exact MAE is almost tied
(`0.6461` vs `0.6473`), but correlation and QWK are much stronger. This suggests
that `gpt-5.4-mini` is not just matching the average error; it gives a more
useful ordinal ranking of satisfaction levels.

## Boundary Metrics

| Method | Acc | F1-DSAT | Kappa | AUC | False SAT | False DSAT |
|---|---:|---:|---:|---:|---:|---:|
| `gpt54mini_v2_none` | `0.7705` | **`0.4172`** | **`0.2763`** | **`0.6886`** | **`0.5342`** | `0.1642` |
| `qwen3_none` | `0.7488` | `0.3376` | `0.1835` | `0.6534` | `0.6370` | `0.1686` |
| `gpt4o_mini_none_v2` | **`0.7754`** | `0.2619` | `0.1333` | `0.6359` | `0.7740` | **`0.1070`** |
| `gpt4o_mini_none_old` | `0.7524` | `0.2756` | `0.1265` | `0.6294` | `0.7329` | `0.1437` |

The most important change is boundary behavior. `gpt-5.4-mini` predicts more
DSAT turns than `gpt4o_mini_none_v2` (`180` vs `106`) and has much lower false
SAT rate (`0.5342` vs `0.7740`). It sacrifices some false DSAT rate, but the
net boundary quality is better: F1-DSAT, kappa, and AUC are all highest among
the compared LLM no-memory runs.

Prediction distributions on the same subset:

| Method | Score 1 | Score 2 | Score 3 | Score 4 | Score 5 |
|---|---:|---:|---:|---:|---:|
| gold | `11` | `41` | `94` | `378` | `304` |
| `gpt54mini_v2_none` | `2` | `29` | `149` | `551` | `97` |
| `qwen3_none` | `0` | `10` | `158` | `502` | `158` |
| `gpt4o_mini_none_v2` | `2` | `7` | `97` | `500` | `222` |

`gpt-5.4-mini` is still conservative about score `5`, but it is less
SAT-biased at the 3/4 boundary than `gpt4o_mini_none_v2`.

## History Baseline Context

On this same subset, simple history baselines are still very strong:

| Method | MAE | Pearson | Spearman | QWK | F1-DSAT |
|---|---:|---:|---:|---:|---:|
| `gpt54mini_v2_none` | `0.6461` | `0.3667` | `0.3410` | `0.3340` | **`0.4172`** |
| `user_history_mean` | **`0.5447`** | **`0.3959`** | **`0.4293`** | **`0.3515`** | `0.2222` |
| `user_history_median` | `0.5314` | `0.3586` | `0.4389` | `0.2719` | `0.0000` |
| `nearest_history_turn` | `0.6787` | `0.3378` | `0.3690` | `0.3376` | `0.3894` |
| `nearest_history_turn_k3` | `0.5918` | `0.3601` | `0.4040` | `0.3482` | `0.3037` |

This means model capacity helps the LLM no-memory judge, especially on DSAT
boundary detection, but it does not solve the full personalized 1-5 prediction
problem by itself. The user's historical score distribution still explains a
large part of the full-score signal.

## Interpretation

The result supports three conclusions:

1. Model capacity matters. With the same V2 no-memory prompt and same sample
   subset, `gpt-5.4-mini` improves over Qwen3 on MAE, ranking metrics, QWK, and
   DSAT boundary metrics.
2. The improvement is not enough to replace personalization. Simple source-task
   user history statistics beat all no-memory LLM judges on MAE and most
   full-score ranking metrics.
3. The most promising use of a stronger judge is boundary semantics, not raw
   user prior modeling. `gpt-5.4-mini` is best among LLM no-memory runs on
   F1-DSAT, kappa, and AUC, suggesting it recognizes dissatisfaction evidence
   better than Qwen3 or gpt-4o-mini.

For next experiments, a stronger model should be tested as a semantic component
combined with explicit user-history calibration, rather than as a standalone
no-memory predictor.
