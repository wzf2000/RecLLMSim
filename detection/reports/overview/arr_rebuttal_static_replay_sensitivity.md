# ARR Rebuttal Static Replay Sensitivity

Date: 2026-07-09

This report records the static replay scoring sensitivity analysis prepared for the ARR May 2026 rebuttal.
It addresses reviewer concerns that PersTurnBench only reported reference-CDF calibrated scores and that calibration may compress or reorder model differences.

## Source Files

The analysis uses existing static replay evaluation files under `detection/outputs/static_replay/`.
No new candidate generation or judge scoring was run for this report.

For each of the seven main candidate models, we read:

- raw Qwen3-8B evaluator score summaries: `{model}_test_hard_scored_by_qwen3_8b_v2_eval.json`;
- reference mean-shift summaries: `{model}_test_hard_scored_by_qwen3_8b_v2_refMS_eval.json`;
- reference-CDF summaries: `{model}_test_hard_scored_by_qwen3_8b_v2_refCDF_eval.json`;
- reference-transfer-CDF summaries for rank correlation checks: `{model}_test_hard_scored_by_qwen3_8b_v2_refTransferCDF_eval.json`.

## Main Sensitivity Table

Rows are sorted by reference-CDF user-macro mean.
The low-side rate is the fraction of replay turns scored at most 3 under the reference-CDF view.

| Model | Raw user macro | RefMS user macro | RefCDF user macro | RefCDF low-side rate |
|---|---:|---:|---:|---:|
| kimi-k2.6 | 4.8400 | 4.8535 | 4.8121 | 0.0133 |
| glm-5.1 | 4.8415 | 4.8273 | 4.8003 | 0.0267 |
| deepseek-v4-pro | 4.8024 | 4.8153 | 4.7911 | 0.0267 |
| gpt-5.5 | 4.7213 | 4.7578 | 4.6893 | 0.0567 |
| claude-opus-4-7 | 4.5553 | 4.6338 | 4.6361 | 0.0467 |
| gemini-3.1-pro-preview | 4.6310 | 4.6516 | 4.6283 | 0.0600 |
| minimax-m2.7 | 4.2057 | 4.3621 | 4.4256 | 0.1367 |

## Rank Correlation

Kendall rank correlations are computed over the seven candidate models using user-macro mean as the ranking score.

| Compared leaderboard | Kendall's tau vs RefCDF | Concordant pairs | Discordant pairs |
|---|---:|---:|---:|
| Raw | 0.8095 | 19 | 2 |
| RefMS | 0.9048 | 20 | 1 |
| RefTransferCDF | 0.9048 | 20 | 1 |

## Interpretation for Rebuttal

Calibration affects the exact ordering of close models, especially inside the top group and the middle group.
However, the broad model tiers are stable across scoring views:

- `kimi-k2.6`, `glm-5.1`, and `deepseek-v4-pro` remain the strongest group.
- `gpt-5.5`, `claude-opus-4-7`, and `gemini-3.1-pro-preview` remain a close middle group.
- `minimax-m2.7` remains clearly separated as the weakest model.

The rebuttal should therefore avoid claiming a strict human-preference leaderboard.
The stronger and safer claim is that PersTurnBench provides a tiered automatic screening signal, and that the broad tiers are stable across raw and calibrated scoring views.

Suggested rebuttal wording:

> We agree that reporting only calibrated scores can obscure score-scale effects.
> In the revision, we will add raw and reference mean-shift benchmark results together with the reference-CDF view and report Kendall rank correlations across these leaderboards.
> The exact order of close models changes slightly, but the broad model tiers remain stable.

