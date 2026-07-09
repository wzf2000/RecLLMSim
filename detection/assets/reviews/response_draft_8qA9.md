# Response Draft for Reviewer 8qA9

本文档是给 Reviewer 8qA9 的 rebuttal 草稿。
英文段落可直接进入 rebuttal，中文 TODO 表示还需要补实验或最终确认。

## Reviewer Position

8qA9 整体偏正面，认可问题设定、comparative memory、meta-evaluation 结果和 benchmark 价值。
主要担忧集中在 PersTurnBench 的 benchmark stage：

- frozen evaluator 的系统级 ranking 没有人类验证；
- benchmark 只报告 reference-CDF calibrated scores，可能压缩模型差异。

## Response to W1: Human Validation of Benchmark Ranking

### Draft Response

Thank you for pointing out the need to validate the replay-based system ranking beyond per-turn agreement on original responses.
We agree that PersTurnBench should not be interpreted as a definitive replacement for direct user studies.
In the revision, we will make this scope more explicit and describe PersTurnBench as a reproducible automatic screening benchmark for comparing candidate models under fixed personalized conversation states.

We are preparing a small stratified validation over replayed candidate responses.
Annotators will be shown the user profile, task background, conversation prefix, current user request, and two candidate responses, and will be asked which response is more likely to satisfy the target user.
We have implemented the item construction and annotation interface for this validation.
The item sampler builds blinded A/B comparisons from scored replay outputs and stratifies pairs by model-pair difficulty, evaluator score margin, source task, and user coverage.
The annotation page hides model names, evaluator scores, and gold labels, and asks annotators for A/B/tie/uncertain preference, confidence, and optional reason categories.
If completed in time, we will report human-evaluator agreement overall, agreement on high-margin pairs, inter-annotator agreement, and bootstrap confidence intervals over items.
If the validation is not completed before the response deadline, we will still revise the paper to describe this as the most important next validation step rather than implying that the current automatic ranking is final.

### TODO

- [ ] 如果完成人类验证，补具体结果：
  - sample size；
  - annotator 数量；
  - overall agreement；
  - high-margin agreement；
  - bootstrap CI。
- [ ] 如果来不及做人类验证，改成 limitation/future work 版本：

> We agree that direct human validation of counterfactual candidate responses is the most important next step. In the revision, we will explicitly state this limitation and avoid presenting PersTurnBench as a final human preference leaderboard.

## Response to W2: Calibrated-Only Benchmark Results

### Confirmed Result

We agree that calibrated-only reporting can obscure score-scale effects.
We therefore re-examined the replay results under raw scoring, reference mean-shift calibration, and reference-CDF calibration.
The results show that calibration changes the exact order of close models, but the broad tiers remain stable.

| Model | Raw user macro | RefMS user macro | RefCDF user macro | RefCDF low-side rate |
|---|---:|---:|---:|---:|
| kimi-k2.6 | 4.8400 | 4.8535 | 4.8121 | 0.0133 |
| glm-5.1 | 4.8415 | 4.8273 | 4.8003 | 0.0267 |
| deepseek-v4-pro | 4.8024 | 4.8153 | 4.7911 | 0.0267 |
| gpt-5.5 | 4.7213 | 4.7578 | 4.6893 | 0.0567 |
| claude-opus-4-7 | 4.5553 | 4.6338 | 4.6361 | 0.0467 |
| gemini-3.1-pro-preview | 4.6310 | 4.6516 | 4.6283 | 0.0600 |
| minimax-m2.7 | 4.2057 | 4.3621 | 4.4256 | 0.1367 |

The rank correlation with the reference-CDF leaderboard is high:

- raw vs reference-CDF: Kendall's $\tau=0.8095$;
- reference mean-shift vs reference-CDF: Kendall's $\tau=0.9048$;
- reference-transfer-CDF vs reference-CDF: Kendall's $\tau=0.9048$.

### Draft Response

We agree that reporting only calibrated scores can obscure score-scale effects.
In the revision, we will add raw and reference mean-shift benchmark results together with the reference-CDF view and report Kendall rank correlations across these leaderboards.
The exact order of close models changes slightly, but the broad model tiers remain stable: the strongest group remains `kimi-k2.6`, `glm-5.1`, and `deepseek-v4-pro`, the middle group remains close, and `minimax-m2.7` remains clearly separated.
We will therefore revise the text to describe PersTurnBench results as tiered screening signals rather than a strict human preference ranking.

### Planned Paper Revision

- Add sensitivity table to appendix or Section 5.
- Add Kendall's tau values.
- Replace strict ranking wording with broad groups / tiers.
- Make clear that reference-CDF is the official reported score because it reduces score inflation, not because raw scores are ignored.
