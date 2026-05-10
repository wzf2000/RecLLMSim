# Static Replay Hard Benchmark Model Comparison

## Setup

This report summarizes the current hard-mode static replay benchmark results
for seven candidate response models. Candidate responses were collected with the
hard static replay selection mode and scored by the current judge:

- judge: `Qwen/Qwen3-8B`
- judge prompt: `v2`
- judge memory update: `none`
- memory version: `v2`
- benchmark size: 300 replay turns per model

The compared candidate models are:

- `claude-opus-4-7`
- `deepseek-v4-pro`
- `gemini-3.1-pro-preview`
- `gpt-5.5`
- `glm-5.1`
- `kimi-k2.6`
- `minimax-m2.7`

Post-hoc calibration was added with the existing personalized calibration
script:

- `calCDF`: block-wise rank-to-user-history-CDF mapping
- `calMS`: block-wise mean-shift to user-history mean

For this static replay subset, calibration was applied to 75 out of 180
`(user, target_task, judge_memory_model)` blocks, covering 195 out of 300
records for each model. The remaining 105 single-record blocks fell back to
identity because CDF/mean-shift calibration requires at least two predictions in
the block.

Note: the original `gpt-5.5` trace contained 321 scored records from an earlier
hard-selection state. The 21 records outside the canonical 300-turn hard subset
were removed from the active response/scored files before computing the numbers
below. Backups were kept as `*.bak_extra21_20260510`.

## Output Files

Raw scored files:

- `outputs/static_replay/claude-opus-4-7_test_hard_scored_by_qwen3_8b_v2.jsonl`
- `outputs/static_replay/deepseek-v4-pro_test_hard_scored_by_qwen3_8b_v2.jsonl`
- `outputs/static_replay/gemini-3.1-pro-preview_test_hard_scored_by_qwen3_8b_v2.jsonl`
- `outputs/static_replay/gpt-5.5_test_hard_scored_by_qwen3_8b_v2.jsonl`
- `outputs/static_replay/glm-5.1_test_hard_scored_by_qwen3_8b_v2.jsonl`
- `outputs/static_replay/kimi-k2.6_test_hard_scored_by_qwen3_8b_v2.jsonl`
- `outputs/static_replay/minimax-m2.7_test_hard_scored_by_qwen3_8b_v2.jsonl`

For each raw file, corresponding `_calCDF.jsonl`, `_calMS.jsonl`, and
`*_eval.json` files were generated in `outputs/static_replay/`.

## Main Comparison

Rows are grouped by scoring method and sorted by `user_macro_mean` within each
method. `Dist 1/2/3/4/5` is the predicted score distribution.

| Model | Method | n | Micro | User macro | Task macro | Block macro | SAT rate | DSAT rate | Dist 1/2/3/4/5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| glm-5.1 | raw | 300 | 4.7900 | 4.8415 | 4.7812 | 4.8167 | 0.9933 | 0.0067 | 0/2/0/57/241 |
| kimi-k2.6 | raw | 300 | 4.8133 | 4.8400 | 4.8026 | 4.8306 | 0.9967 | 0.0033 | 0/0/1/54/245 |
| deepseek-v4-pro | raw | 300 | 4.7433 | 4.8024 | 4.7312 | 4.7815 | 0.9800 | 0.0200 | 0/1/5/64/230 |
| gpt-5.5 | raw | 300 | 4.6367 | 4.7213 | 4.6224 | 4.6731 | 0.9800 | 0.0200 | 0/0/6/97/197 |
| gemini-3.1-pro-preview | raw | 300 | 4.5633 | 4.6310 | 4.5640 | 4.5963 | 0.9900 | 0.0100 | 0/0/3/125/172 |
| claude-opus-4-7 | raw | 300 | 4.5400 | 4.5553 | 4.5271 | 4.5565 | 0.9867 | 0.0133 | 0/1/3/129/167 |
| minimax-m2.7 | raw | 300 | 4.1633 | 4.2057 | 4.1518 | 4.2019 | 0.8700 | 0.1300 | 0/2/37/171/90 |
| glm-5.1 | calCDF | 300 | 4.3367 | 4.6417 | 4.3318 | 4.5269 | 0.8467 | 0.1533 | 1/5/40/100/154 |
| deepseek-v4-pro | calCDF | 300 | 4.3333 | 4.6320 | 4.3322 | 4.5213 | 0.8467 | 0.1533 | 1/4/41/102/152 |
| kimi-k2.6 | calCDF | 300 | 4.3333 | 4.6232 | 4.3298 | 4.5213 | 0.8467 | 0.1533 | 1/4/41/102/152 |
| gpt-5.5 | calCDF | 300 | 4.2933 | 4.5739 | 4.2898 | 4.4546 | 0.8467 | 0.1533 | 1/4/41/114/140 |
| gemini-3.1-pro-preview | calCDF | 300 | 4.2600 | 4.4910 | 4.2590 | 4.3991 | 0.8433 | 0.1567 | 1/4/42/122/131 |
| claude-opus-4-7 | calCDF | 300 | 4.2400 | 4.4315 | 4.2392 | 4.3657 | 0.8500 | 0.1500 | 1/4/40/132/123 |
| minimax-m2.7 | calCDF | 300 | 4.1300 | 4.2012 | 4.1327 | 4.1824 | 0.8267 | 0.1733 | 1/4/47/151/97 |
| glm-5.1 | calMS | 300 | 4.3000 | 4.6158 | 4.2963 | 4.5028 | 0.8900 | 0.1100 | 0/1/32/143/124 |
| kimi-k2.6 | calMS | 300 | 4.3100 | 4.6119 | 4.3142 | 4.5056 | 0.9033 | 0.0967 | 0/0/29/149/122 |
| deepseek-v4-pro | calMS | 300 | 4.2767 | 4.5884 | 4.2828 | 4.4815 | 0.8733 | 0.1267 | 0/1/37/140/122 |
| gpt-5.5 | calMS | 300 | 4.2400 | 4.5380 | 4.2375 | 4.4204 | 0.8667 | 0.1333 | 0/1/39/147/113 |
| gemini-3.1-pro-preview | calMS | 300 | 4.1933 | 4.4527 | 4.1870 | 4.3583 | 0.8533 | 0.1467 | 0/0/44/154/102 |
| claude-opus-4-7 | calMS | 300 | 4.1867 | 4.3897 | 4.1819 | 4.3287 | 0.8733 | 0.1267 | 0/2/36/166/96 |
| minimax-m2.7 | calMS | 300 | 4.1033 | 4.1781 | 4.1067 | 4.1639 | 0.8600 | 0.1400 | 0/2/40/183/75 |

## Rankings

By `user_macro_mean`:

| Rank | Raw | calCDF | calMS |
|---:|---|---|---|
| 1 | glm-5.1 4.8415 | glm-5.1 4.6417 | glm-5.1 4.6158 |
| 2 | kimi-k2.6 4.8400 | deepseek-v4-pro 4.6320 | kimi-k2.6 4.6119 |
| 3 | deepseek-v4-pro 4.8024 | kimi-k2.6 4.6232 | deepseek-v4-pro 4.5884 |
| 4 | gpt-5.5 4.7213 | gpt-5.5 4.5739 | gpt-5.5 4.5380 |
| 5 | gemini-3.1-pro-preview 4.6310 | gemini-3.1-pro-preview 4.4910 | gemini-3.1-pro-preview 4.4527 |
| 6 | claude-opus-4-7 4.5553 | claude-opus-4-7 4.4315 | claude-opus-4-7 4.3897 |
| 7 | minimax-m2.7 4.2057 | minimax-m2.7 4.2012 | minimax-m2.7 4.1781 |

## Observations

- Raw Qwen3-8B V2 none scores are strongly concentrated at 5 for the top
  models. `glm-5.1`, `kimi-k2.6`, and `deepseek-v4-pro` all have raw user-macro
  means above 4.80 and SAT rates above 0.98.
- Calibration substantially reduces the optimistic ceiling effect. Under
  `calCDF`, SAT rates move to roughly 0.83-0.85 for all models except
  `minimax-m2.7`, and the score distribution contains many more 3s.
- The relative ordering is mostly stable. `glm-5.1` is first under all three
  scoring views. `kimi-k2.6` and `deepseek-v4-pro` form the next tier and swap
  order depending on calibration method. `gpt-5.5` is stable at rank 4 after
  filtering to the canonical 300-turn subset.
- `minimax-m2.7` is consistently last under raw, `calCDF`, and `calMS`, and it
  is the only model whose raw SAT rate is already clearly below 0.90.
- `calCDF` is more aggressive than `calMS` in lowering SAT rate and introducing
  low-score mass. For benchmark reporting, raw and `calCDF` should both be
  shown: raw reflects the judge's direct satisfaction estimate, while `calCDF`
  controls for user-level score-style priors and reduces score inflation.

## Reproduction Commands

Example for one model:

```bash
cd detection
input=outputs/static_replay/glm-5.1_test_hard_scored_by_qwen3_8b_v2.jsonl \
method=cdf \
output=outputs/static_replay/glm-5.1_test_hard_scored_by_qwen3_8b_v2_calCDF.jsonl \
bash scripts/calibrate.sh

input_jsonl=outputs/static_replay/glm-5.1_test_hard_scored_by_qwen3_8b_v2_calCDF.jsonl \
output_json=outputs/static_replay/glm-5.1_test_hard_scored_by_qwen3_8b_v2_calCDF_eval.json \
bash scripts/eval_static_replay.sh
```
