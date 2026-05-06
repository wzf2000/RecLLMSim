# GPT-5.4 Mini V2 None Limit-10 Calibration Results

## Setup

This report evaluates post-hoc calibration on the same 10-user subset used in
`detection/reports/overview/gpt54mini_v2_none_limit10_results.md`.

Raw run:

- model: `gpt-5.4-mini-2026-03-17`
- turn eval prompt: `v2`
- memory update mode: `none`
- records: `828`
- raw output:
  `detection/outputs/personalized/gpt-5.4-mini-2026-03-17_test_v2_none_limit10.jsonl`

Calibration commands:

```bash
cd detection
input=outputs/personalized/gpt-5.4-mini-2026-03-17_test_v2_none_limit10.jsonl \
output=outputs/personalized/gpt-5.4-mini-2026-03-17_test_v2_none_limit10_calMS.jsonl \
method=mean_shift \
bash scripts/calibrate.sh

input=outputs/personalized/gpt-5.4-mini-2026-03-17_test_v2_none_limit10.jsonl \
output=outputs/personalized/gpt-5.4-mini-2026-03-17_test_v2_none_limit10_calCDF.jsonl \
method=cdf \
bash scripts/calibrate.sh
```

Calibration coverage:

- `mean_shift`: `40 / 40` blocks calibrated, no identity fallback
- `cdf`: `40 / 40` blocks calibrated, no identity fallback

Fair-comparison outputs are in:

- `detection/outputs/personalized/gpt54mini_limit10_calibration_compare/`

Evaluation file:

- `detection/outputs/personalized/gpt54mini_limit10_calibration_compare/eval.json`

## Main Metrics

| Method | MAE | RMSE | Pearson | Spearman | QWK |
|---|---:|---:|---:|---:|---:|
| `gpt54_raw` | `0.6461` | `0.9267` | `0.3667` | `0.3410` | `0.3340` |
| `gpt54_ms` | **`0.5761`** | **`0.8881`** | `0.4002` | `0.4057` | `0.3874` |
| `gpt54_cdf` | `0.6014` | `0.9593` | **`0.4129`** | **`0.4238`** | **`0.4125`** |
| `qwen_raw` | `0.6727` | `0.9587` | `0.2710` | `0.2795` | `0.2551` |
| `qwen_ms` | `0.6051` | `0.9162` | `0.3727` | `0.4039` | `0.3633` |
| `qwen_cdf` | `0.6473` | `1.0333` | `0.3187` | `0.3683` | `0.3185` |
| `gpt4o_raw` | `0.6473` | `0.9479` | `0.2697` | `0.2707` | `0.2574` |
| `gpt4o_ms` | `0.6147` | `0.9319` | `0.2933` | `0.3123` | `0.2772` |
| `gpt4o_cdf` | `0.6413` | `1.0185` | `0.3195` | `0.3679` | `0.3194` |

Calibration improves all three LLM no-memory runs, but the improvement is
largest and cleanest for `gpt-5.4-mini`.

For `gpt-5.4-mini`:

- `mean_shift` is best for exact 1-5 error:
  `MAE 0.6461 -> 0.5761`.
- `cdf` is best for ordinal/ranking quality:
  `Pearson 0.3667 -> 0.4129`,
  `Spearman 0.3410 -> 0.4238`,
  `QWK 0.3340 -> 0.4125`.

## Boundary Metrics

| Method | Acc | F1-DSAT | Kappa | AUC | False SAT | False DSAT |
|---|---:|---:|---:|---:|---:|---:|
| `gpt54_raw` | `0.7705` | `0.4172` | `0.2763` | `0.6886` | **`0.5342`** | `0.1642` |
| `gpt54_ms` | `0.7899` | `0.4238` | `0.2955` | `0.7008` | `0.5616` | `0.1349` |
| `gpt54_cdf` | **`0.7971`** | **`0.4362`** | **`0.3126`** | **`0.7291`** | `0.5548` | **`0.1276`** |
| `qwen_raw` | `0.7488` | `0.3376` | `0.1835` | `0.6534` | `0.6370` | `0.1686` |
| `qwen_ms` | `0.7826` | `0.4231` | `0.2898` | `0.7072` | `0.5479` | `0.1466` |
| `qwen_cdf` | `0.7826` | `0.3960` | `0.2635` | `0.6855` | `0.5959` | `0.1364` |
| `gpt4o_raw` | `0.7754` | `0.2619` | `0.1333` | `0.6359` | `0.7740` | `0.1070` |
| `gpt4o_ms` | `0.7935` | `0.2723` | `0.1602` | `0.6439` | `0.7808` | `0.0836` |
| `gpt4o_cdf` | `0.7766` | `0.3599` | `0.2245` | `0.6629` | `0.6438` | `0.1334` |

`gpt54_cdf` is the strongest overall boundary variant in this comparison. It
improves F1-DSAT from `0.4172` to `0.4362`, kappa from `0.2763` to `0.3126`,
and AUC from `0.6886` to `0.7291`.

Compared with calibrated Qwen:

- `gpt54_ms` and `qwen_ms` have nearly tied F1-DSAT
  (`0.4238` vs `0.4231`), but `gpt54_ms` has better MAE and QWK.
- `gpt54_cdf` is better than `qwen_cdf` on all listed global and boundary
  metrics in this subset.

Compared with calibrated gpt-4o-mini:

- `gpt4o_cdf` fixes part of the raw model's SAT-heavy behavior, but still trails
  `gpt54_cdf` by a large margin on F1-DSAT, kappa, AUC, and QWK.

## Distribution Shift

Gold distribution on this subset:

| Score | Gold |
|---|---:|
| 1 | `11` |
| 2 | `41` |
| 3 | `94` |
| 4 | `378` |
| 5 | `304` |

Prediction distributions:

| Method | 1 | 2 | 3 | 4 | 5 |
|---|---:|---:|---:|---:|---:|
| `gpt54_raw` | `2` | `29` | `149` | `551` | `97` |
| `gpt54_ms` | `0` | `19` | `137` | `475` | `197` |
| `gpt54_cdf` | `13` | `38` | `101` | `395` | `281` |
| `qwen_raw` | `0` | `10` | `158` | `502` | `158` |
| `qwen_ms` | `0` | `20` | `146` | `452` | `210` |
| `qwen_cdf` | `13` | `38` | `101` | `395` | `281` |
| `gpt4o_raw` | `2` | `7` | `97` | `500` | `222` |
| `gpt4o_ms` | `1` | `6` | `82` | `484` | `255` |
| `gpt4o_cdf` | `9` | `37` | `97` | `387` | `298` |

`gpt54_raw` is strongly underusing score `5`. Mean shift partially corrects
that while preserving relatively conservative behavior. CDF almost matches the
gold marginal distribution, which explains its strong ranking/QWK and boundary
metrics.

## Direct Change Analysis

For `gpt-5.4-mini`:

- `mean_shift`: changed `138 / 828` predictions, `98` better, `40` worse,
  net `+58`.
- `cdf`: changed `322 / 828` predictions, `179` better, `143` worse,
  net `+36`.

Mean shift makes fewer and safer edits. CDF makes broader edits and is better
for distribution/ranking/boundary, but it also introduces more per-turn losses.

## Interpretation

Adding post-hoc calibration changes the conclusion from the raw model-only
comparison:

1. `gpt-5.4-mini + mean_shift` is the best exact-score LLM variant in this
   subset. It reaches `MAE=0.5761`, close to `nearest3=0.5918` and much closer
   to the history-only mean baseline (`0.5447`) than the raw LLM result.
2. `gpt-5.4-mini + CDF` is the best semantic/ranking/boundary variant. It has
   the best Pearson, Spearman, QWK, F1-DSAT, boundary kappa, and AUC among all
   compared LLM variants.
3. Model capacity and calibration are complementary. The stronger raw judge
   gives better within-block ordering; calibration then maps that ordering onto
   the user's historical score scale.

Recommended use:

- If optimizing full 1-5 MAE: use `gpt-5.4-mini + mean_shift`.
- If optimizing user-specific ranking and SAT/DSAT boundary: use
  `gpt-5.4-mini + CDF`.
- For benchmark-style Static Replay scoring, report both raw and calibrated
  variants separately, because CDF imposes the historical user score
  distribution mechanically.
