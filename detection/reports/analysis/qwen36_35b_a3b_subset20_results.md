# Qwen3.6-35B-A3B Subset-20 Results

## Setup

Backbone migration smoke/validation run:

- Model: `Qwen/Qwen3.6-35B-A3B`
- Output: `outputs/personalized/Qwen_Qwen3.6-35B-A3B_test_none_v2_u20.jsonl`
- Prompt: `turn_eval_prompt_version=v2`
- Memory mode: `memory_update_mode=none`
- Users: 20
- Records: 1594

Gold distribution:

| score | count |
|---:|---:|
| 1 | 38 |
| 2 | 72 |
| 3 | 181 |
| 4 | 611 |
| 5 | 692 |

Prediction distribution:

| score | count |
|---:|---:|
| 1 | 7 |
| 2 | 81 |
| 3 | 408 |
| 4 | 834 |
| 5 | 264 |

Compared with Qwen3-8B, Qwen3.6-35B-A3B is much stricter: it predicts many more
DSAT-side scores (`<=3`) and many fewer 5s.

## Main Metrics

All comparisons below are filtered to the same 1594 `sample_id`s.

| method | MAE | RMSE | Pearson | Spearman | QWK | SAT Acc | F1-macro | F1-DSAT | False SAT | False DSAT |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen3.6 raw v2 none | 0.7842 | 1.0756 | 0.3411 | 0.3335 | 0.3083 | 0.7120 | 0.6128 | **0.4168** | **127** | 332 |
| Qwen3-8B raw v2 none | 0.7077 | 1.0031 | 0.2999 | 0.3012 | 0.2807 | 0.7629 | 0.6027 | 0.3505 | 189 | 189 |
| Qwen3-8B + mean-shift | 0.6556 | 0.9736 | 0.3529 | 0.3737 | 0.3386 | 0.7930 | 0.6343 | 0.3934 | 184 | 146 |
| Qwen3-8B + CDF | 0.6719 | 1.0665 | 0.3384 | 0.3760 | 0.3374 | 0.7898 | 0.6387 | 0.4050 | 177 | 158 |
| HPD v3 episodic anchor4 | **0.6048** | **0.9114** | **0.4016** | 0.4211 | 0.3690 | **0.7936** | 0.6117 | 0.3459 | 204 | **125** |
| HPD v3 episodic two-pass | 0.6368 | 0.9455 | 0.3611 | 0.3695 | 0.3353 | 0.7754 | 0.6078 | 0.3514 | 194 | 164 |

Raw Qwen3.6 does not improve MAE, but it improves rank/ordinal signal over raw
Qwen3-8B:

- Pearson: `0.2999 -> 0.3411`
- Spearman: `0.3012 -> 0.3335`
- QWK: `0.2807 -> 0.3083`
- F1-DSAT: `0.3505 -> 0.4168`

The cost is many more false DSAT errors:

- False SAT decreases: `189 -> 127`
- False DSAT increases: `189 -> 332`

## Calibration

Both user-history calibration methods were run with all 80 blocks calibrated:

- `outputs/personalized/Qwen_Qwen3.6-35B-A3B_test_none_v2_u20_calMS.jsonl`
- `outputs/personalized/Qwen_Qwen3.6-35B-A3B_test_none_v2_u20_calCDF.jsonl`

| method | MAE | RMSE | Pearson | Spearman | QWK | SAT Acc | F1-macro | F1-DSAT | False SAT | False DSAT | pred dist |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Qwen3.6 raw | 0.7842 | 1.0756 | 0.3411 | 0.3335 | 0.3083 | 0.7120 | 0.6128 | 0.4168 | 127 | 332 | 1:7, 2:81, 3:408, 4:834, 5:264 |
| Qwen3.6 + mean-shift | 0.6631 | **0.9826** | 0.3828 | 0.4028 | 0.3758 | 0.7754 | 0.6407 | **0.4207** | 161 | 197 | 1:2, 2:36, 3:289, 4:688, 5:579 |
| Qwen3.6 + CDF | **0.6405** | 1.0177 | **0.3977** | **0.4238** | **0.3966** | 0.7923 | **0.6430** | 0.4121 | 175 | **156** | 1:21, 2:70, 3:181, 4:646, 5:676 |

Calibration is strongly beneficial for Qwen3.6. The raw model has useful
relative signal but poor absolute calibration. CDF is the best post-hoc variant
for ordinal/ranking metrics and nearly matches the best HPD v3 episodic method
on MAE while beating it on QWK.

## Error Shape

Confusion highlights:

| gold -> pred | count |
|---|---:|
| 5 -> 4 | 388 |
| 4 -> 3 | 177 |
| 5 -> 3 | 111 |
| 3 -> 4 | 78 |
| 4 -> 5 | 71 |
| 2 -> 3 | 33 |
| 2 -> 4 | 27 |
| 4 -> 2 | 27 |

Mean prediction bias by gold score:

| gold | n | mean bias | MAE |
|---:|---:|---:|---:|
| 1 | 38 | +2.053 | 2.053 |
| 2 | 72 | +1.250 | 1.250 |
| 3 | 181 | +0.409 | 0.652 |
| 4 | 611 | -0.272 | 0.504 |
| 5 | 692 | -0.948 | 0.948 |

The model compresses extremes toward the middle: low gold scores are too high,
while 5s are often downgraded to 4 or 3.

Mean bias by task:

| task | n | mean bias | MAE |
|---|---:|---:|---:|
| 技能学习规划 | 335 | -0.281 | 0.770 |
| 旅行规划 | 463 | -0.397 | 0.734 |
| 礼物准备 | 460 | -0.491 | 0.870 |
| 菜谱规划 | 336 | -0.226 | 0.750 |

`礼物准备` is the weakest task and shows the strongest under-scoring.

## Interpretation

Qwen3.6-35B-A3B is not a drop-in improvement for raw absolute 1-5 scoring. It is
more critical than Qwen3-8B and hurts raw MAE, but it improves:

- relative correlation,
- ordinal agreement,
- DSAT recall / F1,
- false-SAT control.

This is consistent with a stronger judge that can identify more weaknesses but
needs calibration to match the dataset's user-specific score scale.

## Recommended Next Experiments

1. Run `history_prior_delta_v3_episodic_anchor4` with Qwen3.6 on the same 20-user
   subset. This tests whether the stronger backbone improves the currently best
   relative-reconstruction pipeline.
2. Run Qwen3.6 `v2 none` on the full set only after confirming the HPD variant;
   raw v2 alone is not compelling enough for full expensive evaluation.
3. Always evaluate Qwen3.6 with both raw and CDF/mean-shift calibration. The
   raw output underestimates high scores and over-predicts DSAT.
4. If optimizing boundary metrics, Qwen3.6 raw is promising because it has the
   best DSAT F1 in this subset, but it needs a mechanism to reduce false DSAT.
