# Static Replay Reference-Based Calibration

## Motivation

The original post-hoc calibration script calibrates only the predictions present
in the input file. For hard static replay, many `(user, target_task)` blocks
contain only one replay record, so block-wise CDF or mean-shift calibration often
falls back to identity.

This is avoidable because we already have a full Qwen3-8B V2 none prediction
file on all original target responses:

- `outputs/personalized/Qwen_Qwen3-8B_test_none.jsonl`

That file contains 6474 original-response predictions across the same
personalized test split. It can provide per-user/task ranking context for every
static replay record.

## Implementation

Added:

- `eval/static_replay_reference_calibrate.py`
- `scripts/calibrate_static_replay_reference.sh`

The script keeps the existing `eval/calibrate.py` unchanged and adds static
replay specific methods.

## Methods

| Method | Idea | Coverage |
|---|---|---:|
| `reference_cdf` | For each `(user, target_task)`, rank full original Qwen predictions and replay predictions together, then map replay ranks to the user's history-score CDF. | 300/300 |
| `reference_mean_shift` | Estimate block-level score shift from full original Qwen predictions to user history mean, then apply the shift to replay predictions. | 300/300 |
| `reference_transfer` | Use original Qwen raw predictions and their already calibrated version to learn a raw-score to calibrated-score mapping, then apply it to replay predictions. | 300/300 |

For `reference_transfer`, the current run used:

- raw reference: `outputs/personalized/Qwen_Qwen3-8B_test_none.jsonl`
- calibrated reference: `outputs/personalized/Qwen_Qwen3-8B_test_none_calCDF.jsonl`

## Example Commands

```bash
cd detection
input=outputs/static_replay/gpt-5.5_test_hard_scored_by_qwen3_8b_v2.jsonl \
method=reference_cdf \
output=outputs/static_replay/gpt-5.5_test_hard_scored_by_qwen3_8b_v2_refCDF.jsonl \
bash scripts/calibrate_static_replay_reference.sh

input_jsonl=outputs/static_replay/gpt-5.5_test_hard_scored_by_qwen3_8b_v2_refCDF.jsonl \
output_json=outputs/static_replay/gpt-5.5_test_hard_scored_by_qwen3_8b_v2_refCDF_eval.json \
bash scripts/eval_static_replay.sh
```

For transfer calibration:

```bash
cd detection
input=outputs/static_replay/gpt-5.5_test_hard_scored_by_qwen3_8b_v2.jsonl \
method=reference_transfer \
reference_calibrated=outputs/personalized/Qwen_Qwen3-8B_test_none_calCDF.jsonl \
output=outputs/static_replay/gpt-5.5_test_hard_scored_by_qwen3_8b_v2_refTransferCDF.jsonl \
bash scripts/calibrate_static_replay_reference.sh
```

## Results

All seven candidate models were calibrated with all three methods. Each run
calibrated all 300 records and all 180 blocks, with no identity fallback.

### Reference CDF

| Model | Method | n | Micro | User macro | SAT rate | Dist 1/2/3/4/5 |
|---|---:|---:|---:|---:|---:|---|
| kimi-k2.6 | refCDF | 300 | 4.7367 | 4.8121 | 0.9867 | 0/0/4/71/225 |
| glm-5.1 | refCDF | 300 | 4.7067 | 4.8003 | 0.9733 | 1/1/6/69/223 |
| deepseek-v4-pro | refCDF | 300 | 4.6667 | 4.7911 | 0.9733 | 0/2/6/82/210 |
| gpt-5.5 | refCDF | 300 | 4.5733 | 4.6893 | 0.9433 | 1/6/10/86/197 |
| claude-opus-4-7 | refCDF | 300 | 4.5400 | 4.6361 | 0.9533 | 1/2/11/106/180 |
| gemini-3.1-pro-preview | refCDF | 300 | 4.5267 | 4.6283 | 0.9400 | 1/3/14/101/181 |
| minimax-m2.7 | refCDF | 300 | 4.2633 | 4.4256 | 0.8633 | 3/10/28/123/136 |

### Reference Mean Shift

| Model | Method | n | Micro | User macro | SAT rate | Dist 1/2/3/4/5 |
|---|---:|---:|---:|---:|---:|---|
| kimi-k2.6 | refMS | 300 | 4.8133 | 4.8535 | 0.9967 | 0/0/1/54/245 |
| glm-5.1 | refMS | 300 | 4.7867 | 4.8273 | 0.9833 | 0/1/4/53/242 |
| deepseek-v4-pro | refMS | 300 | 4.7533 | 4.8153 | 0.9733 | 0/0/8/58/234 |
| gpt-5.5 | refMS | 300 | 4.6667 | 4.7578 | 0.9633 | 0/0/11/78/211 |
| gemini-3.1-pro-preview | refMS | 300 | 4.5833 | 4.6516 | 0.9733 | 0/1/7/108/184 |
| claude-opus-4-7 | refMS | 300 | 4.5833 | 4.6338 | 0.9800 | 0/0/6/113/181 |
| minimax-m2.7 | refMS | 300 | 4.2633 | 4.3621 | 0.8800 | 0/3/33/146/118 |

### Reference Transfer From Qwen calCDF

| Model | Method | n | Micro | User macro | SAT rate | Dist 1/2/3/4/5 |
|---|---:|---:|---:|---:|---:|---|
| kimi-k2.6 | refTransferCDF | 300 | 4.6933 | 4.7575 | 1.0000 | 0/0/0/92/208 |
| deepseek-v4-pro | refTransferCDF | 300 | 4.6500 | 4.7460 | 0.9900 | 0/1/2/98/199 |
| glm-5.1 | refTransferCDF | 300 | 4.6667 | 4.7422 | 0.9933 | 0/1/1/95/203 |
| gpt-5.5 | refTransferCDF | 300 | 4.5967 | 4.6903 | 0.9767 | 0/3/4/104/189 |
| claude-opus-4-7 | refTransferCDF | 300 | 4.5533 | 4.6319 | 0.9800 | 0/1/5/121/173 |
| gemini-3.1-pro-preview | refTransferCDF | 300 | 4.5300 | 4.6102 | 0.9800 | 0/1/5/128/166 |
| minimax-m2.7 | refTransferCDF | 300 | 4.3033 | 4.4480 | 0.9100 | 1/9/17/144/129 |

## Interpretation

- Reference-based calibration solves the coverage problem: every replay record
  can be calibrated because the full original-response predictions provide
  block-level context.
- `reference_cdf` is the most defensible default for static replay reporting.
  It uses the full original prediction set as the ranking background and avoids
  forcing the small replay subset to match the user's history distribution by
  itself.
- `reference_mean_shift` is very mild and often remains close to raw scores.
  It is useful as a conservative sensitivity check, but it does not reduce the
  high-score ceiling effect much.
- `reference_transfer` is easy to interpret but coarse because the raw Qwen
  scores are integer buckets. It mostly preserves high SAT rates and should be
  treated as an auxiliary calibration view rather than the main one.
- Across reference methods, `kimi-k2.6`, `glm-5.1`, and `deepseek-v4-pro` remain
  the top tier, `gpt-5.5` remains a stable middle-high model, and
  `minimax-m2.7` remains last.
