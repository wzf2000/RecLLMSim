# Static Replay Pairwise Comparison vs GPT-5.5

## Goal

Absolute static replay scores are useful, but LLM benchmarks often also report
relative comparisons. This report adds pairwise win/tie/lose comparisons against
`gpt-5.5`, used here as a fixed general-capability reference model.

Implemented:

- `eval/static_replay_pairwise.py`
- `scripts/compare_static_replay_pairwise.sh`

The script aligns two scored static replay JSONL files by `sample_id` and
compares `pred_score` on each shared replay case.

## Metrics

For each candidate model versus `gpt-5.5`:

- `win`: candidate score > GPT-5.5 score
- `tie`: candidate score = GPT-5.5 score
- `lose`: candidate score < GPT-5.5 score
- `nt_win`: non-tie win rate, i.e. `win / (win + lose)`
- `delta`: mean score difference, candidate minus GPT-5.5
- `user_nt`: non-tie win rate after aggregating scores per user
- `block_nt`: non-tie win rate after aggregating scores per `(user,target_task)`

Because scores are discrete 1-5 integers, ties are common. `nt_win` is therefore
important for interpreting the decided cases.

## Example Command

```bash
cd detection
reference_file=outputs/static_replay/gpt-5.5_test_hard_scored_by_qwen3_8b_v2_refCDF.jsonl \
reference_name=gpt-5.5 \
candidate_files="glm-5.1=outputs/static_replay/glm-5.1_test_hard_scored_by_qwen3_8b_v2_refCDF.jsonl kimi-k2.6=outputs/static_replay/kimi-k2.6_test_hard_scored_by_qwen3_8b_v2_refCDF.jsonl" \
output_json=outputs/static_replay/pairwise_vs_gpt55_refCDF.json \
bash scripts/compare_static_replay_pairwise.sh
```

## Raw Pairwise Results

| Candidate | Reference | n | Win | Tie | Lose | nt_win | Delta | user_nt | block_nt |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| kimi-k2.6 | gpt-5.5 | 300 | 0.2233 | 0.7267 | 0.0500 | 0.8171 | 0.1767 | 0.7674 | 0.8182 |
| glm-5.1 | gpt-5.5 | 300 | 0.1967 | 0.7567 | 0.0467 | 0.8082 | 0.1533 | 0.8250 | 0.8226 |
| deepseek-v4-pro | gpt-5.5 | 300 | 0.1900 | 0.7200 | 0.0900 | 0.6786 | 0.1067 | 0.7027 | 0.6984 |
| gemini-3.1-pro-preview | gpt-5.5 | 300 | 0.1300 | 0.6633 | 0.2067 | 0.3861 | -0.0733 | 0.3571 | 0.3733 |
| claude-opus-4-7 | gpt-5.5 | 300 | 0.1267 | 0.6533 | 0.2200 | 0.3654 | -0.0967 | 0.2857 | 0.3247 |
| minimax-m2.7 | gpt-5.5 | 300 | 0.0500 | 0.4833 | 0.4667 | 0.0968 | -0.4733 | 0.0308 | 0.1034 |

## calCDF Pairwise Results

| Candidate | Reference | n | Win | Tie | Lose | nt_win | Delta | user_nt | block_nt |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| glm-5.1 | gpt-5.5 | 300 | 0.1267 | 0.7900 | 0.0833 | 0.6032 | 0.0433 | 0.7619 | 0.7826 |
| kimi-k2.6 | gpt-5.5 | 300 | 0.1333 | 0.7733 | 0.0933 | 0.5882 | 0.0400 | 0.7391 | 0.7500 |
| deepseek-v4-pro | gpt-5.5 | 300 | 0.1367 | 0.7633 | 0.1000 | 0.5775 | 0.0400 | 0.7895 | 0.7391 |
| gemini-3.1-pro-preview | gpt-5.5 | 300 | 0.1533 | 0.6700 | 0.1767 | 0.4646 | -0.0333 | 0.3571 | 0.3548 |
| claude-opus-4-7 | gpt-5.5 | 300 | 0.1400 | 0.6700 | 0.1900 | 0.4242 | -0.0533 | 0.2812 | 0.2647 |
| minimax-m2.7 | gpt-5.5 | 300 | 0.1033 | 0.6400 | 0.2567 | 0.2870 | -0.1633 | 0.0465 | 0.1071 |

## refCDF Pairwise Results

| Candidate | Reference | n | Win | Tie | Lose | nt_win | Delta | user_nt | block_nt |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| kimi-k2.6 | gpt-5.5 | 300 | 0.1467 | 0.8233 | 0.0300 | 0.8302 | 0.1633 | 0.8065 | 0.8444 |
| glm-5.1 | gpt-5.5 | 300 | 0.1300 | 0.8433 | 0.0267 | 0.8298 | 0.1333 | 0.8667 | 0.8571 |
| deepseek-v4-pro | gpt-5.5 | 300 | 0.1167 | 0.8200 | 0.0633 | 0.6481 | 0.0933 | 0.7500 | 0.6667 |
| gemini-3.1-pro-preview | gpt-5.5 | 300 | 0.0900 | 0.7700 | 0.1400 | 0.3913 | -0.0467 | 0.3947 | 0.3704 |
| claude-opus-4-7 | gpt-5.5 | 300 | 0.0767 | 0.7967 | 0.1267 | 0.3770 | -0.0333 | 0.4242 | 0.3600 |
| minimax-m2.7 | gpt-5.5 | 300 | 0.0400 | 0.6767 | 0.2833 | 0.1237 | -0.3100 | 0.0889 | 0.1139 |

## Interpretation

- The top tier is stable in pairwise form: `kimi-k2.6`, `glm-5.1`, and
  `deepseek-v4-pro` beat `gpt-5.5` on decided cases under raw and refCDF.
- `glm-5.1` and `kimi-k2.6` are very close. `kimi-k2.6` has slightly higher
  micro non-tie win rate under raw/refCDF, while `glm-5.1` has slightly stronger
  user/block macro non-tie win rates in several views.
- `gemini-3.1-pro-preview` and `claude-opus-4-7` are below `gpt-5.5` in
  pairwise decided cases despite many ties.
- `minimax-m2.7` is consistently worse than `gpt-5.5` across all views.
- `calCDF` compresses differences and creates many ties, so refCDF is currently
  the more useful relative view for static replay: it uses full original Qwen
  predictions as calibration context while preserving more model-level
  separation than subset-only calCDF.
