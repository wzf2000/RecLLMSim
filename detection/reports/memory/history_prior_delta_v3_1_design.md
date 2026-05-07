# History Prior Delta V3.1 Design

## Motivation

The first `history_prior_delta_v3` subset20 run restored DSAT discovery compared
with v2, but the DSAT vote still had weak precision:

```text
dsat_signal_votes >= 2: marked=252, DSAT precision=0.373, DSAT recall=0.323
```

Breaking down the vote combinations showed that the 2-vote bucket was small and
noisy:

```text
('B4', 'R3', 'Dneg') n=20, DSAT rate=0.250
('B3', 'R4', 'Dneg') n=1,  DSAT rate=0.000
```

Most useful DSAT discovery came from 3-vote cases:

```text
('B3', 'R3', 'Dneg') n=231, DSAT rate=0.385
```

## Change

New prompt version:

```text
turn_eval_prompt_version=history_prior_delta_v3_1
```

V3.1 keeps the same schema and diagnostic fields as v3, but tightens the
code-side DSAT trigger:

```text
score = round(history_prior_score)

dsat_votes =
  (boundary_score == 3)
  + (history_prior_delta_raw_score <= 3)
  + (delta_score < 0)

if dsat_votes >= 3:
    score = min(score, 3)
elif delta_confidence == "high":
    score += sign(delta_score)
elif delta_confidence == "medium" and abs(delta_score) == 2:
    score += sign(delta_score)

if boundary_confidence == "high" and boundary_score == 4:
    score = max(score, 4)

score = clip(score, 1, 5)
```

Compared with v3, 2-vote cases remain visible through `dsat_signal_votes` but no
longer force the final score to 3.

## Offline Replay on V3 Subset20 Outputs

Applying the stricter trigger to existing v3 diagnostic fields gives:

| variant | MAE | RMSE | Pearson | Spearman | QWK | Boundary Acc | F1-DSAT | DSAT Recall | pred DSAT | False SAT |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| v3 current (`votes>=2`) | 0.6311 | 0.9298 | 0.3999 | 0.4179 | 0.3777 | 0.7723 | 0.3816 | 0.3849 | 296 | 0.6151 |
| v3.1 replay (`votes>=3`) | 0.6267 | 0.9261 | 0.4001 | 0.4163 | 0.3764 | 0.7767 | 0.3754 | 0.3677 | 279 | 0.6323 |

Expected tradeoff:

- slightly better MAE / RMSE / binary accuracy
- slightly lower DSAT recall / F1-DSAT
- fewer noisy 2-vote false DSAT cases

## Suggested Run

```bash
cd detection
turn_eval_prompt_version=history_prior_delta_v3_1 \
memory_update_mode=none \
n_anchors=3 \
limit_users=20 \
output_jsonl=outputs/personalized/history_prior_delta_v3_1_none_n3_limit20.jsonl \
bash scripts/collect_personalized_vllm.sh
```

