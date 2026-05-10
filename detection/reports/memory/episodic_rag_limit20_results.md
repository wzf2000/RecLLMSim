# Episodic RAG Limit-20 Results

## Setup

Evaluated two raw episodic-memory RAG variants on the same 20-user subset:

- `outputs/personalized/episodic_rag_nearest_boundary_k6_limit20.jsonl`
- `outputs/personalized/qwen3_8b_test_episodic_rag_boundary_k6_limit20.jsonl`

Both use:

- `memory_version=episodic_rag`
- `memory_update_mode=episodic_rag`
- `retrieval_strategy=boundary_paired`
- `top_k=6`
- `n=1594` target turns from 20 users

Gold distribution:

| score | count |
|---:|---:|
| 1 | 38 |
| 2 | 72 |
| 3 | 181 |
| 4 | 611 |
| 5 | 692 |

Gold SAT/DSAT distribution:

- SAT (`score>=4`): 1303
- DSAT (`score<=3`): 291

## Main Metrics

| method | n | MAE | RMSE | Pearson | Spearman | QWK | SAT Acc | F1-macro | F1-DSAT | pred dist |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| episodic nearest | 1594 | 0.6769 | 0.9626 | 0.3195 | 0.3348 | 0.2846 | 0.7666 | 0.5925 | 0.3261 | 3:261, 4:1014, 5:319 |
| Qwen episodic RAG | 1594 | 0.7095 | 0.9710 | N/A | N/A | 0.0000 | 0.8174 | 0.4498 | 0.0000 | 4:1594 |

The Qwen LLM version predicts every turn as score 4. Its binary SAT accuracy is
high only because the subset is SAT-heavy; it has no dissatisfied detection
ability.

## Same-Subset Comparison With Existing Methods

Filtered existing outputs to the same 1594 `sample_id`s:

| method | MAE | RMSE | Pearson | Spearman | QWK | SAT Acc | F1-macro | F1-DSAT | pred dist |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Qwen v2 none | 0.7077 | 1.0031 | 0.2999 | 0.3012 | 0.2807 | 0.7629 | 0.6027 | 0.3505 | 1:1, 2:23, 3:267, 4:924, 5:379 |
| selective refute v2 fullscale u20 | 0.8124 | 1.1420 | 0.2331 | 0.2296 | 0.2271 | 0.7290 | 0.5979 | 0.3684 | 1:4, 2:97, 3:292, 4:743, 5:458 |
| history prior delta v3 episodic anchor4 u20 | 0.6048 | 0.9114 | 0.4016 | 0.4211 | 0.3690 | 0.7936 | 0.6117 | 0.3459 | 3:212, 4:919, 5:463 |
| history prior delta v3 episodic twopass u20 | 0.6368 | 0.9455 | 0.3611 | 0.3695 | 0.3353 | 0.7754 | 0.6078 | 0.3514 | 3:261, 4:898, 5:435 |
| raw episodic nearest | 0.6769 | 0.9626 | 0.3195 | 0.3348 | 0.2846 | 0.7666 | 0.5925 | 0.3261 | 3:261, 4:1014, 5:319 |
| raw episodic Qwen prompt | 0.7095 | 0.9710 | N/A | N/A | 0.0000 | 0.8174 | 0.4498 | 0.0000 | 4:1594 |

## Diagnostics

Average retrieved score mix is identical between nearest and Qwen prompt runs,
because they share retrieval:

| retrieved score | average share |
|---:|---:|
| 1 | 0.042 |
| 2 | 0.082 |
| 3 | 0.270 |
| 4 | 0.265 |
| 5 | 0.341 |

The `boundary_paired` retriever does retrieve both sides:

- `boundary_sat`: 5797 retrieved examples
- `boundary_dsat`: 3767 retrieved examples

Therefore the all-4 collapse in Qwen is mainly a prompt/decision policy issue,
not an absence of DSAT evidence in retrieval.

Qwen output diagnostics:

- `reason_prediction`: all `满意`
- `episodic_boundary_side`: all `sat`
- `evidence_confidence`: 1583 `medium`, 11 `high`
- Typical analysis says the current response is "matched with historical 4-point
  examples but not clearly above them", which drives all predictions to 4.

Nearest baseline error pattern:

| gold -> pred | count |
|---|---:|
| 5 -> 4 | 408 |
| 3 -> 4 | 130 |
| 4 -> 3 | 113 |
| 4 -> 5 | 87 |
| 5 -> 3 | 58 |
| 2 -> 4 | 41 |
| 2 -> 3 | 31 |
| 1 -> 4 | 24 |

Nearest baseline by task MAE:

| task | MAE |
|---|---:|
| 旅行规划 | 0.635 |
| 技能学习规划 | 0.672 |
| 菜谱规划 | 0.699 |
| 礼物准备 | 0.707 |

Qwen prompt by task MAE:

| task | MAE |
|---|---:|
| 旅行规划 | 0.685 |
| 技能学习规划 | 0.699 |
| 礼物准备 | 0.709 |
| 菜谱规划 | 0.756 |

## Takeaways

1. Raw episodic retrieval contains useful signal: the no-LLM nearest baseline is
   not random and roughly matches Qwen v2 none on global metrics.
2. Current Qwen episodic prompt is not usable as-is because it collapses to score
   4 and loses all user-aware/ranking information.
3. The best current direction remains `history_prior_delta_v3_episodic_anchor4_u20`
   among comparable 20-user experiments.
4. A better raw episodic RAG prompt should force a two-step decision:
   first classify relative to retrieved DSAT/SAT contrastive evidence, then
   map to 1-5 with explicit rules preventing default-to-4.
5. Retrieval itself should also be improved: `boundary_paired` provides both
   sides, but similarity is still lexical and may retrieve high-level comparable
   turns rather than turns with comparable response-quality failure modes.

## Recommended Next Change

Use the existing raw episodic index, but replace the LLM scoring prompt with a
delta-style reconstruction:

- compute a numeric prior from retrieved scores in code;
- expose retrieved mean, nearest score, DSAT/SAT counts, and score-5 evidence;
- ask Qwen to output only a constrained adjustment `delta_score` from `-1/0/+1`
  plus boundary evidence flags;
- reconstruct final score in code.

This would reuse the strongest lesson from `history_prior_delta_*`: Qwen is more
stable when it judges relative adjustment rather than absolute 1-5 score.

## Boundary-First Follow-Up

A follow-up run tested `episodic_rag_boundary_first` on the same 20-user subset:

- `outputs/personalized/qwen3_8b_test_episodic_rag_boundary_first_k6_limit20.jsonl`
- `n=1594`, all expected records completed after rerun

Main outcome:

| method | MAE | RMSE | Pearson | Spearman | QWK | SAT Acc | F1-macro | F1-DSAT | pred dist |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| episodic absolute prompt | 0.7095 | 0.9710 | N/A | N/A | 0.0000 | 0.8174 | 0.4498 | 0.0000 | 4:1594 |
| episodic boundary-first | 0.7102 | 0.9714 | 0.0042 | 0.0118 | 0.0002 | 0.8168 | 0.4496 | 0.0000 | 3:1, 4:1593 |

Boundary-first therefore did not solve the score-4 collapse. It predicted:

- `boundary_decision=sat`: 1593
- `boundary_decision=dsat`: 1
- `score_refinement=qualified_sat`: 1593
- `score_refinement=near_boundary_dsat`: 1

All 291 true dissatisfied turns were predicted as satisfied. Example analyses
show the model often acknowledges missing details but still maps them to
`qualified_sat=4`, e.g. "未提供完整制作流程 / 油温控制 / 糖醋汁比例" is treated
as a weakness inside SAT rather than a DSAT reason.

Retrieval diagnostics suggest the raw retrieved score mix does contain signal:

| subset | mean retrieved score | avg DSAT count in top-6 | avg score-5 count in top-6 | avg top retrieved score |
|---|---:|---:|---:|---:|
| gold DSAT | 3.513 | 2.952 | 1.268 | 2.677 |
| gold SAT | 3.840 | 2.232 | 2.200 | 2.912 |

This means retrieval is not enough by itself, but it is also not completely
uninformative. The failure is mainly in the LLM decision mapping: Qwen uses
"has some useful content" as sufficient for SAT, while the annotation boundary
penalizes unresolved concrete requirements more strongly.

Updated recommendation:

1. Do not continue prompt-only boundary-first variants in the same form.
2. Move DSAT/SAT decision partly into code using retrieved score statistics or a
   constrained pairwise comparison task.
3. Prefer either:
   - `retrieval prior + delta reconstruction`; or
   - a two-call contrastive classifier that forces the model to select whether
     the current turn is closer to retrieved DSAT or SAT exemplars, without
     allowing a direct `qualified_sat` default.
