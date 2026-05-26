# URS Episodic Retrieval Memory + Task-Guarded Prompt

## Motivation

The previous URS evaluator mainly used summary-style `UserMemory`, where a user's cross-intent history is compressed into a small memory schema before prediction.
Recent URS experiments showed that summary memory is often weak or noisy, especially when task intents differ.
This version keeps each historical URS session as a separate retrievable memory item and uses the retrieved raw cases as evidence for the current session-level satisfaction prediction.

## Implementation

Added files:

- `detection/lib/urs_episodic.py`: builds a per-user TF-IDF retrieval index over raw historical URS sessions.
- `detection/lib/urs_episodic_prompts.py`: builds `urs_episodic_task_guarded`, a prompt that combines episodic evidence with the existing `urs_v2_calibrated_task_guarded` calibration and task-aware guard.
- `detection/trace/urs/episodic_runner.py`: runs URS prediction with retrieved episodic memories.
- `detection/trace/collect_urs_episodic_rag.py`: collection entrypoint.
- `detection/scripts/collect_urs_episodic_rag.sh`: runnable shell wrapper.

Modified file:

- `detection/trace/urs/session_eval.py`: added `evaluate_urs_session_episodic` while keeping the original summary-memory and no-memory routes unchanged.

## Retrieval Design

Each historical session from `sample.history_sessions` becomes one memory item with:

- source intent and task context
- full dialogue text
- gold session-level score
- gold reason label
- retrieval similarity and evidence role

Supported retrieval strategies:

- `topk_similar`: use the most similar historical sessions.
- `boundary_paired`: retrieve both DSAT (`score <= 3`) and SAT (`score >= 4`) evidence when available, then fill remaining slots by similarity.
- `score_balanced`: retrieve from low-score, score-4, and score-5 buckets.
- `nearest`: same retrieval as `topk_similar`, intended for nearest-neighbor style diagnostics.

The default is `boundary_paired` with `top_k=4`, because current URS evaluator tuning is most sensitive to the 3/4 satisfied-vs-dissatisfied boundary.

## Prompt Design

Prompt version: `urs_episodic_task_guarded`.

The prompt reuses the existing `urs_v2_calibrated_task_guarded` calibration and task-aware guard, then adds an episodic evidence procedure:

1. First judge the current session using current-session evidence and task-aware guard.
2. Compare the current session with retrieved SAT and DSAT cases.
3. Allow episodic evidence to affect the 3/4 boundary only when the retrieved case is specifically relevant to the current intent, request, or defect type.
4. Do not mechanically copy historical scores.
5. Keep the same strict JSON output schema: `classification`, `reason`, `analysis`.

## Example Run

Start local Qwen3-8B vLLM separately, then run:

```bash
model=Qwen/Qwen3-8B \
vllm_base_url=http://localhost:8001/v1 \
vllm_api_key=EMPTY \
split=test \
retrieval_strategy=boundary_paired \
top_k=4 \
urs_prompt_version=urs_episodic_task_guarded \
max_workers=4 \
output_jsonl=outputs/urs/Qwen_Qwen3-8B_test_urs_episodic_task_guarded_boundary_k4.jsonl \
bash scripts/collect_urs_episodic_rag.sh
```

Small smoke test:

```bash
model=Qwen/Qwen3-8B \
vllm_base_url=http://localhost:8001/v1 \
vllm_api_key=EMPTY \
split=test \
retrieval_strategy=boundary_paired \
top_k=4 \
limit=10 \
max_workers=2 \
output_jsonl=outputs/urs/Qwen_Qwen3-8B_test_urs_episodic_task_guarded_boundary_k4_smoke.jsonl \
bash scripts/collect_urs_episodic_rag.sh
```

Evaluation:

```bash
result_file=outputs/urs/Qwen_Qwen3-8B_test_urs_episodic_task_guarded_boundary_k4.jsonl \
output_json=outputs/urs/Qwen_Qwen3-8B_test_urs_episodic_task_guarded_boundary_k4_eval.json \
bash scripts/eval_urs_predictor.sh
```

Direct comparison against task-guarded v1:

```bash
result_files="task_guarded=outputs/urs/Qwen_Qwen3-8B_test_none_urs_v2_calibrated_task_guarded.jsonl episodic=outputs/urs/Qwen_Qwen3-8B_test_urs_episodic_task_guarded_boundary_k4.jsonl" \
output_json=outputs/urs/qwen3_8b_urs_task_guarded_vs_episodic.json \
bash scripts/eval_urs_predictor.sh
```

## Expected Use

This should be treated as a memory baseline for URS, not an immediate replacement for `urs_v2_calibrated_task_guarded`.
The key diagnostic question is whether raw retrieved evidence improves F1-DSAT and rank correlation without increasing false SAT too much.

## Full-Test Result on Qwen3-8B

Run output:

- `detection/outputs/urs/Qwen_Qwen3-8B_test_urs_episodic_task_guarded_boundary_k4.jsonl`
- `detection/outputs/urs/Qwen_Qwen3-8B_test_urs_episodic_task_guarded_boundary_k4_eval.json`

Overall result on URS test (`n=584`):

| Method | MAE | Pearson | Spearman | QWK | Boundary Acc | F1-DSAT | FalseSAT |
|---|---:|---:|---:|---:|---:|---:|---:|
| `urs_v2_calibrated_task_guarded` | 0.6952 | 0.2760 | 0.2883 | 0.2424 | 0.6627 | 0.5207 | 0.4780 |
| `urs_episodic_task_guarded`, `boundary_paired`, `k=4` | 0.7295 | 0.2347 | 0.2360 | 0.2031 | 0.6695 | 0.3754 | 0.7171 |

Prediction distribution for the episodic version:

- Gold score distribution: `1:19, 2:47, 3:139, 4:250, 5:129`
- Predicted score distribution: `1:1, 2:14, 3:89, 4:415, 5:65`
- Gold SAT/DSAT: `379 / 205`
- Predicted SAT/DSAT: `480 / 104`

Task-level observations:

- `retrieval` is the strongest task: Pearson `0.4801`, Spearman `0.4096`, QWK `0.3727`, F1-DSAT `0.4872`.
- `leisure` has relatively balanced boundary behavior: F1-DSAT `0.5714`.
- `creative` is the weakest boundary task: F1-DSAT `0.0714`, mostly because the model nearly never predicts DSAT for this task.
- `advice`, `creative`, and `professional` have high false-SAT rates, suggesting episodic evidence is not strong enough to counter the base prompt's SAT tendency.

Interpretation:

The first episodic-retrieval version does not outperform the current best URS evaluator.
It slightly improves boundary accuracy over `task_guarded_v1`, but this comes from over-predicting SAT rather than better DSAT detection.
The major regression is F1-DSAT: the model predicts only 104 DSAT cases while the gold set contains 205 DSAT cases.
This means raw retrieved episodes are currently being used more as reassurance evidence than as failure evidence.

Recommended next step:

Do not replace `urs_v2_calibrated_task_guarded` with this version.
If continuing this direction, the prompt should force a separate retrieved-DSAT evidence check before allowing `classification >= 4`, and the retriever should expose the closest DSAT case more explicitly instead of mixing all evidence in one block.

## DSAT-First Prompt Variant

Added prompt version:

- `urs_episodic_task_guarded_dsat_first`

This keeps the same retriever and output schema, but changes how retrieved evidence is presented and used:

- retrieved evidence is split into `Closest DSAT failure evidence` and `Closest SAT success evidence`
- the prompt first asks the model to inspect DSAT failures before allowing a SAT score
- if `classification >= 4`, the analysis must explain why the current session is not the same kind of failure as the closest DSAT evidence
- if the current session shares a concrete failure type with DSAT evidence, such as missing the core request, missing a required artifact/source/product/route/format, being generic, unverifiable, or requiring follow-up to complete the task, the prompt says it cannot output 4/5
- the prompt also guards against mechanical downgrading: DSAT evidence can only affect the boundary when the current intent/request/defect is specifically similar

Run command:

```bash
model=Qwen/Qwen3-8B \
vllm_base_url=http://localhost:8001/v1 \
vllm_api_key=EMPTY \
split=test \
retrieval_strategy=boundary_paired \
top_k=4 \
urs_prompt_version=urs_episodic_task_guarded_dsat_first \
max_workers=4 \
output_jsonl=outputs/urs/Qwen_Qwen3-8B_test_urs_episodic_task_guarded_dsat_first_boundary_k4.jsonl \
bash scripts/collect_urs_episodic_rag.sh
```

Evaluation command:

```bash
result_files="task_guarded=outputs/urs/Qwen_Qwen3-8B_test_none_urs_v2_calibrated_task_guarded.jsonl episodic_dsat_first=outputs/urs/Qwen_Qwen3-8B_test_urs_episodic_task_guarded_dsat_first_boundary_k4.jsonl" \
output_json=outputs/urs/qwen3_8b_urs_task_guarded_vs_episodic_dsat_first.json \
bash scripts/eval_urs_predictor.sh
```

## DSAT-First Full-Test Result

Run output:

- `detection/outputs/urs/Qwen_Qwen3-8B_test_urs_episodic_task_guarded_dsat_first_boundary_k4.jsonl`

Computed result on URS test (`n=584`):

| Method | MAE | Pearson | Spearman | QWK | Boundary Acc | F1-DSAT | FalseSAT | Pred SAT/DSAT |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `urs_v2_calibrated_task_guarded` | 0.6952 | 0.2760 | 0.2883 | 0.2424 | 0.6627 | 0.5207 | 0.4780 | 378 / 206 |
| `urs_episodic_task_guarded`, `boundary_paired`, `k=4` | 0.7295 | 0.2347 | 0.2360 | 0.2031 | 0.6695 | 0.3754 | 0.7171 | 480 / 104 |
| `urs_episodic_task_guarded_dsat_first`, `boundary_paired`, `k=4` | 0.7346 | 0.2175 | 0.1952 | 0.1760 | 0.6661 | 0.3253 | 0.7707 | 500 / 84 |

Prediction distribution for `dsat_first`:

- Gold score distribution: `1:19, 2:47, 3:139, 4:250, 5:129`
- Predicted score distribution: `1:1, 2:3, 3:80, 4:419, 5:81`
- Gold SAT/DSAT: `379 / 205`
- Predicted SAT/DSAT: `500 / 84`

Task-level DSAT-first observations:

| Task | n | MAE | Spearman | QWK | F1-DSAT | FalseSAT | Pred SAT/DSAT |
|---|---:|---:|---:|---:|---:|---:|---:|
| `advice` | 96 | 0.7083 | 0.0856 | 0.1403 | 0.2308 | 0.8378 | 81 / 15 |
| `creative` | 64 | 0.7969 | -0.0811 | -0.0872 | 0.1481 | 0.9130 | 60 / 4 |
| `leisure` | 48 | 0.6875 | 0.2305 | 0.2231 | 0.4706 | 0.5556 | 32 / 16 |
| `other` | 8 | 1.1250 | -0.5988 | -0.6190 | 0.2857 | 0.7500 | 5 / 3 |
| `professional` | 117 | 0.8205 | 0.2331 | 0.1879 | 0.2951 | 0.8085 | 103 / 14 |
| `retrieval` | 165 | 0.6909 | 0.2710 | 0.2380 | 0.3939 | 0.7234 | 146 / 19 |
| `text` | 86 | 0.6744 | 0.3472 | 0.2596 | 0.3810 | 0.7241 | 73 / 13 |

Compared with the previous episodic prompt, `dsat_first` moved only `16` previous SAT predictions to DSAT, while moving `36` previous DSAT predictions to SAT.
Thus the explicit DSAT-first instruction did not increase failure sensitivity; it made the model more willing to justify SAT.

Manual inspection of false-SAT analyses shows a repeated pattern:

- The model often writes that the current answer satisfies the core request in `StepD1`.
- It then states that no DSAT failure type is matched, even when retrieved DSAT scores are present.
- It uses generic SAT evidence such as "structured solution" or "directly answers the request" to justify `classification=4`.
- The required `why_not_dsat_failure` behavior is not reliably expressed as a real comparison against the closest DSAT evidence; it often becomes a template sentence.

Conclusion:

The bottleneck is not only evidence formatting.
The evaluator has a strong prior that "directly answers the request" implies SAT, and prompt-level DSAT-first wording is insufficient to make it use retrieved failure cases as a real boundary comparator.
Further prompt strengthening may continue to hurt global ranking by adding verbosity without changing the actual decision rule.

Recommended follow-up:

- Do not continue with single-pass DSAT-first as the main URS direction.
- If episodic retrieval is still explored, use a two-stage route: first ask for a binary `same_failure_as_dsat_evidence` decision with forced evidence IDs, then score only when this binary decision is stable.
- A cheaper alternative is task-specific calibration on top of `urs_v2_calibrated_task_guarded`, because the current best clean method still dominates both episodic variants on MAE, correlation, QWK, and F1-DSAT.

## Two-Stage DSAT Failure Gate Variant

Added prompt version:

- `urs_episodic_task_guarded_dsat_twostage`

This version separates failure matching from final scoring:

1. Stage 1 uses `build_urs_dsat_failure_check_prompt` to output:
   - `same_failure_as_dsat_evidence`
   - `matched_evidence_ids`
   - `failure_type`
   - `confidence`
   - `analysis`
2. Stage 2 uses `build_urs_episodic_twostage_score_prompt` to score the session with the Stage-1 decision injected as a hard constraint.
3. Program-side gate: if Stage 1 returns `same_failure_as_dsat_evidence=true` with `confidence=medium/high`, and Stage 2 still predicts `classification>=4`, the final score is clamped to `3`.

The output JSONL keeps the standard fields and adds diagnostic metadata:

- `dsat_failure_check`
- `dsat_gate_applied`
- `raw_pred_score_before_dsat_gate`

Run command:

```bash
model=Qwen/Qwen3-8B \
vllm_base_url=http://localhost:8001/v1 \
vllm_api_key=EMPTY \
split=test \
retrieval_strategy=boundary_paired \
top_k=4 \
urs_prompt_version=urs_episodic_task_guarded_dsat_twostage \
max_workers=4 \
output_jsonl=outputs/urs/Qwen_Qwen3-8B_test_urs_episodic_task_guarded_dsat_twostage_boundary_k4.jsonl \
bash scripts/collect_urs_episodic_rag.sh
```

Evaluation command:

```bash
result_files="task_guarded=outputs/urs/Qwen_Qwen3-8B_test_none_urs_v2_calibrated_task_guarded.jsonl episodic_dsat_twostage=outputs/urs/Qwen_Qwen3-8B_test_urs_episodic_task_guarded_dsat_twostage_boundary_k4.jsonl" \
output_json=outputs/urs/qwen3_8b_urs_task_guarded_vs_episodic_dsat_twostage.json \
bash scripts/eval_urs_predictor.sh
```

## Two-Stage Full-Test Result

Run output:

- `detection/outputs/urs/Qwen_Qwen3-8B_test_urs_episodic_task_guarded_dsat_twostage_boundary_k4.jsonl`

Computed result on URS test (`n=584`):

| Method | MAE | Pearson | Spearman | QWK | Boundary Acc | F1-DSAT | FalseSAT | Pred SAT/DSAT |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `urs_v2_calibrated_task_guarded` | 0.6952 | 0.2760 | 0.2883 | 0.2424 | 0.6627 | 0.5207 | 0.4780 | 378 / 206 |
| `urs_episodic_task_guarded`, `boundary_paired`, `k=4` | 0.7295 | 0.2347 | 0.2360 | 0.2031 | 0.6695 | 0.3754 | 0.7171 | 480 / 104 |
| `urs_episodic_task_guarded_dsat_first`, `boundary_paired`, `k=4` | 0.7346 | 0.2175 | 0.1952 | 0.1760 | 0.6661 | 0.3253 | 0.7707 | 500 / 84 |
| `urs_episodic_task_guarded_dsat_twostage`, `boundary_paired`, `k=4` | 0.8579 | 0.2454 | 0.2443 | 0.2416 | 0.6233 | 0.5582 | 0.3220 | 291 / 293 |

Prediction distribution for `dsat_twostage`:

- Gold score distribution: `1:19, 2:47, 3:139, 4:250, 5:129`
- Predicted score distribution: `1:12, 2:23, 3:258, 4:187, 5:104`
- Gold SAT/DSAT: `379 / 205`
- Predicted SAT/DSAT: `291 / 293`

Diagnostics:

- `dsat_gate_applied=0`: Stage 2 already followed Stage 1 whenever a medium/high same-failure decision mattered, so the program-side clamp did not need to fire.
- Stage 1 predicted `same_failure_as_dsat_evidence=true` for `163/584` cases and `false` for `421/584` cases.
- Stage 1 confidence distribution: `high=438`, `medium=141`, `low=5`.
- Stage 1 as a standalone SAT/DSAT gate gives Acc `0.6507`, F1-DSAT `0.4457`, precision-DSAT `0.5031`, recall-DSAT `0.4000`, FalseSAT `0.6000`.
- The final two-stage scorer is more aggressive than Stage 1 alone: it predicts `293` DSAT cases and reaches recall-DSAT `0.6780`.

Task-level two-stage result:

| Task | n | MAE | Spearman | QWK | F1-DSAT | FalseSAT | Pred SAT/DSAT |
|---|---:|---:|---:|---:|---:|---:|---:|
| `advice` | 96 | 0.8229 | 0.1565 | 0.1433 | 0.5714 | 0.1892 | 28 / 68 |
| `creative` | 64 | 0.8750 | 0.1906 | 0.0727 | 0.5714 | 0.3913 | 38 / 26 |
| `leisure` | 48 | 0.8958 | 0.1882 | 0.2047 | 0.5532 | 0.2778 | 19 / 29 |
| `other` | 8 | 1.1250 | -0.2069 | -0.2258 | 0.6000 | 0.2500 | 2 / 6 |
| `professional` | 117 | 0.8547 | 0.2782 | 0.3108 | 0.5882 | 0.2553 | 45 / 72 |
| `retrieval` | 165 | 0.8000 | 0.2700 | 0.3280 | 0.4906 | 0.4468 | 106 / 59 |
| `text` | 86 | 0.9535 | 0.2359 | 0.2209 | 0.5806 | 0.3793 | 53 / 33 |

Interpretation:

Two-stage fixes the earlier episodic variants' main failure mode: it no longer collapses toward SAT.
It improves F1-DSAT from `0.5207` to `0.5582` over the current best clean `task_guarded_v1`, and it cuts FalseSAT from `0.4780` to `0.3220`.
However, it overcorrects toward DSAT: predicted DSAT count is `293` while gold DSAT count is `205`.
This hurts MAE (`0.8579` vs `0.6952`) and boundary accuracy (`0.6233` vs `0.6627`), even though QWK remains close (`0.2416` vs `0.2424`).

Conclusion:

The two-stage failure-check design is useful as a high-recall DSAT detector, but it is not a better final 1-5 URS evaluator.
The evidence-matching signal may be valuable as an auxiliary feature or selective override, but not as a hard global scoring route.

Recommended follow-up:

- Use `urs_v2_calibrated_task_guarded` as the clean default final evaluator.
- If using two-stage, apply it selectively only on high-risk tasks or high-uncertainty samples instead of all samples.
- A practical next variant is selective arbitration: start from `task_guarded_v1`, run Stage-1 failure check only when `task_guarded_v1` predicts `4` but the retrieved DSAT evidence is close, then downgrade only for `same_failure=true, confidence=high`.
