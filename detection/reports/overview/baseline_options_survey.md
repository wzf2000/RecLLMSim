# Baseline Options for Personalized Satisfaction Prediction

## Context

Current Qwen V2 experiments optimize a training-free personalized satisfaction
predictor on RecLLMSim:

- target: turn-level 1-5 satisfaction prediction plus SAT/DSAT boundary
- split: cross-task personalized split, where each `(user, target_task)` block
  uses the same user's other tasks as labeled history
- strongest current internal references:
  - `Qwen none` / memory v2 raw
  - `Qwen none + MS/CDF`
  - `boundary_34_selective_refute_v2`
  - `memv3_two_stage_v2`

The missing piece is a set of comparable basic baselines from adjacent user
satisfaction / dialogue evaluation work.

## Baseline Families

### 1. Distribution and history baselines

These are cheap and should be included before external methods.

Recommended variants:

| Baseline | Output | Uses user history? | Notes |
|---|---|---:|---|
| global majority / global mean | 1-5 or binary | no | lower bound |
| task mean | 1-5 | no | controls task-level score bias |
| user history mean | 1-5 | yes | predicts all target turns with source-task user mean |
| user history empirical CDF sampler/ranker | 1-5 | yes | distribution-only personalized baseline |
| nearest historical turn label | 1-5 | yes | TF-IDF anchor without LLM reasoning |

Why useful:

- They separate "personalized distribution prior" from "LLM semantic judging".
- `Qwen + CDF` already shows distribution calibration is strong, so these
  baselines are necessary to prove the model adds semantic signal beyond history
  statistics.

Implementation effort: very low. Most can be computed from
`PersonalizedSample.history_sessions` and target records without LLM calls.

### 2. SPUR-style supervised rubric baseline

Source: Lin et al., ACL 2024, "Interpretable User Satisfaction Estimation for
Conversational Systems with Large Language Models".

Paper link:
`https://aclanthology.org/2024.acl-long.598/`

Local implementation:

- `detection/eval/spur.py`
- `detection/scripts/run_spur.sh`

Current local SPUR behavior:

- task: binary SAT/DSAT only (`score >= 4` vs `score <= 3`)
- split: user-group train / valid / test over all turns
- Phase 1: extract SAT/DSAT rubric candidates from labeled train examples
- Phase 2: summarize to `k` global SAT and DSAT rubrics
- Phase 3: apply global rubrics with an LLM
- optional Phase 4: train logistic regression over rubric-match features and
  text embeddings

Feasibility:

- Good baseline for the 3/4 boundary task.
- Not a direct baseline for full 1-5 prediction.
- Not personalized in the current implementation.
- The current split is not aligned with the cross-task personalized split, so
  results should not be compared directly against Qwen V2 unless the data split
  is adapted.

Recommended adaptation:

1. Keep SPUR as a global supervised boundary baseline.
2. Rebuild its train rows from RecLLMSim training users only, then score the
   same cross-task test records used by Qwen V2.
3. Export records with `sample_id`, `gold_score`, `pred_score` mapped as
   `SAT -> 4`, `DSAT -> 3`, and `pred_label`, so `eval/personalized.py` and
   `eval/binary_sat.py` can compare it against existing boundary metrics.
4. Report SPUR only on boundary metrics:
   `F1-DSAT`, `false_sat_rate`, boundary kappa, AUC, PU-bin metrics.

Variants worth running:

- `SPUR-direct`: LLM directly applies learned rubrics.
- `SPUR-rubric-LR`: logistic regression over rubric match features.
- `SPUR-embedding-LR`: embedding-only logistic regression.
- `SPUR-combined`: rubric match + embedding logistic regression.

Not recommended as first pass:

- per-user SPUR rubrics. Each user has too few labeled history examples and
  extracting/summarizing per user would be expensive and unstable.

### 3. Supervised RecLLMSim predictors

Local code already exists:

- `detection/predictor/bert.py`
- `detection/predictor/bert_ordinal.py`
- `detection/predictor/lora.py`
- `detection/predictor/lora_ordinal.py`

Feasibility:

- Strong supervised baseline for full 1-5 prediction.
- Uses labels from train users, not training-free.
- Does not use per-user cross-task memory unless user/history features are
  explicitly added.

Recommended variants:

| Baseline | Output | Priority |
|---|---|---:|
| BERT regression + reason head | 1-5 | medium |
| BERT ordinal | 1-5 | high |
| Qwen LoRA ordinal | 1-5 | high |
| Qwen LoRA ordinal without profile/reason | 1-5 | medium |

Why useful:

- Gives a supervised upper/reference line against the training-free Qwen V2
  agent.
- Existing code already computes MAE/RMSE/Pearson/Spearman/QWK.

Important caveat:

- Current predictor split is user-group train/valid/test, not necessarily the
  same cross-task personalized test records. For fair comparison, add an eval
  mode that emits JSONL on the exact personalized test sample_ids.

### 4. Generic LLM-as-judge baselines

Feasible variants:

| Baseline | Output | Notes |
|---|---|---|
| zero-shot generic judge | 1-5 | no user memory; same prompt for all users |
| few-shot global judge | 1-5 or binary | examples sampled from train users |
| task-rubric judge | 1-5 | one rubric per task, no per-user memory |
| Prometheus-style rubric judge | 1-5 | open evaluator LM / custom score rubric |

Sources:

- Prometheus: `https://huggingface.co/papers/2310.08491`
- Prometheus 2: `https://huggingface.co/papers/2405.01535`

Feasibility:

- Very easy if implemented as prompt variants using the existing
  `collect_personalized.py` scoring loop.
- Good for showing whether personalized memory beats a strong generic LLM
  evaluator.
- Prometheus-style evaluators are designed for custom rubrics, but they are
  general response evaluators, not user satisfaction models. Treat them as
  generic judge baselines, not as satisfaction-specific prior work.

Recommended first variant:

- `generic_1_5_llm_judge`: same input as no-memory but with a clearer 1-5
  rubric and no profile/history-derived memory.

### 5. General dialogue evaluation metrics

These methods are adjacent, but less directly aligned with personalized
satisfaction.

| Method | Source | Fit to current task |
|---|---|---|
| RUBER | AAAI 2018 | weak; reference/unreferenced response quality, not satisfaction |
| FED | SIGDIAL 2020 | medium for dialogue quality; no user history |
| GRADE | EMNLP 2020 | weak/medium; coherence-focused |
| DialogRPT | EMNLP 2020 | medium; predicts engagement-style feedback |
| UniEval | EMNLP 2022 | medium; multi-dimensional NLG/dialogue evaluator |
| DynaEval / FlowEval | ACL/EMNLP 2021-2022 | weak; dialogue coherence/flow, not user satisfaction |

Sources:

- RUBER: `https://aaai.org/papers/11321-ruber-an-unsupervised-method-for-automatic-evaluation-of-open-domain-dialog-systems/`
- FED: `https://aclanthology.org/2020.sigdial-1.28/`
- GRADE: `https://aclanthology.org/2020.emnlp-main.742/`
- DialogRPT: `https://aclanthology.org/2020.emnlp-main.28/`
- UniEval: `https://aclanthology.org/2022.emnlp-main.131/`

Feasibility ranking:

1. DialogRPT / UniEval: easiest to run as off-the-shelf scoring baselines if
   dependencies are acceptable.
2. FED: possible, but older DialoGPT-based assumptions may be brittle on Chinese
   task-oriented/planning dialogues.
3. RUBER/GRADE/DynaEval/FlowEval: more engineering for weaker conceptual fit.

Recommended use:

- Include at most one or two generic dialogue evaluator baselines.
- Do not over-invest unless the paper narrative needs broad dialogue-evaluation
  comparison.

### 6. Task-oriented satisfaction modeling baselines

Relevant work:

- USS: "Simulating User Satisfaction for the Evaluation of Task-oriented
  Dialogue Systems" introduces 5-level satisfaction labels and baselines such as
  feature models, hierarchical GRU, and BERT.
  Source: `https://huggingface.co/papers/2105.03748`
- SG-USM: schema-guided user satisfaction modeling for TOD.
  Source: `https://aclanthology.org/2023.acl-long.116/`
- CAUSE: counterfactual robustness assessment for TOD satisfaction estimation.
  Source: `https://aclanthology.org/2024.findings-acl.871/`

Fit:

- USS-style BERT / hierarchical models are relevant as supervised satisfaction
  predictors.
- SG-USM depends on structured task schema and goal fulfillment. RecLLMSim tasks
  do not expose TOD schema/slots, so migration would require artificial schema
  extraction.
- CAUSE is more an evaluation/augmentation benchmark than a direct baseline.

Recommended use:

- Cite USS as closest supervised 5-level satisfaction baseline family.
- Implement BERT/ordinal or LoRA/ordinal locally rather than reproducing
  hierarchical GRU from scratch.

## Recommended Baseline Set

### Minimal publishable set

1. `global_mean` / `user_history_mean`
2. `Qwen no_memory`
3. `generic_1_5_llm_judge`
4. `SPUR-direct` for SAT/DSAT only
5. `SPUR-combined` if embeddings are available
6. `BERT ordinal` or `Qwen LoRA ordinal` supervised baseline

### Stronger but still practical set

1. all minimal baselines
2. `nearest_history_turn`
3. `task_mean`
4. `task-level rubric judge`
5. `DialogRPT` or `UniEval`

## Priority

Highest priority:

1. Add statistical/history baselines.
2. Adapt SPUR to the personalized test record format and boundary metrics.
3. Add one supervised ordinal baseline emitted as JSONL on the same sample_ids.

Medium priority:

4. Generic 1-5 LLM judge / task-rubric judge.
5. DialogRPT or UniEval as a generic dialogue-quality baseline.

Low priority:

6. RUBER/GRADE/FED/DynaEval/FlowEval unless a broad dialogue-evaluation
   comparison is required.
7. Per-user SPUR, due to history sparsity and high LLM cost.

## Main Conclusion

SPUR is convenient and relevant, but only as a global supervised SAT/DSAT
baseline. It should not be presented as a direct competitor to Qwen V2 full 1-5
personalized memory unless adapted to the same cross-task test records and
reported with boundary metrics.

For full 1-5 prediction, the most useful basic baselines are simpler:

- history/statistical baselines to isolate distribution priors
- supervised ordinal BERT/LoRA baselines to show where training-free methods
  stand relative to trained predictors
- generic no-memory LLM judges to show the value of personalization
