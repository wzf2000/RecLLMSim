# ARR Paper Outline: UATSBench and Conversation Satisfaction Evaluator

## Purpose

This document records the updated paper organization after advisor discussion.
The paper should now be framed around **user-aware turn-level conversation satisfaction evaluation**.
The previous framing around a "satisfaction predictor" and "Static Replay Evaluation" should be replaced by:

1. **Conversation Satisfaction Evaluator**: the evaluator we construct and verify.
2. **UATSBench**: a user-aware turn-level conversation satisfaction benchmark for evaluating generation models.
3. **Replay-based benchmark protocol**: a controlled replay protocol, described simply as replay rather than static replay.

Terminology constraints:

- Use **Conversation** instead of **dialogue** whenever referring to the object of evaluation.
- Use **user-aware turn-level** as the main task emphasis.
- Use **Conversation Satisfaction Evaluator** instead of **satisfaction predictor**.
- Avoid foregrounding "static" in the benchmark description.
- Mention replay as the benchmark protocol, not as the paper title-level contribution.

## Updated Positioning

### Candidate Title

Recommended:

> UATSBench: User-Aware Turn-Level Conversation Satisfaction Evaluation with LLM Judges

Alternative titles:

- User-Aware Turn-Level Conversation Satisfaction Evaluation for LLM Assistants
- Building User-Aware Conversation Satisfaction Evaluators for Turn-Level Assistant Evaluation
- UATSBench: A User-Aware Benchmark for Turn-Level Conversation Satisfaction

### Core Claim

Current LLM evaluation often measures generic response quality, but user satisfaction in conversations is user-aware, turn-level, and history-dependent.
We build a **Conversation Satisfaction Evaluator** that estimates user-specific turn-level satisfaction from user history and conversation context.
We verify the evaluator through meta-evaluation against human satisfaction annotations, and then use it to construct **UATSBench**, a benchmark that evaluates arbitrary generation models through replay under fixed user-aware conversation states.

### Contributions

Recommended contribution wording:

1. **Evaluator construction**: We formulate user-aware turn-level conversation satisfaction evaluation and build a Conversation Satisfaction Evaluator that uses user history, conversation context, and assistant responses to estimate 1--5 satisfaction scores and dissatisfaction reasons.
2. **Evaluator verification**: We meta-evaluate the evaluator against human satisfaction annotations on RecLLMSim and auxiliary URS experiments, comparing personalized and non-personalized baselines, calibration variants, and backbone choices.
3. **Benchmark**: We introduce **UATSBench**, a user-aware turn-level conversation satisfaction benchmark where arbitrary generation models are evaluated by replaying fixed conversation states and scoring generated responses with the verified evaluator.

Avoid saying:

- "We introduce a dataset."
- "We propose Static Replay Evaluation" as the central contribution.
- "Satisfaction predictor" as the method name.

## Updated Abstract Direction

Draft direction:

> User satisfaction in assistant conversations is inherently user-aware: the same response may satisfy one user but disappoint another depending on preferences, strictness, task expectations, and prior interaction history.
> Existing automatic evaluation methods mostly measure generic response quality and rarely evaluate user-aware turn-level satisfaction.
> We build a Conversation Satisfaction Evaluator that uses user histories and target conversation context to estimate turn-level 1--5 satisfaction scores and dissatisfaction reasons.
> We verify the evaluator through meta-evaluation against human satisfaction annotations, showing that user-aware memory and score calibration improve ordinal agreement and dissatisfaction-boundary detection over generic LLM-as-a-judge and supervised baselines.
> We then introduce UATSBench, a user-aware turn-level conversation satisfaction benchmark that evaluates arbitrary generation models through replay under fixed user-aware conversation states.
> UATSBench enables controlled comparison of both generic generation models and memory-augmented user-aware systems without collecting new human labels for every candidate model.

## Updated Paper Structure

Recommended main-body structure:

1. Introduction
2. Related Work
3. Conversation Satisfaction Evaluator
4. Personalized Satisfaction Evaluator Verification
5. UATSBench: User-Aware Turn-Level Conversation Satisfaction Benchmark
6. Conclusion and Limitations

Appendix:

- Detailed baseline implementations.
- Detailed metric definitions.
- Full evaluator ablations.
- Pairwise replay comparisons.
- Memory-augmented generation model comparisons.

## 1. Introduction

### Goal

Motivate user-aware turn-level conversation satisfaction as a missing evaluation axis for LLM assistants.

### Suggested Flow

1. LLM assistants are increasingly used in personalized task-oriented conversations.
2. Existing evaluation often measures generic quality, task success, or population-level preference.
3. Real satisfaction is user-aware: preferences, strictness, constraints, and prior interactions change what counts as a satisfactory response.
4. Turn-level evaluation is important because conversation failures are local; a conversation can contain both useful and unsatisfactory assistant turns.
5. Historical satisfaction feedback can be used to build a user-aware evaluator.
6. Once an evaluator is verified, it can be used to evaluate arbitrary generation models through replay.
7. This yields UATSBench: a benchmark for user-aware turn-level conversation satisfaction.

### Notes

- Mention personalization work only as broad motivation in the introduction.
- Do not include a full "personalization and user modeling" related-work section unless it directly concerns evaluation.
- Avoid positioning the data as a new contribution.
- Avoid using "dialogue"; use "conversation".

## 2. Related Work

The updated related work should have three main subsections.

### 2.1 Conversation Satisfaction Estimation

This subsection should explicitly distinguish:

1. **Conversation-level satisfaction**
   - Satisfaction labels at the whole conversation/session level.
   - Useful for overall user experience, but unable to locate local failures.
2. **Turn-level satisfaction**
   - Satisfaction labels for individual assistant turns.
   - Better for identifying local response failures and supporting fine-grained evaluator verification.
3. **Whether user simulation is used**
   - Some satisfaction or evaluation work uses simulated users or interaction rollouts.
   - User simulation can be useful, but it introduces drift and may not faithfully represent a real user's future satisfaction.

Ending contrast:

- Prior satisfaction estimation motivates the problem, but often does not model user-aware turn-level satisfaction from real user histories.
- Our work is explicitly **turn-level**, **user-aware**, and verified against human satisfaction annotations.

Possible literature groups:

- Conversation/session-level satisfaction estimation.
- Turn-level satisfaction estimation such as USS/DEUS/USDA-style work.
- User simulation for satisfaction or conversation evaluation, discussed as a related route but not the same as our replay benchmark.

### 2.2 Automatic Conversation Evaluation Method

This subsection should first acknowledge the benefits of automatic evaluation:

- Human evaluation is expensive and hard to repeat for every candidate model.
- Automatic evaluation methods enable scalable, repeatable evaluation.
- Non-LLM metrics and LLM-as-a-judge methods should be distinguished, because they represent different evaluation assumptions.

Then distinguish:

1. **Non-LLM automatic evaluation**
   - BLEU/ROUGE/BERTScore-style metrics.
   - These are scalable and reproducible, but often weak proxies for open-ended assistant response quality.
2. **LLM-as-a-judge automatic evaluation**
   - Methods such as G-Eval, MT-Bench/Chatbot Arena, and Prometheus.
   - LLM judges can evaluate rich natural-language responses beyond narrow task-completion metrics.
   - These typically evaluate generic quality or population-level preference.
3. **Personalized or user-aware evaluation**
   - A newer direction where the evaluator considers the user's preferences, history, or context.
   - This should lead into Section 2.3 rather than be fully covered here.

Ending contrast:

- We adopt LLM-as-a-judge because it supports scalable automatic evaluation, but we adapt it into a user-aware Conversation Satisfaction Evaluator rather than a generic quality judge.

### 2.3 User-Aware and Personalized Evaluation

This subsection replaces the old "Personalization and user modeling" subsection.
It should focus on evaluation rather than generation.

Use prior replay/user-simulation work where relevant, plus additional personalized evaluation work if available.
Possible angles:

- Personalized preference evaluation.
- User-conditioned reward modeling or judging.
- Evaluation with user profiles, interaction histories, or long-term memory.
- Replay/offline evaluation under fixed user contexts.

What to avoid:

- Do not spend much space on personalization work whose main goal is generation rather than evaluation.
- Mention such work only briefly in the introduction or as motivation.

Ending contrast:

- Existing personalized systems often focus on adapting model outputs.
- Our focus is user-aware evaluation: estimating whether a specific user would be satisfied with a specific assistant turn, and using the evaluator to benchmark generation models.

## 3. Conversation Satisfaction Evaluator

This should be the first main technical section after Related Work.
The full framework figure should be placed at the start of this section.

### Opening Message

At the start of this section, make clear:

- Once we have a verified Conversation Satisfaction Evaluator, it can evaluate arbitrary generation models.
- The evaluator is the bridge between human satisfaction annotations and UATSBench.
- The benchmark later reuses the evaluator as a frozen judge.

### Recommended Subsections

#### 3.1 Problem Formulation and Notation

Merge the old Task Formulation content here.
Do not keep "Task Formulation" as a separate section.

Define:

- user `u`
- target scenario/task `t`
- source user history `H_{u,\neg t}`
- user profile `p_u`
- task/conversation context `c_{u,t}`
- target conversation prefix `P_{u,t,i}`
- current user request `x_{u,t,i}`
- assistant response `a_{u,t,i}`
- human satisfaction score `y_{u,t,i} in {1,2,3,4,5}`
- dissatisfaction reason `r_{u,t,i}` when applicable
- SAT/DSAT boundary at 3/4

Use this formulation to introduce the evaluator:

```text
Conversation Satisfaction Evaluator:
E(P, x, a, p, c, H) -> score, reason
```

Recommended name:

- General component: **Conversation Satisfaction Evaluator**
- If a concrete method name is needed: **User-Aware Memory Evaluator (UAME)** or **Memory-Augmented Conversation Satisfaction Evaluator (MACSE)**.
- Keep UATSBench as the benchmark name, not the evaluator name.

#### 3.2 User Memory Construction

Describe the V2 memory construction:

- Input: source-scenario histories with satisfaction scores and reasons.
- Output: structured user memory.
- Fields: score distribution, average score, scoring style, 3/4 boundary, 4/5 boundary, user-specific requirements, response format preferences, task observations.
- Emphasize comparative memory: adjacent score-level comparisons turn historical ratings into an executable scoring rubric.

#### 3.3 Turn-Level Evaluation

Describe:

- The evaluator receives memory, user profile, target context, conversation prefix, and assistant response.
- It predicts a 1--5 score, dissatisfaction reason, and rationale.
- Reason is meaningful only for scores 1--3.
- Scores 4--5 use the default satisfied label.
- No target scenario gold labels are used.

#### 3.4 Score Calibration

Describe:

- Raw LLM scores are semantically useful but miscalibrated.
- Mean shift aligns block mean with user historical mean.
- CDF calibration maps within-block ranks to user historical score distribution.
- Raw and calibrated outputs should be reported separately.

### Content To Move Out

- Detailed baseline setup should move to Section 4 or Appendix.
- Detailed replay protocol should move to Section 5.
- Static replay notation should not be in this section.

## 4. Personalized Satisfaction Evaluator Verification

This is the meta-evaluation/meta-judge section.
It verifies whether the Conversation Satisfaction Evaluator agrees with human satisfaction annotations.

Recommended section title:

> Personalized Satisfaction Evaluator Verification

### 4.1 Setup

This subsection should include:

1. **Data**
   - Main data: RecLLMSim released conversation data and turn-level satisfaction annotations.
   - Auxiliary data: URS, with session-level satisfaction; used only as supplementary validation.
   - Clarify no new data collection in this paper.
2. **Baselines**
   - Briefly mention baseline categories in the main text:
     - supervised BERT / ordinal BERT
     - retrieval/RAG baselines
     - generic non-personalized LLM-as-a-judge
     - SPUR-style baseline
     - our user-aware evaluator
   - Put detailed baseline prompts/settings in Appendix.
3. **Metrics**
   - Main text: Pearson, Spearman, QWK, F1-DSAT.
   - Appendix: MAE/RMSE, false-SAT/false-DSAT, user-aware centered metrics, full formulas.

### 4.2 Evaluator Performance

This subsection reports the main meta-evaluation result.

Main table:

- Compare user-aware evaluator with baselines.
- Use current Table 1 style.
- Emphasize:
  - generic LLM judges underperform user-aware evaluator
  - raw retrieval can be competitive on rank metrics but weak on DSAT
  - supervised BERT is weak due to data scale and user-specificity

Main argument:

- User-aware evaluator better matches human turn-level satisfaction than generic automatic judges.

### 4.3 Additional Analysis

Include concise analysis in main text.
Move detailed table to Appendix.

Suggested paragraphs:

1. **Memory ablation**
   - No-memory vs V2 memory.
   - Main point: memory improves both ordinal agreement and dissatisfaction detection.
2. **Calibration analysis**
   - Mean shift vs CDF.
   - Main point: calibration corrects user-specific score scale.
3. **Backbone analysis**
   - Qwen3-8B full split vs Qwen3.6/GPT pilot subsets.
   - Main point: stronger backbones are promising but not final model rankings.

Appendix:

- Full ablation/backbone table.
- Baseline implementation details.
- URS supplementary results.

## 5. UATSBench: User-Aware Turn-Level Conversation Satisfaction Benchmark

Recommended section title:

> UATSBench: User-Aware Turn-Level Conversation Satisfaction Benchmark

This section should not be framed as "Static Replay Evaluation."
Replay is the protocol used inside UATSBench.

### Opening Message

After verifying the evaluator, we use it to evaluate arbitrary generation models.
UATSBench fixes user-aware conversation states and uses the Conversation Satisfaction Evaluator as a frozen scorer.

### 5.1 Protocol

Describe the benchmark protocol:

1. Select replay states from held-out conversation turns.
2. Each state contains:
   - user
   - task/scenario
   - conversation prefix
   - current user request
   - task context
   - source-scenario user history available to the evaluator
3. Candidate model generates an assistant response.
4. Frozen Conversation Satisfaction Evaluator scores the generated response.
5. Aggregate scores across turns/users/tasks.

Mention context modes:

- Prefix-only generation: model sees only the current conversation prefix.
- User-memory generation: model also receives retrieved user conversation memories.
- These modes can compare generic generation models and personalized memory-augmented systems under the same evaluator.

Mention calibration:

- Benchmark reports reference-CDF calibrated scores to reduce raw score inflation.
- Raw results may be reported in Appendix if needed.

### 5.2 Results

Main table:

- Absolute results for direct LLM evaluation.
- Current replay/refCDF table can become the UATSBench main result table.
- Rename caption to avoid "Static Replay Benchmark."

Recommended caption:

> UATSBench results for prefix-only candidate LLM responses with reference-CDF calibrated evaluator scores.

Main text:

- Report model ranking and score separation.
- Discuss DSAT rate as a sign of predicted dissatisfaction frequency.
- Avoid overclaiming exact ranking because evaluator is automatic.

Appendix references:

- Pairwise relative comparisons versus original assistant turns and GPT-5.5.
- Memory-augmented generation systems:
  - compare different generation models with user memory
  - compare same model with vs without user memory
  - report whether memory improves predicted user-aware satisfaction

## Appendix Plan

### Appendix A: Metric Definitions

Contains:

- MAE/RMSE
- Pearson/Spearman/QWK
- false-SAT/false-DSAT
- user-aware centered metrics

### Appendix B: Baseline and Evaluator Details

Contains:

- Baseline prompt templates and settings.
- SPUR-style baseline setup.
- BERT/ordinal BERT setup.
- RAG/retrieval baseline setup.
- Generic LLM-as-a-judge prompt setup.

### Appendix C: Evaluator Ablations

Contains:

- No memory vs memory.
- Mean shift vs CDF.
- Backbone pilot table.
- Additional prompt variants if still relevant.

### Appendix D: UATSBench Additional Results

Contains:

- Pairwise comparisons against original assistant turns.
- Pairwise comparisons against GPT-5.5.
- Memory-augmented generation system results.
- Prefix-only vs user-memory generation comparison.

## Terminology Replacement Checklist

Global replacements to apply during manuscript revision:

- "dialogue" -> "conversation"
- "assistant dialogues" -> "assistant conversations"
- "satisfaction predictor" -> "Conversation Satisfaction Evaluator"
- "predictor" -> "evaluator" when referring to the final evaluation model
- "Static Replay Evaluation" -> "UATSBench" or "replay protocol"
- "Static Replay Benchmark" -> "UATSBench"
- "static replay" -> "replay" except where contrasting with online simulation is necessary
- "personalized satisfaction prediction" -> "user-aware turn-level conversation satisfaction evaluation" when discussing paper positioning

## Current Manuscript Sections To Rewrite

### Abstract

Needs update to:

- emphasize user-aware turn-level conversation satisfaction
- name Conversation Satisfaction Evaluator
- name UATSBench
- remove "Static Replay Evaluation"

### Introduction

Needs update to:

- use Conversation terminology
- mention personalization only as broad motivation
- introduce UATSBench earlier
- avoid dataset contribution
- contributions should be evaluator construction, evaluator verification, benchmark

### Related Work

Needs restructuring:

1. Conversation satisfaction estimation
2. Automatic conversation evaluation method
3. User-aware personalized evaluation

Remove the old standalone personalization/user modeling subsection unless it is refocused on evaluation.

### Current Task/Data Sections

Task formulation should be merged into the evaluator section.
Data should move to evaluator verification setup.

### Current Predictor Section

Rename and rewrite as Conversation Satisfaction Evaluator.
Use evaluator terminology throughout.

### Current Experiments Section

Rename to Personalized Satisfaction Evaluator Verification.
Use:

- Setup
- Evaluator Performance
- Additional Analysis

### Current Replay Section

Rename to UATSBench.
Use:

- Protocol
- Results

Do not emphasize "Static."
