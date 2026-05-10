# ARR Paper Outline and LaTeX Draft

## Purpose

This document proposes a paper structure for an ARR-cycle submission based on
the current RecLLMSim work. The intended paper scope is:

1. a turn-level satisfaction dataset for task-oriented assistant dialogues
2. a personalized satisfaction predictor agent
3. a Static Replay Evaluation benchmark that uses the predictor to evaluate
   candidate LLM responses under fixed user histories

The outline is written to support direct migration into a LaTeX project. The
LaTeX source block near the end can be copied into an ACL/ARR-style manuscript
and expanded.

## Recommended Paper Positioning

### Working title

Recommended:

> User-Specific Satisfaction Prediction and Static Replay Evaluation for
> Task-Oriented Assistant Dialogues

Alternative titles:

- Personalized Turn-Level Satisfaction Modeling for Task-Oriented LLM Assistants
- From User Histories to Static Replay: Personalized Satisfaction Evaluation for
  Assistant Dialogues
- RecLLMSim: A User-Specific Satisfaction Dataset and Replay Benchmark for LLM
  Assistants

### Core claim

The paper should not be framed only as "we built a judge". A stronger framing is:

> Current LLM evaluation often measures generic response quality, but real
> satisfaction is user-specific and history-dependent. We introduce a turn-level
> personalized satisfaction setting, build a training-free predictor agent that
> uses user history to estimate satisfaction, and use it to construct a Static
> Replay Evaluation protocol for benchmarking candidate LLMs under fixed
> historical dialogue contexts.

This gives the paper three connected contributions:

1. **Dataset / task**: turn-level 1-5 satisfaction labels and dissatisfaction
   reasons in task-oriented assistant dialogues, with user histories enabling
   cross-task personalization.
2. **Predictor**: a memory-based personalized satisfaction agent with explicit
   user-history modeling, 1-5 scoring, 3/4 satisfaction boundary analysis, and
   post-hoc calibration.
3. **Benchmark**: Static Replay Evaluation, where candidate LLMs generate
   responses for historical prefixes and a frozen personalized predictor scores
   response satisfaction.

## Proposed Abstract

Draft:

> User satisfaction in task-oriented assistant dialogues is inherently
> personalized: the same response may be acceptable for one user but
> insufficient for another depending on preferences, strictness, and prior
> interaction history. Existing automatic evaluation methods mostly estimate
> generic response quality and rarely model user-specific satisfaction at the
> turn level. We introduce a personalized turn-level satisfaction prediction
> setting for task-oriented assistant dialogues, where each target dialogue is
> evaluated using the same user's labeled histories from other tasks. We build a
> memory-based satisfaction predictor agent that summarizes user preferences,
> predicts 1-5 satisfaction scores and dissatisfaction reasons, and supports
> post-hoc calibration to align predictions with user-specific scoring scales.
> Experiments show that simple user-history statistics are strong full-score
> baselines, while stronger LLM judges and calibration substantially improve
> ordinal ranking and dissatisfaction boundary detection. Finally, we propose
> Static Replay Evaluation: candidate LLMs generate responses for fixed
> historical dialogue prefixes, and a frozen personalized satisfaction predictor
> scores the resulting responses. This turns user-specific satisfaction
> prediction into a practical benchmark for comparing assistant models under
> controlled user histories.

## Paper Structure

### 1. Introduction

Goal: motivate personalized satisfaction as a missing evaluation axis.

Suggested flow:

1. Modern assistant evaluation often uses generic LLM-as-judge or task success,
   but user satisfaction depends on individual preferences and expectations.
2. Turn-level satisfaction is important because failures are local: a dialogue
   can contain both helpful and unsatisfactory responses.
3. Personalization requires history. A predictor should know whether the user is
   strict, prefers detailed plans, values concise answers, dislikes generic
   advice, etc.
4. This paper contributes a dataset/task, a predictor agent, and a replay-based
   benchmark.

Key message:

- The predictor is not only an offline analysis model; it enables a new
  benchmark protocol where candidate LLMs are evaluated for predicted
  user-specific satisfaction.

Suggested contributions paragraph:

- We formulate cross-task personalized turn-level satisfaction prediction.
- We introduce a dataset/protocol with 1-5 satisfaction labels and
  dissatisfaction reasons.
- We build and analyze a memory-based predictor agent with calibration and
  boundary metrics.
- We introduce Static Replay Evaluation for model benchmarking under fixed user
  histories.

### 2. Related Work

Recommended subsections:

1. **User satisfaction estimation in dialogue**
   - Task-oriented satisfaction modeling, 5-level satisfaction prediction,
     USS-style datasets, BERT/GRU supervised predictors.
   - Contrast: this work focuses on personalized cross-task user histories and
     turn-level assistant response evaluation.
2. **LLM-as-a-judge and automatic dialogue evaluation**
   - Generic judge methods, rubric-based evaluation, Prometheus-style judges,
     dialogue quality metrics.
   - Contrast: generic quality is not equivalent to user-specific satisfaction.
3. **Personalization and user modeling**
   - User profiles, preference modeling, memory agents, personalized assistants.
   - Contrast: we evaluate whether historical satisfaction labels can improve
     satisfaction prediction.
4. **Counterfactual/static replay evaluation**
   - Offline evaluation under fixed contexts, replay evaluation in recommenders
     or dialogue.
   - Contrast: our replay target is generated assistant responses scored by a
     user-specific satisfaction predictor.

### 3. Task and Dataset

Recommended section title:

> Personalized Turn-Level Satisfaction Prediction

Define one example as:

- user `u`
- target task `t`
- source-task history sessions `H_{u,\neg t}`
- target dialogue prefix and assistant response at turn `i`
- gold satisfaction score `y_i in {1,2,3,4,5}`
- dissatisfaction reason `r_i`, valid only when `y_i <= 3`

Prediction target:

- full 1-5 score prediction
- satisfaction boundary prediction: dissatisfied if `score <= 3`, satisfied if
  `score >= 4`
- dissatisfaction reason prediction only for predicted/gold dissatisfied cases

Current experimental setting to report:

- test split: `90` users
- test records: `356` user-task blocks
- test turns: `6474`
- task types: recipe planning, gift preparation, travel planning, skill learning
  planning
- user history source: same user's other tasks

Important dataset/protocol details:

- Cross-task personalization prevents using target-task gold labels for the same
  block.
- User history allows estimating scoring style and preference requirements.
- The 3/4 boundary is semantically important because it separates satisfied and
  dissatisfied turns.

Recommended table:

- number of users
- number of tasks
- number of user-task blocks
- number of turns
- score distribution
- dissatisfaction reason distribution

### 4. Personalized Satisfaction Predictor Agent

Recommended section title:

> Memory-Based Personalized Satisfaction Prediction

System components:

1. **Memory construction**
   - Input: source-task history sessions with gold satisfaction labels and
     dissatisfaction reasons.
   - Output: user memory containing scoring style, score distribution,
     preference requirements, positive/negative evidence, boundary distinctions.
2. **Turn-level judge**
   - Input: user memory, current user request, assistant response, task context.
   - Output: structured prediction with `classification` / `pred_score`,
     `reason_prediction`, and short analysis.
3. **Reason validity constraint**
   - Dissatisfaction reason is meaningful only when score `<= 3`.
   - For score `>= 4`, reason defaults to satisfied / no dissatisfaction.
4. **Post-hoc calibration**
   - Mean shift: align predicted block mean with user history mean.
   - CDF: map within-block prediction ranks to the user's historical score CDF.
5. **Boundary analysis**
   - Evaluate 3/4 boundary separately with F1-DSAT, false SAT, kappa, AUC, and
     user-aware binary metrics.

Recommended system figure:

```text
Source-task user histories
          |
          v
   Memory builder
          |
          v
 User-specific memory  ----->  Turn-level judge  -----> raw 1-5 score
                                      |                     |
                                      v                     v
                         reason prediction          post-hoc calibration
                                      |                     |
                                      v                     v
                           boundary metrics        calibrated 1-5 score
```

### 5. Experimental Setup

Recommended subsections:

1. **Baselines**
   - **Distributional baselines**: global mean/majority and task
     mean/majority estimated from train users.
   - **User-history baselines**: source-task user-history mean, median,
     majority, and empirical CDF/hash assignment. These use the same user's
     labeled histories but do not inspect the target turn semantics.
   - **Retrieval baselines**: nearest historical turn and top-$k$ nearest
     historical turns, using source-task labeled turns as non-parametric
     personalized evidence.
   - **Generic judge baselines**: zero-shot and few-shot LLM-as-a-judge
     variants that see the target context and assistant response but do not
     receive user profile, source-task history, or memory.
   - **External/potential baselines**: personalized SPUR-style rubric
     induction and supervised BERT ordinal prediction. These are aligned to the
     personalized split; if full results are not ready, mark them as pending in
     paper tables.
   - **Our predictor**: Qwen3-8B memory-v2 predictor as the full-split default,
     with stronger-backbone subset analyses for Qwen3.6-35B-A3B and
     gpt-5.4-mini.
2. **Predictor variants**
   - no-memory judge vs memory-v2 personalized judge
   - memory update variants v2.1-v2.5
   - boundary-specific prompt families and selective-refute variants
   - history-prior-delta and episodic/RAG ablations on fixed subsets
   - post-hoc calibration: mean shift and CDF/rank mapping
   - backbone comparison: Qwen3-8B, Qwen3.6-35B-A3B, gpt-4o-mini,
     gpt-5.4-mini
3. **Metrics**
   - Full score: MAE, RMSE, Pearson, Spearman, and Quadratic Weighted Kappa.
   - Boundary: accuracy, F1-SAT, F1-DSAT, boundary kappa, AUC, false-SAT rate,
     and false-DSAT rate under the 3/4 split.
   - User-aware: per-user aggregation and within-user centering.
   - Static replay: micro mean, user macro mean, task macro mean, user-task
     block macro mean, SAT/DSAT rate, and user-level bootstrap confidence
     intervals.
4. **Implementation**
   - OpenAI-compatible APIs and vLLM support
   - structured JSON output and parse failure logging
   - fixed frozen predictor for replay benchmark

### 6. Results: What Have We Learned?

This section should be organized by research question, not by chronological
experiment version.

#### RQ1: How strong are history-only baselines?

Current full test result:

- `user_history_median`: `MAE=0.5542`, `Spearman=0.3972`,
  `QWK=0.3075`
- `user_history_mean`: `MAE=0.5661`, `Pearson=0.3560`, `QWK=0.3105`

Interpretation:

- User strictness / leniency transfers strongly across tasks.
- A predictor must beat or complement user-history distribution baselines to
  show semantic value.

#### RQ2: Does an LLM judge add semantic signal?

Current full Qwen result:

- `Qwen none`: `MAE=0.7110`, `Pearson=0.2967`, `Spearman=0.2820`,
  `QWK=0.2815`
- `Qwen none + MS`: `MAE=0.6277`, `Pearson=0.3668`,
  `Spearman=0.3674`, `QWK=0.3589`
- `Qwen none + CDF`: `MAE=0.6355`, `Pearson=0.3601`,
  `Spearman=0.3716`, `QWK=0.3595`

Interpretation:

- Raw LLM predictions have useful but miscalibrated signal.
- Calibration reveals that relative ranking is stronger than raw absolute
  scoring.

#### RQ3: How important is the satisfaction boundary?

Current full Qwen boundary result:

- `Qwen none`: F1-DSAT around `0.3312`, false SAT around `0.6459`
- calibrated Qwen variants improve boundary behavior in previous reports
- boundary-only prompts improved DSAT sensitivity but often overcorrected or
  collapsed full 1-5 scoring

Interpretation:

- 3/4 is a separate decision problem and should be reported separately.
- Boundary-only prompts are useful analysis probes, but the final predictor
  should still output full 1-5 scores.

#### RQ4: Does stronger model capacity help?

Current 10-user subset result:

- `gpt-5.4-mini raw`: `MAE=0.6461`, `Pearson=0.3667`,
  `Spearman=0.3410`, `QWK=0.3340`, `F1-DSAT=0.4172`
- `Qwen3 raw` on same subset: `MAE=0.6727`, `Pearson=0.2710`,
  `QWK=0.2551`, `F1-DSAT=0.3376`
- `gpt-5.4-mini + MS`: `MAE=0.5761`
- `gpt-5.4-mini + CDF`: `Pearson=0.4129`, `Spearman=0.4238`,
  `QWK=0.4125`, `F1-DSAT=0.4362`

Interpretation:

- Model capacity matters, especially for semantic ranking and DSAT detection.
- The best predictor likely combines strong semantic judging with explicit
  user-history calibration.

#### RQ5: Are online memory updates useful?

Current observation:

- v2.1-v2.5 update variants did not consistently improve over frozen memory.
- Updates often behave like weak calibration shifts rather than robust boundary
  improvements.

Interpretation:

- The paper should probably present online memory update as an ablation, not as
  the main system claim.
- Frozen memory plus post-hoc calibration is more stable and easier to audit.

### 7. Static Replay Evaluation Benchmark

Recommended section title:

> Static Replay Evaluation with Personalized Satisfaction Prediction

Protocol:

1. Select a historical target dialogue and assistant turn.
2. Give candidate LLM the original dialogue prefix only.
3. Candidate LLM generates one response.
4. The generated response is not rolled into future turns.
5. A frozen personalized satisfaction predictor scores the candidate response.
6. Aggregate predicted satisfaction by candidate model.

Important design choice:

- Candidate model should not receive hidden user profile or task context in the
  main benchmark. It should only receive the raw dialogue prefix, matching the
  likely deployment condition.

Scoring variants:

- raw judge score
- calibrated judge score
- optional multi-judge ensemble in future work

Aggregation metrics:

- micro mean
- user macro mean
- task macro mean
- user-task macro mean
- SAT rate / DSAT rate
- score distribution
- bootstrap confidence intervals

Recommended static replay figure:

```text
Original historical dialogue prefix
          |
          v
  Candidate LLM response
          |
          v
Frozen personalized satisfaction predictor
          |
          v
Predicted user-specific satisfaction score
          |
          v
Model-level replay benchmark aggregation
```

### 8. Discussion

Recommended discussion points:

1. **User history is not a nuisance variable**
   - History-only baselines are strong; personalization is structurally central.
2. **Semantic judging and calibration are different capabilities**
   - LLMs provide semantic ordering; calibration maps it to user score scales.
3. **3/4 boundary matters**
   - Global accuracy can be misleading because most turns are satisfied.
4. **Static replay is useful but predictor-dependent**
   - Benchmark scores should report judge configuration and calibration variant.
5. **Frozen judge vs adaptive judge**
   - Frozen judge is fairer for comparing candidate models.
   - Adaptive judge can be studied later but creates model-dependent judge state.

### 9. Limitations

Recommended limitations:

- Predictor scores are automatic estimates, not new human labels for candidate
  responses.
- Calibration can mechanically impose historical score distributions, so raw
  and calibrated benchmark tracks must be reported separately.
- Current dataset focuses on a limited set of planning-oriented tasks.
- Dissatisfaction reasons are only valid for scores `<= 3`, which limits reason
  supervision for satisfied turns.
- Static replay is single-turn and does not model downstream effects of
  replacing a response in the dialogue.
- Strong API-model results are currently subset-level and should be expanded if
  used as a main paper claim.

### 10. Ethics and Reproducibility

Recommended points:

- User-specific satisfaction modeling can encode personal preferences and
  should be handled as sensitive user modeling.
- The benchmark should not be used as a sole quality measure for deployment.
- Report model, prompt version, memory version, calibration method, parse
  failure rate, and evaluation split.
- Release scripts for data loading, prediction, calibration, static replay
  collection, replay scoring, and aggregation.

## Suggested Main Tables and Figures

### Tables

1. Dataset statistics
2. History-only baselines
3. LLM predictor and calibration comparison
4. Boundary metrics comparison
5. User-aware metrics comparison
6. Static replay benchmark results
7. Ablation over memory update / calibration / model size

### Figures

1. Dataset/task formulation
2. Memory-based predictor agent
3. Calibration methods: raw prediction ranking to user score distribution
4. Static Replay Evaluation pipeline

## Recommended Paper Narrative

The safest and most coherent narrative is:

1. **Dataset/task**: We define personalized turn-level satisfaction prediction.
2. **Finding 1**: User history distribution is extremely strong, proving that
   satisfaction is personalized.
3. **Finding 2**: Raw LLM judges provide semantic signal but are poorly
   calibrated.
4. **Finding 3**: Post-hoc calibration and stronger judge models improve
   ranking/boundary performance.
5. **Application**: With a frozen calibrated predictor, we can evaluate
   candidate LLMs via Static Replay.

Avoid claiming:

- Online memory update is the key improvement. Current evidence does not support
  this strongly.
- Boundary-only prompts are the final solution. They are useful but incomplete
  because the final predictor should output 1-5 scores.
- Static replay produces human-equivalent labels. It produces predictor-based
  benchmark estimates.

## Copyable LaTeX Draft

```latex
\documentclass[11pt]{article}

% Replace with ARR/ACL style in the actual submission:
% \usepackage[review]{acl}

\usepackage{times}
\usepackage{latexsym}
\usepackage{amsmath}
\usepackage{booktabs}
\usepackage{multirow}
\usepackage{graphicx}
\usepackage{url}

\title{User-Specific Satisfaction Prediction and Static Replay Evaluation for Task-Oriented Assistant Dialogues}

\author{
  Anonymous Authors \\
  Anonymous Affiliation \\
  \texttt{anonymous@example.com}
}

\begin{document}
\maketitle

\begin{abstract}
User satisfaction in task-oriented assistant dialogues is inherently personalized:
the same response may be acceptable for one user but insufficient for another
depending on preferences, strictness, and prior interaction history. Existing
automatic evaluation methods mostly estimate generic response quality and rarely
model user-specific satisfaction at the turn level. We introduce a personalized
turn-level satisfaction prediction setting for task-oriented assistant
dialogues, where each target dialogue is evaluated using the same user's labeled
histories from other tasks. We build a memory-based satisfaction predictor agent
that summarizes user preferences, predicts 1--5 satisfaction scores and
dissatisfaction reasons, and supports post-hoc calibration to align predictions
with user-specific scoring scales. Experiments show that simple user-history
statistics are strong full-score baselines, while stronger LLM judges and
calibration substantially improve ordinal ranking and dissatisfaction boundary
detection. Finally, we propose Static Replay Evaluation: candidate LLMs generate
responses for fixed historical dialogue prefixes, and a frozen personalized
satisfaction predictor scores the resulting responses. This turns user-specific
satisfaction prediction into a practical benchmark for comparing assistant
models under controlled user histories.
\end{abstract}

\section{Introduction}

Evaluating assistant responses requires more than estimating generic response
quality. In task-oriented dialogues, user satisfaction depends on the user's
preferences, strictness, task expectations, and prior interaction history. A
concise answer may satisfy one user but disappoint another who expects detailed
plans, concrete recommendations, or explicit constraints. This makes
user-specific satisfaction a central evaluation target for personalized
assistants.

We study turn-level personalized satisfaction prediction. Given a user's labeled
histories from source tasks and a target assistant response, the model predicts
a 1--5 satisfaction score and, when the predicted or gold score indicates
dissatisfaction, a dissatisfaction reason. The setting differs from generic
LLM-as-a-judge evaluation because the predictor must account for user-specific
scoring style and preferences rather than applying a single global rubric.

This paper makes three contributions. First, we formulate a cross-task
personalized turn-level satisfaction prediction setting for task-oriented
assistant dialogues. Second, we build a memory-based satisfaction predictor
agent that summarizes user histories, predicts turn-level scores and reasons,
and supports post-hoc calibration to align predictions with user-specific score
distributions. Third, we introduce Static Replay Evaluation, a benchmark
protocol where candidate LLMs generate responses for fixed historical dialogue
prefixes and a frozen personalized predictor scores their user-specific
satisfaction.

\section{Related Work}

\paragraph{Dialogue satisfaction estimation.}
Prior work on user satisfaction estimation studies automatic prediction of
dialogue-level or turn-level satisfaction, often in task-oriented dialogue
settings. These methods typically rely on supervised models, task features, or
generic dialogue representations. Our setting differs by explicitly requiring
cross-task personalization from the same user's labeled histories.

\paragraph{LLM-as-a-judge and dialogue evaluation.}
LLM-based evaluators and rubric-based judges have become common tools for
automatic assessment of response quality. However, generic quality does not
fully capture user-specific satisfaction. We evaluate whether LLM judges can
serve as personalized satisfaction predictors when combined with user memory and
calibration.

\paragraph{Personalization and user modeling.}
Personalized assistants increasingly use profiles, memories, or preference
summaries. We focus on a complementary question: whether historical
satisfaction labels can be used to predict how satisfied the same user will be
with future assistant responses.

\paragraph{Static replay evaluation.}
Offline replay protocols evaluate models under fixed contexts. We adapt this
idea to assistant evaluation: candidate models generate a single response for a
historical dialogue prefix, and a frozen personalized satisfaction predictor
scores the response.

\section{Task Formulation}

Let $u$ denote a user and $t$ a target task. For each target dialogue turn $i$,
we observe a user request $x_i$, an assistant response $a_i$, source-task user
histories $H_{u,\neg t}$, a satisfaction score $y_i \in \{1,2,3,4,5\}$, and a
dissatisfaction reason $r_i$. We define turns with $y_i \leq 3$ as dissatisfied
and turns with $y_i \geq 4$ as satisfied. Dissatisfaction reasons are only valid
for dissatisfied turns; satisfied turns use a default satisfied label.

The predictor estimates
\begin{equation}
  \hat{y}_i = f(x_i, a_i, H_{u,\neg t}, c_t),
\end{equation}
where $c_t$ is optional task context. We evaluate both full 1--5 score
prediction and the binary satisfaction boundary induced by the 3/4 split.

\section{Dataset and Protocol}

Our experimental protocol uses task-oriented assistant dialogues with
turn-level satisfaction labels. Each example contains a user, a target task, a
target dialogue turn, and the same user's labeled histories from other tasks.
The test split used in our current experiments contains 90 users, 356 user-task
blocks, and 6,474 assistant turns across four planning-oriented task families:
recipe planning, gift preparation, travel planning, and skill learning
planning.

\begin{table}[t]
\centering
\small
\begin{tabular}{lr}
\toprule
Statistic & Value \\
\midrule
Test users & 90 \\
Test user-task blocks & 356 \\
Test assistant turns & 6,474 \\
Task families & 4 \\
Score scale & 1--5 \\
Satisfaction boundary & 3/4 \\
\bottomrule
\end{tabular}
\caption{Dataset statistics for the current personalized satisfaction test
split. Final paper numbers should be updated after freezing the dataset split.}
\label{tab:data-stats}
\end{table}

\section{Memory-Based Personalized Satisfaction Predictor}

Our predictor has three components. First, a memory builder reads the user's
source-task histories and constructs a user memory containing scoring style,
score distribution, preference requirements, positive and negative examples,
and boundary distinctions. Second, a turn-level judge predicts a structured
1--5 satisfaction score and a dissatisfaction reason. Third, optional post-hoc
calibration aligns raw predictions with the user's historical scoring scale.

\paragraph{Memory construction.}
Given labeled histories $H_{u,\neg t}$, the memory builder summarizes what the
user tends to reward or penalize. It also records the empirical score
distribution and average satisfaction score, which are later used for
calibration.

\paragraph{Turn-level judging.}
The judge receives the user memory, current user request, assistant response,
and task context, then outputs a structured score prediction. We enforce the
constraint that dissatisfaction reasons are meaningful only for scores
$\leq 3$.

\paragraph{Post-hoc calibration.}
We consider two calibration methods. Mean-shift calibration adjusts all
predictions in a block so that the predicted mean matches the user's historical
mean. CDF calibration ranks predictions within a block and maps their ranks to
the user's historical score distribution.

\section{Experimental Setup}

\paragraph{Baselines.}
We compare against four classes of baselines. First, distributional baselines
estimate scores from train users only, including global mean/majority and
task-specific mean/majority predictors. Second, user-history baselines use the
same user's labeled source-task histories without inspecting the target
response semantics; these include user-history mean, median, majority, and an
empirical CDF/hash assignment baseline. Third, retrieval baselines use the
nearest labeled source-task turns as non-parametric personalized evidence,
including top-1 and top-$k$ nearest historical turn variants. Fourth, generic
LLM-as-a-judge baselines score the target assistant response from the target
context only, under zero-shot or few-shot global demonstrations, without user
profile, source-task history, or memory. We additionally include aligned
implementations of a SPUR-style rubric induction baseline and a supervised BERT
ordinal baseline as potential external comparisons; these are reported when
full-split results are available.

Our main predictor is a training-free memory-based judge. Unless otherwise
specified, the full-split experiments use Qwen3-8B with memory version v2, no
target-task memory update, and the v2 turn-evaluation prompt. We ablate the
effect of removing memory, changing memory/update variants, adding
boundary-specific prompts, using history-prior-delta or episodic retrieval
modules, and applying post-hoc calibration. For model-capacity analysis, we
also run fixed-subset comparisons with Qwen3.6-35B-A3B and gpt-5.4-mini.

\paragraph{Post-hoc calibration.}
Let $B=(u,t)$ denote a user--target-task block, and let
$\hat{y}_i \in \{1,\ldots,5\}$ be the raw prediction for turn $i\in B$. Mean
shift uses the user's historical mean $\mu^{\mathrm{hist}}_{u,\neg t}$ and the
block prediction mean $\bar{\hat{y}}_B$:
\begin{equation}
  \tilde{y}_i =
  \operatorname{clip}_{1,5}\!\left(
  \operatorname{round}\left(\hat{y}_i +
  \mu^{\mathrm{hist}}_{u,\neg t} - \bar{\hat{y}}_B\right)\right).
\end{equation}
CDF calibration ranks predictions within the block and maps each rank to the
inverse empirical CDF of the user's source-task history:
\begin{equation}
  \tilde{y}_i =
  F^{-1}_{u,\neg t}\!\left(\frac{\operatorname{rank}_B(\hat{y}_i)+0.5}{|B|}\right).
\end{equation}
We report raw and calibrated variants separately because calibration changes
the score scale and can mechanically impose the user's historical distribution.

\paragraph{Metrics.}
For full 1--5 score prediction, we report absolute error, ranking, and ordinal
agreement metrics. Given gold scores $y_i$ and predictions $\hat{y}_i$ over
$N$ target turns, mean absolute error and root mean squared error are
\begin{equation}
  \mathrm{MAE} = \frac{1}{N}\sum_{i=1}^{N}|y_i-\hat{y}_i|,
  \qquad
  \mathrm{RMSE} = \sqrt{\frac{1}{N}\sum_{i=1}^{N}(y_i-\hat{y}_i)^2}.
\end{equation}
Pearson correlation measures linear score association, while Spearman
correlation applies Pearson correlation to the score ranks:
\begin{equation}
  r =
  \frac{\sum_i (y_i-\bar{y})(\hat{y}_i-\bar{\hat{y}})}
       {\sqrt{\sum_i (y_i-\bar{y})^2}\sqrt{\sum_i(\hat{y}_i-\bar{\hat{y}})^2}}.
\end{equation}
Quadratic weighted kappa (QWK) measures ordinal agreement while penalizing
larger score disagreements more heavily:
\begin{equation}
  \kappa_{\mathrm{QW}} =
  1 - \frac{\sum_{a,b} w_{ab} O_{ab}}{\sum_{a,b} w_{ab} E_{ab}},
  \qquad
  w_{ab}=\frac{(a-b)^2}{(K-1)^2},
\end{equation}
where $O$ is the observed confusion matrix, $E$ is the expected confusion matrix
under independent marginals, and $K=5$.

We also evaluate the satisfaction boundary induced by the 3/4 split:
$\mathrm{DSAT}$ if $y_i\leq 3$ and $\mathrm{SAT}$ if $y_i\geq 4$. We report
accuracy, class-specific F1 for SAT and DSAT, Cohen's kappa, AUC, false-SAT
rate, and false-DSAT rate. In particular,
\begin{equation}
  \mathrm{FalseSAT} =
  \frac{\#\{i: y_i\leq 3,\ \hat{y}_i\geq 4\}}
       {\#\{i: y_i\leq 3\}},
  \qquad
  \mathrm{FalseDSAT} =
  \frac{\#\{i: y_i\geq 4,\ \hat{y}_i\leq 3\}}
       {\#\{i: y_i\geq 4\}}.
\end{equation}
False-SAT is especially important because it corresponds to missing turns where
the user was actually dissatisfied.

To separate global score-scale effects from personalized discrimination, we
report user-aware variants. Per-user metrics compute the metric independently
for each user and average over users. Within-user centered metrics first remove
each user's mean score:
\begin{equation}
  y_i^{c}=y_i-\bar{y}_{u(i)}, \qquad
  \hat{y}_i^{c}=\hat{y}_i-\bar{\hat{y}}_{u(i)},
\end{equation}
and then compute correlation on the centered values.

\paragraph{Static replay metrics.}
For Static Replay Evaluation, candidate models generate responses for the same
fixed historical prefixes and a frozen personalized predictor assigns
satisfaction scores. We report micro mean satisfaction
$\frac{1}{N}\sum_i \hat{y}_i$, user macro mean
$\frac{1}{|U|}\sum_{u\in U}\frac{1}{N_u}\sum_{i:u(i)=u}\hat{y}_i$, task macro
mean, and user-task block macro mean. We also report SAT rate, DSAT rate, and
user-level bootstrap confidence intervals. The replay benchmark uses a frozen
judge state for all candidate models; generated responses are not rolled into
future dialogue turns.

\section{Results}

\subsection{History-only baselines are strong}

Simple user-history statistics are strong full-score baselines, indicating that
user strictness and scoring style transfer across tasks. On the full test split,
user-history median reaches MAE 0.5542, Spearman 0.3972, and QWK 0.3075, while
user-history mean reaches MAE 0.5661, Pearson 0.3560, and QWK 0.3105.

\begin{table}[t]
\centering
\small
\begin{tabular}{lrrrrr}
\toprule
Method & MAE & RMSE & Pearson & Spearman & QWK \\
\midrule
Global mean & 0.6985 & 0.9434 & -- & -- & 0.0000 \\
User-history mean & 0.5661 & 0.8870 & 0.3560 & 0.3917 & 0.3105 \\
User-history median & \textbf{0.5542} & 0.9150 & 0.3526 & \textbf{0.3972} & 0.3075 \\
Nearest history turn & 0.7079 & 1.1064 & 0.2550 & 0.2891 & 0.2543 \\
Nearest history turn, $k=3$ & 0.6285 & 0.9798 & 0.3076 & 0.3406 & 0.2974 \\
\bottomrule
\end{tabular}
\caption{History-only and retrieval baselines on the full personalized
satisfaction test split.}
\label{tab:history-baselines}
\end{table}

\subsection{LLM judges provide useful but miscalibrated signal}

Raw LLM predictions are less accurate than user-history mean for full-score
MAE, but calibration substantially improves their ranking and agreement with
gold scores. For Qwen3-8B, mean shift and CDF calibration improve the raw
no-memory judge from MAE 0.7110 and QWK 0.2815 to around MAE 0.63 and QWK 0.36.

\begin{table}[t]
\centering
\small
\begin{tabular}{lrrrrr}
\toprule
Method & MAE & RMSE & Pearson & Spearman & QWK \\
\midrule
Qwen3 raw & 0.7110 & -- & 0.2967 & 0.2820 & 0.2815 \\
Qwen3 + mean shift & \textbf{0.6277} & -- & \textbf{0.3668} & 0.3674 & 0.3589 \\
Qwen3 + CDF & 0.6355 & 1.0191 & 0.3601 & \textbf{0.3716} & \textbf{0.3595} \\
\bottomrule
\end{tabular}
\caption{Effect of post-hoc calibration for Qwen3-8B no-memory predictions.
Final paper should include RMSE for all rows from the frozen evaluation file.}
\label{tab:qwen-calibration}
\end{table}

\subsection{Model capacity improves semantic ranking and boundary detection}

On a 10-user subset, gpt-5.4-mini improves over Qwen3-8B under the same V2
no-memory prompt. With post-hoc calibration, gpt-5.4-mini becomes the strongest
LLM judge variant in this subset.

\begin{table}[t]
\centering
\small
\begin{tabular}{lrrrrr}
\toprule
Method & MAE & Pearson & Spearman & QWK & F1-DSAT \\
\midrule
Qwen3 raw & 0.6727 & 0.2710 & 0.2795 & 0.2551 & 0.3376 \\
Qwen3 + mean shift & 0.6051 & 0.3727 & 0.4039 & 0.3633 & 0.4231 \\
gpt-4o-mini raw & 0.6473 & 0.2697 & 0.2707 & 0.2574 & 0.2619 \\
gpt-4o-mini + CDF & 0.6413 & 0.3195 & 0.3679 & 0.3194 & 0.3599 \\
gpt-5.4-mini raw & 0.6461 & 0.3667 & 0.3410 & 0.3340 & 0.4172 \\
gpt-5.4-mini + mean shift & \textbf{0.5761} & 0.4002 & 0.4057 & 0.3874 & 0.4238 \\
gpt-5.4-mini + CDF & 0.6014 & \textbf{0.4129} & \textbf{0.4238} & \textbf{0.4125} & \textbf{0.4362} \\
\bottomrule
\end{tabular}
\caption{Subset comparison of model capacity and calibration. These numbers
are from a 10-user subset and should be reported as pilot results unless
expanded to the full split.}
\label{tab:model-capacity}
\end{table}

\section{Static Replay Evaluation}

We propose Static Replay Evaluation to benchmark candidate LLMs under fixed
historical dialogue contexts. For each original assistant turn, we keep the
dialogue prefix before that turn and ask a candidate model to generate one
assistant response. The candidate model does not receive hidden user profiles in
the main benchmark condition. We then score the generated response using a
frozen personalized satisfaction predictor.

\begin{figure}[t]
\centering
\fbox{\parbox{0.9\linewidth}{
\centering
Historical dialogue prefix $\rightarrow$ Candidate LLM response
$\rightarrow$ Frozen personalized satisfaction predictor
$\rightarrow$ Model-level replay score
}}
\caption{Static Replay Evaluation protocol. Candidate responses are generated
from fixed historical prefixes and scored by a frozen user-specific satisfaction
predictor.}
\label{fig:static-replay}
\end{figure}

We aggregate predicted satisfaction scores using micro average, user macro
average, task macro average, user-task macro average, SAT/DSAT rate, score
distribution, and bootstrap confidence intervals. Raw and calibrated judge
scores are reported as separate benchmark tracks.

\section{Discussion}

Our results suggest that personalized satisfaction prediction requires both
user-history modeling and semantic response understanding. History-only
statistics capture user strictness and score scale, while LLM judges capture
turn-level semantic quality and dissatisfaction evidence. Calibration is a
simple but effective bridge between the two: it maps LLM-derived rankings onto
the user's historical scoring scale.

The 3/4 boundary should be treated as a first-class evaluation target. Overall
accuracy can be high even when dissatisfied turns are missed, because satisfied
turns dominate the dataset. We therefore report F1-DSAT, false-SAT rate,
boundary kappa, and user-aware binary metrics.

\section{Limitations}

The predictor provides automatic satisfaction estimates, not new human labels
for replayed candidate responses. Calibration can mechanically impose a user's
historical score distribution, so raw and calibrated benchmark tracks should be
reported separately. The current task families are planning-oriented and may
not cover all assistant use cases. Static replay is single-turn and does not
simulate downstream dialogue effects after replacing a response. Finally,
subset-level results with stronger API models should be expanded before being
used as primary claims.

\section{Conclusion}

We introduced a personalized turn-level satisfaction prediction setting for
task-oriented assistant dialogues, built a memory-based predictor agent, and
proposed Static Replay Evaluation for benchmarking candidate LLMs under fixed
user histories. Our experiments show that user-history statistics are strong,
raw LLM judges are semantically useful but miscalibrated, and calibration
substantially improves ranking and satisfaction-boundary metrics. These results
support user-specific satisfaction prediction as a practical foundation for
evaluating personalized assistant behavior.

\end{document}
```

## Immediate TODOs Before Submission

1. Freeze dataset statistics and update `Table 1`.
2. Decide whether strong API-model results will be full-run or subset-only.
3. Add static replay results for at least two candidate LLMs.
4. Report parse failure rates for all predictor variants.
5. Decide main predictor configuration:
   - full-score track: likely `gpt-5.4-mini + mean_shift` if full run is
     affordable, otherwise `Qwen none + MS/CDF`
   - boundary/ranking track: likely CDF-calibrated variant
6. Add a short ablation table showing that online memory updates are not the
   main source of gains.
7. Add examples: one successful personalized prediction, one calibration
   correction, one static replay comparison.
