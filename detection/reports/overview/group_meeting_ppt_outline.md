# Group Meeting PPT Outline

## Reference Style

Reference deck:

- `detection/assets/大组会分享RecLLMSim-20240521.pptx`

Observed structure:

- 19 slides
- section divider slides are used for major parts
- flow: title -> motivation -> related work -> dataset construction -> data
  characteristics -> case study -> potential usage -> conclusion
- slide density is moderate, with one main visual/table per content slide

This outline adapts that structure to the current personalized satisfaction
prediction and static replay work.

## Suggested Deck Title

`User-Specific Satisfaction Prediction and Static Replay Evaluation for RecLLMSim`

Chinese title option:

`RecLLMSim中的个性化满意度预测与Static Replay评价`

## Slide 1: Title

Title:

- `Personalized Satisfaction Prediction and Static Replay Evaluation`

Subtitle:

- `Progress Update on RecLLMSim`
- presenter name, group meeting date

Main message:

- From real user dialogues to user-specific satisfaction predictors, then to
  static replay evaluation of candidate LLMs.

Expected visual:

- Use the final four-part framework figure as a faded background or a small
  centered teaser.

## Slide 2: Motivation

Title:

- `Motivation: Satisfaction Is User-Specific`

Text:

- Generic response quality is not enough for task-oriented assistants.
- The same reply may satisfy one user but disappoint another because users have
  different strictness, preferences, and expectations.
- Turn-level satisfaction matters because failures are local and can occur
  inside otherwise successful conversations.

Expected visual:

- Two-user contrast graphic:
  - same assistant response
  - User A gives 5, User B gives 3
  - short labels such as `prefers concise answer` vs `expects detailed plan`

## Slide 3: Research Questions

Title:

- `Research Questions`

Text:

1. Can historical satisfaction labels from the same user predict satisfaction in
   a new task?
2. Can an LLM judge add semantic signal beyond history-only baselines?
3. How should we model the 3/4 satisfaction boundary?
4. Can a frozen personalized predictor support static replay benchmarking?

Expected visual:

- Four numbered question cards, aligned with the four parts of the framework
  figure.

## Slide 4: Overall Framework

Title:

- `Overall Framework`

Text:

- Real user dialogue data collection
- Cross-task turn-level prediction task construction
- Memory-based personalized satisfaction predictor
- Static replay benchmark for candidate LLM evaluation

Expected visual:

- The main framework figure from
  `detection/reports/overview/arr_framework_figure_design.md`.
- Use this as the deck's anchor figure.

## Slide 5: Section Divider

Title:

- `Part I: Data and Task Formulation`

Expected visual:

- Minimal section slide, similar to the reference deck's `Dataset Construction`
  divider.

## Slide 6: Real User Dialogue Data

Title:

- `Data Collection Protocol`

Text:

- Four task families: travel planning, gift preparation, recipe planning, skill
  learning planning.
- For each user: scenario/task setup, user profile, multi-turn dialogue, and
  post-dialogue satisfaction annotation.
- Each assistant turn has a 1-5 satisfaction score.
- Dissatisfaction reasons are collected for low-score turns.

Expected visual:

- Left: mini pipeline
  `Task scenario -> User profile -> Dialogue -> Turn-level labels`.
- Right: small annotated dialogue example with score badges.

## Slide 7: Cross-Task Personalized Prediction

Title:

- `Task: Cross-Task Personalized Turn-Level Prediction`

Text:

- Given target task `t`, use the same user's histories from other tasks
  `H_{u, not t}`.
- Input: history dialogues with labels, user profile, current dialogue context,
  current assistant reply.
- Output: predicted satisfaction score `1-5`; optional dissatisfaction reason.
- Main binary boundary: `1-3 = dissatisfied`, `4-5 = satisfied`.

Expected visual:

- Diagram showing three source tasks feeding into one target-task prediction.
- Include the formula-like notation:
  `f(H_{u, not t}, profile_u, context_i, response_i) -> y_i`.

## Slide 8: Dataset Statistics

Title:

- `Evaluation Split`

Text:

- Test users: `90`
- User-task blocks: `356`
- Assistant turns: `6474`
- Cross-task setting prevents using target-task gold labels during prediction.

Expected visual:

- A compact stats table.
- Optional task distribution bar chart if available.
- Optional score distribution bar chart:
  `1:123, 2:251, 3:733, 4:2449, 5:2918`
  from full-data reports.

## Slide 9: Section Divider

Title:

- `Part II: Personalized Satisfaction Predictor`

Expected visual:

- Simple memory/predictor icon and section title.

## Slide 10: Predictor Design

Title:

- `Memory-Based Predictor Agent`

Text:

- Memory construction summarizes the user's historical scoring behavior.
- Turn-level judge predicts the current assistant response score.
- Post-hoc calibration maps raw model predictions to the user's historical
  score scale.

Expected visual:

- Three-stage model figure:
  `History with labels -> User Memory -> Turn-level Scoring -> Calibration`.
- Use fields inside memory:
  `score distribution`, `strictness`, `3/4 boundary`, `4/5 boundary`,
  `preference requirements`.

## Slide 11: Memory Construction

Title:

- `User Memory: What Is Extracted?`

Text:

- Score distribution and average historical satisfaction.
- Scoring style: strict vs lenient.
- Boundary distinctions:
  - what separates 3 from 4
  - what separates 4 from 5
- User-specific response requirements and task observations.

Expected visual:

- Memory schema card, preferably based on `UserMemoryContent`.
- Highlight `three_vs_four_distinction` and `four_vs_five_distinction`.

## Slide 12: Calibration

Title:

- `Calibration: Separating Ranking from Score Scale`

Text:

- Raw LLM judges often have useful relative ranking but poor absolute score
  calibration.
- Mean shift aligns block-level mean with user history.
- CDF/rank mapping maps predicted ranks to the user's historical score
  distribution.
- Raw and calibrated scores should be reported separately.

Expected visual:

- Before/after mini chart:
  - raw prediction distribution
  - calibrated prediction distribution
- Include a small equation-like caption:
  `raw semantic ranking -> user-specific score scale`.

## Slide 13: Section Divider

Title:

- `Part III: Experimental Results`

Expected visual:

- Minimal section slide with a chart icon.

## Slide 14: History-Only Baselines

Title:

- `Finding 1: User History Alone Is Strong`

Text:

- User strictness/leniency transfers strongly across tasks.
- History-only mean/median are strong full-score baselines.
- This proves personalization is not a minor nuisance variable.

Expected visual:

- Table:
  - `global_mean`: MAE `0.6985`, QWK `0.0000`
  - `user_history_mean`: MAE `0.5661`, Pearson `0.3560`, QWK `0.3105`
  - `user_history_median`: MAE `0.5542`, Spearman `0.3972`, QWK `0.3075`
  - `nearest_history_turn_k3`: MAE `0.6285`, QWK `0.2974`
- Highlight `user_history_median` and `user_history_mean`.

## Slide 15: LLM Predictor and Calibration

Title:

- `Finding 2: LLM Judges Need User-Aware Calibration`

Text:

- Qwen3-8B raw memory predictor improves over no-memory on correlation/QWK.
- Post-hoc calibration gives the best full-run Qwen results.
- The main signal: LLMs provide semantic ordering; calibration maps it to each
  user's score scale.

Expected visual:

- Table:
  - `Qwen no_memory`: MAE `0.7471`, Pearson `0.1868`, QWK `0.1096`
  - `Qwen none`: MAE `0.7110`, Pearson `0.2967`, QWK `0.2815`
  - `Qwen none + MS`: MAE `0.6277`, Pearson `0.3668`, QWK `0.3589`
  - `Qwen none + CDF`: MAE `0.6355`, Spearman `0.3716`, QWK `0.3595`

## Slide 16: Boundary Results

Title:

- `Finding 3: The 3/4 Boundary Needs Separate Evaluation`

Text:

- Accuracy is misleading because most turns are satisfied.
- `F1-DSAT`, false-SAT rate, and boundary kappa better reflect dissatisfaction
  detection.
- Boundary-specific prompts improve DSAT sensitivity but can overcorrect.

Expected visual:

- Table:
  - `Qwen none`: F1-DSAT `0.3312`, false SAT `0.6459`
  - `boundary_34_refute`: F1-DSAT `0.3418`, false SAT `0.5131`
  - `selective_refute_v2`: F1-DSAT `0.3177`, false SAT `0.6269`
  - `reasonfix + anchor2`: F1-DSAT `0.3197`, false SAT `0.6224`
- A small `<=3 DSAT | >=4 SAT` boundary graphic.

## Slide 17: Larger Backbone Validation

Title:

- `Finding 4: Larger Models Change the Error Shape`

Text:

- Qwen3.6-35B-A3B improves ordinal signal and DSAT detection on a 20-user
  subset.
- Raw Qwen3.6 is stricter and hurts MAE.
- Calibration is especially important for the larger model.

Expected visual:

- Table:
  - `Qwen3-8B raw`: MAE `0.7077`, Pearson `0.2999`, QWK `0.2807`,
    F1-DSAT `0.3505`
  - `Qwen3.6 raw`: MAE `0.7842`, Pearson `0.3411`, QWK `0.3083`,
    F1-DSAT `0.4168`
  - `Qwen3.6 + CDF`: MAE `0.6405`, Pearson `0.3977`, QWK `0.3966`,
    F1-DSAT `0.4121`
- Optional prediction distribution chart showing Qwen3.6 uses fewer 5s and more
  DSAT-side scores.

## Slide 18: Static Replay Evaluation

Title:

- `Application: Static Replay Benchmark`

Text:

- Candidate LLMs answer fixed historical dialogue prefixes.
- Generated responses are not rolled into later turns.
- A frozen personalized predictor scores every candidate response.
- Aggregate by micro mean, user macro, task macro, user-task macro, SAT/DSAT
  rate, and confidence interval.

Expected visual:

- Static replay pipeline:
  `historical prefix -> candidate LLM response -> frozen predictor -> benchmark metrics`.
- Show parallel candidate models `LLM A/B/C`.
- Mark `profile/history` as judge-side information, not candidate-visible
  context.

## Slide 19: Current Status

Title:

- `Current Implementation Status`

Text:

- Personalized prediction pipeline implemented with API/vLLM support.
- History baselines, calibration, boundary analysis, and user-aware metrics are
  implemented.
- Static replay collection, scoring, and aggregation scripts are implemented.
- Code has been refactored into modular data, memory, trace, and eval
  components.

Expected visual:

- Checklist table:
  - data/task construction: done
  - memory predictor: done
  - calibration: done
  - static replay pipeline: implemented
  - full replay benchmark results: next

## Slide 20: Case Study

Title:

- `Case Study: Why Personalization Matters`

Text:

- Show one user history pattern and one target assistant response.
- Compare generic judgment vs user-specific memory-based judgment.
- Highlight which user preference or strictness cue changes the predicted
  score.

Expected visual:

- Left: user memory snippets.
- Middle: dialogue turn.
- Right: prediction card with score and reason.

Candidate source:

- Use examples from output JSONL files under `detection/outputs/personalized/`.
- Prefer a case where `Qwen none + calibration` or Qwen3.6 changes the raw
  score in an interpretable way.

## Slide 21: Limitations

Title:

- `Limitations`

Text:

- Predictor scores are automatic estimates, not new human labels for candidate
  responses.
- Calibration can mechanically impose historical score distributions.
- Static replay is single-turn and does not model downstream conversation
  effects.
- Current experiments focus on planning-oriented task families.
- Stronger-model validation is currently subset-level.

Expected visual:

- Four caution cards:
  `judge noise`, `calibration bias`, `single-turn replay`, `task scope`.

## Slide 22: Next Steps

Title:

- `Next Steps`

Text:

1. Finalize the main framework figure for the paper and slides.
2. Run fixed-subset comparison across Qwen3-8B, Qwen3.6-35B-A3B, and API judge
   variants.
3. Produce first static replay leaderboard with raw and calibrated judge tracks.
4. Select representative case studies for the paper and group presentation.
5. Decide final ARR narrative: dataset/task, predictor, static replay benchmark.

Expected visual:

- Roadmap timeline with three milestones:
  `predictor validation -> replay benchmark -> paper packaging`.

## Slide 23: Conclusion

Title:

- `Conclusion`

Text:

- User history is a strong signal for satisfaction prediction.
- LLM judges provide semantic ranking but require user-aware calibration.
- 3/4 boundary detection should be evaluated separately from full 1-5 scoring.
- A frozen personalized predictor enables static replay evaluation of candidate
  LLMs.

Expected visual:

- Four takeaway boxes matching the four-part framework.

## Slide 24: Thanks

Title:

- `Thanks!`

Text:

- Optional contact information.

Expected visual:

- Clean closing slide, optionally with the framework figure watermark.

## Optional Backup Slides

### Backup A: Metric Definitions

- MAE / RMSE
- Pearson / Spearman
- Quadratic Weighted Kappa
- F1-DSAT / false SAT / false DSAT
- PU and WC user-aware metrics

### Backup B: Prompt and Schema Examples

- UserMemoryContent fields
- TurnPrediction fields
- Reason legality rule:
  - score `>=4`: reason should be satisfied / no dissatisfaction
  - score `<=3`: dissatisfaction reason is meaningful

### Backup C: Static Replay JSONL Schema

- candidate generation schema
- scored benchmark schema
- aggregation schema

### Backup D: Qwen3.6 Deployment Feasibility

- Host: 8x A100 80GB
- First suggested setup: Qwen3.6-35B-A3B, BF16, TP2, 32K context
- Qwen3.6-27B is a second-stage dense validation point

## Reusable Visual Assets

- Existing image:
  `detection/assets/satisfaction_training_pipeline_drawio_v1_20260325.png`
- Existing draw.io source:
  `detection/assets/satisfaction_training_pipeline_drawio_v1_20260325.drawio`
- Main framework design:
  `detection/reports/overview/arr_framework_figure_design.md`
- Recommended icon sources:
  - Lucide Icons
  - Tabler Icons
  - Heroicons

## Recommended Short Version

If the group meeting slot is short, use 14 slides:

1. Title
2. Motivation
3. Overall framework
4. Data collection
5. Cross-task prediction task
6. Predictor design
7. Calibration
8. History-only baselines
9. Qwen predictor and calibration results
10. Boundary results
11. Qwen3.6 subset validation
12. Static replay benchmark
13. Limitations and next steps
14. Conclusion
