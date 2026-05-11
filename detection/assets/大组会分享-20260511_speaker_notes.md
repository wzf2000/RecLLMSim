# Speaker Notes for `大组会分享-20260511.pptx`

## Slide 1: User-Specific Satisfaction Prediction and Static Replay Evaluation

Good morning / afternoon. Today I will introduce my recent work on user-specific satisfaction prediction and static replay evaluation for RecLLMSim.

The main motivation is that, in task-oriented assistant dialogues, response quality is not only about whether an answer is generally good. It also depends on the individual user's preferences, strictness, and previous interaction history.

So this work has two connected goals. First, we build a personalized turn-level satisfaction predictor. Second, we use this predictor as a frozen judge to evaluate candidate LLM responses under a static replay setting.

## Slide 2: Motivation: Satisfaction Is User-Specific

The core motivation is that user satisfaction is highly personalized.

In many task-oriented dialogues, two users may receive the same assistant response but evaluate it very differently. For example, one user may prefer a concise answer and give a high score. Another user may expect a detailed plan, concrete constraints, or step-by-step instructions, and may be dissatisfied with the same response.

This means that generic response quality is not enough. A response can be generally fluent and helpful, but still fail to satisfy a specific user.

We also focus on turn-level satisfaction. In a multi-turn dialogue, the overall conversation may look successful, but a single assistant turn can still ignore an important user constraint or provide an incomplete recommendation. Turn-level labels allow us to identify these local failures.

So the key question is: can we use the same user's historical satisfaction labels from other scenarios to predict how satisfied they will be with the current assistant response?

## Slide 3: Related Work

This work is related to four lines of research.

The first is dialogue user satisfaction estimation, such as USS, Frames, DEUS, and USDA. These works study how to estimate satisfaction in task-oriented dialogues, but they rarely focus on cross-scenario user-specific satisfaction using the same user's historical labels.

The second line is LLM-as-a-judge evaluation, such as G-Eval, MT-Bench, Chatbot Arena, and Prometheus. These methods evaluate response quality or model preference, but they are usually generic. They do not explicitly model whether a specific user would be satisfied.

The third line is personalized LLMs and user modeling, such as PersonaChat, LaMP, Personalized Soups, and PERSONAMEM. These works focus more on personalized generation or personalized alignment. Our focus is different: we use historical satisfaction labels to predict satisfaction.

The fourth line is user simulation and conversational recommender evaluation, such as UserSimCRS, MACRS, and SimpleUserSim. These methods simulate users dynamically. In contrast, our first benchmark uses static replay, where all candidate models answer the same historical prefixes and are scored by the same frozen personalized predictor.

## Slide 4: Overall Framework

This slide shows the overall framework of the project.

There are four main parts. First, we collect real user dialogue data, including task scenarios, user profiles, dialogues, and turn-level satisfaction labels.

Second, we formulate a personalized satisfaction prediction task. The predictor uses the same user's labeled histories from other scenarios, plus the current dialogue context and assistant response, to predict a 1-to-5 satisfaction score.

Third, we design a user-specific predictor. It builds a memory from historical labels, predicts the current turn satisfaction, and then applies user-aware calibration.

Finally, we use the frozen predictor for static replay evaluation. Candidate LLMs answer fixed historical dialogue prefixes, and the predictor scores their responses.

The important point is that the predictor is not only an analysis tool. It also enables a benchmark for comparing LLMs under personalized satisfaction.

## Slide 5: Data Collection Protocol

The data is collected under four planning-oriented scenarios: travel planning, gift preparation, recipe planning, and skill learning planning.

For each user, we collect profile information and ask the user to participate in multi-turn task-oriented dialogues. After the dialogue, each assistant turn is annotated with a satisfaction score from 1 to 5.

For low-score turns, we also collect dissatisfaction reasons. These reasons are important because they tell us not only that the user was dissatisfied, but also why the assistant response failed.

This data format allows us to study satisfaction at a fine-grained turn level, rather than only at the whole-dialogue level.

## Slide 6: Cross-Scenario Personalized Turn-Level Prediction

This slide defines the main prediction task.

For each target scenario, we do not use the user's labels from the same target scenario. Instead, we use the same user's labeled histories from other scenarios. This creates a cross-scenario personalization setting.

The input includes four parts: the user's labeled histories, the user profile, the current dialogue context, and the current assistant response.

The output is a predicted satisfaction score from 1 to 5. Optionally, when the score is low, the model also predicts a dissatisfaction reason.

The key challenge is that the model must transfer the user's scoring style and preferences from previous scenarios to a new current scenario.

## Slide 7: Evaluation Split

Here is the evaluation split used in the main experiments.

The test set contains 90 users, 356 user-task blocks, and 6,474 assistant turns. Each user-task block corresponds to one user and one target scenario.

The cross-task setting is important. It prevents the predictor from using gold labels from the target scenario during prediction. This makes the evaluation closer to a real personalization problem: we only know the user's past preferences and satisfaction patterns from other tasks.

## Slide 8: Memory-Based Predictor Agent

Our predictor has three main components.

First, memory construction. A memory-building LLM reads the user's historical dialogues and satisfaction labels, and summarizes the user's scoring behavior under different scenarios.

Second, turn-level judging. Given the memory, the current dialogue context, and the assistant response, another LLM predicts the satisfaction score and the reason.

Third, post-hoc calibration. Raw LLM predictions may have useful ranking signal but poor absolute score scale. Calibration maps the raw predictions onto the user's historical scoring scale.

So the predictor combines semantic judgment from the LLM with explicit user-history information.

## Slide 9: User Memory: What Is Extracted?

The user memory is designed to capture both statistical and semantic information.

It contains the score distribution and average historical satisfaction, which reflect whether the user tends to be strict or lenient.

It also contains scoring style and boundary distinctions. In particular, we ask the memory to describe what separates a 3 from a 4, and what separates a 4 from a 5 for this specific user.

This is important because the 3/4 boundary corresponds to dissatisfied versus satisfied. The 4/5 boundary tells us when a response is not only acceptable, but excellent for this user.

Finally, the memory stores user-specific requirements and scenario observations. These fields help the predictor avoid using only generic quality criteria.

## Slide 10: Calibration: Ranking vs. Score Scale

One important finding is that raw LLM judges often have useful relative ranking but poor absolute calibration.

For example, the model may correctly know that response A is better than response B, but its predicted scores may be systematically too high or too low for a particular user.

We therefore use two simple post-hoc calibration methods.

Mean shift aligns the predicted block mean with the user's historical mean. CDF or rank mapping ranks predictions within a block and maps them to the user's historical score distribution.

The intuition is: the LLM provides semantic ordering, while the user's history provides the score scale.

## Slide 11: Satisfaction Prediction Experimental Setup

For satisfaction prediction, we compare several types of baselines.

The first group is supervised baselines, such as BERT classification and BERT ordinal regression. These are trained on the aligned personalized train split.

The second group is RAG-based or retrieval-based methods. These include raw nearest retrieval and RAG-style absolute scoring prompts, where historical turns are used as evidence.

The third group is generic baselines, such as SPUR-style satisfaction estimation and generic LLM-as-a-judge methods. These baselines are useful because they test whether personalization is really necessary.

For evaluation, I focus on relative and ordinal metrics, including Pearson, Spearman, and quadratic weighted kappa. I also report dissatisfaction F1, because detecting dissatisfied turns is especially important.

## Slide 12: Satisfaction Prediction Results

This slide summarizes the main satisfaction prediction results.

The most important takeaway is that simple user-history baselines are very strong. This means user strictness and scoring style transfer across scenarios.

At the same time, the memory-based LLM predictor provides useful semantic signals, especially for ranking and dissatisfaction detection.

However, raw LLM scores are not perfectly calibrated. This motivates the calibration methods shown in the next result slide.

## Slide 13: LLM Judges Need Calibration

Here we compare Qwen3-8B predictor variants.

The no-memory baseline is much weaker, which suggests that user history is important. Adding memory improves correlation and ordinal agreement.

After calibration, the metrics improve further. Mean shift and CDF calibration both help because they adjust the raw LLM outputs to the user's historical score scale.

The main interpretation is that the LLM judge can provide useful semantic ranking, but the user's history is needed to map that ranking to the correct 1-to-5 satisfaction scale.

## Slide 14: Larger Models Change the Error Shape

This slide shows the effect of using stronger backbone models.

Qwen3.6-35B-A3B and gpt-5.4-mini generally improve the semantic signal, especially ranking and dissatisfaction detection.

But a larger model is not automatically better in every metric. For example, Qwen3.6 is stricter than Qwen3-8B. It detects more dissatisfied turns, but it can also under-score some truly satisfied responses.

So the conclusion is that model capacity matters, but calibration is still needed. The stronger model gives better semantic judgment, while calibration aligns it with the user's score distribution.

## Slide 15: Application: Static Replay Benchmark

The second part of the project is static replay evaluation.

We first select replay turns from historical dialogues. We prefer low-score or borderline turns, substantive assistant turns, and turns with enough context. This creates a harder benchmark than randomly selecting easy satisfied turns.

Then each candidate LLM receives the fixed historical dialogue prefix and generates one response. The generated response is not inserted into future dialogue turns, so all models are evaluated on the same contexts.

After that, a frozen personalized predictor scores each candidate response. Finally, we aggregate the predicted scores into benchmark metrics such as micro mean, user macro, task macro, SAT/DSAT rate, and confidence intervals.

The key design choice is that the judge is frozen, so every model is scored by the same user-specific predictor state.

## Slide 16: Static Replay Benchmark Results: Absolute

This slide shows absolute static replay scores for different candidate models.

All models are evaluated on the same hard replay subset. The scores are generated by the frozen Qwen3-8B personalized satisfaction judge.

The table reports different aggregation views. Micro mean averages over all turns. User macro first averages per user, so every user contributes equally. Task macro and block macro provide task-level and user-task-level views.

The main observation is that the top models form a clear group, while Minimax-M2.7 is consistently lower. We also use reference-CDF calibration to reduce raw score inflation and preserve model-level separation.

I would treat these numbers as benchmark estimates rather than human labels for the generated responses. The judge is automatic, so the ranking should be interpreted with that limitation in mind.

## Slide 17: Static Replay Benchmark Results: Pair-wise

Absolute scores are useful, but many LLM benchmarks also report pairwise comparisons.

Here, we compare each candidate model against GPT-5.5 as a reference model. For each replay case, we check whether the candidate receives a higher, equal, or lower predicted satisfaction score.

Because scores are discrete from 1 to 5, ties are common. So the non-tie win rate is especially important. It tells us who wins among the decided cases.

The top models, such as Kimi, GLM, and DeepSeek, are generally above GPT-5.5 on decided cases. Gemini and Claude are closer or below, and Minimax is clearly lower in this benchmark.

## Slide 18: Static Replay Benchmark Results: Pair-wise

This slide continues the pairwise analysis with another calibrated view.

The overall pattern is similar: the top-tier models remain stable, while Minimax is consistently worse than GPT-5.5.

An important point is that calibration can change the size of the gap, but it does not completely change the ranking structure. This suggests that the replay benchmark is capturing some stable differences between candidate models.

At the same time, we should not overclaim. Since the scores come from an automatic personalized predictor, pairwise results should be viewed as predictor-based benchmark signals, not direct human preference labels.

## Slide 19: Limitations

There are three main limitations.

First, language and user population bias. The current data is collected in Chinese, and most participants are students. This may introduce language-specific and population-specific satisfaction patterns.

Second, limited predictor reliability. The predictor captures useful personalized signals, but its correlation with human labels is still moderate. So it should be treated as a noisy evaluator, not a replacement for human judgment.

Third, limited scenario coverage. The current data mainly covers a small set of planning-oriented scenarios. These scenarios do not fully represent all real-world personalized assistant usage.

These limitations are important for interpreting both the prediction results and the static replay benchmark.

## Slide 20: Conclusion

To summarize, this work has three main contributions.

First, we use user-specific dialogue data with user profiles, task requirements, and turn-level satisfaction annotations under four scenarios.

Second, we build a personalized memory-based satisfaction predictor. The predictor uses user memory and post-hoc calibration to estimate turn-level satisfaction.

Third, we propose a static replay benchmark for LLMs. Candidate models answer fixed historical dialogue prefixes, and a frozen personalized predictor scores their responses.

The broader message is that general LLM ability is not the same as the ability to satisfy a specific user in an actual scenario. Personalized satisfaction should be an explicit evaluation target.

## Slide 21: Thanks

Thank you for listening.

I would be happy to discuss any part of the project, including the dataset construction, the satisfaction predictor, the calibration methods, or the static replay benchmark.
