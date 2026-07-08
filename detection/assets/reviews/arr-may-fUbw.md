## Official Review by Reviewer fUbw
*03 Jul 2026, 12:09 (modified: 08 Jul 2026, 20:25)*

### Paper Summary
This paper introduces PersTurnBench, a personalized turn-level user satisfaction benchmark for planning-oriented assistant conversations, and proposes a user-memory- and profile-conditioned LLM judge as the estimator of personalized satisfaction. The judge summarizes each user's past rated conversations into structured memory, including rating style, satisfaction thresholds, preferences, and task-specific requirements, then predicts 1~5 satisfaction scores for target assistant turns. The judge achieves moderate agreement, as measured by Quadratic Weighted Kappa (0.3595), against user provided scores.

### Summary Of Strengths
- **S1.** The work represents a timely contribution in advancing personalized user satisfaction estimation, and the curated dataset is potentially useful for this line of research.
- **S2.** The proposed LLM judge is compared with a comprehensive set of baselines covering different families of automatic evaluators.
- **S3.** Adequate ablation studies show the impacts of structured memory, score calibration, and backbone capacity, as well as the significance of personalized contexts.

### Summary Of Weaknesses
- **W1.** Boundaries of 1/2 and 4/5 scoring are not clear enough, and the value of these differentiations is not sufficiently discussed or compared with a trinary SAT-Neutral-DSAT categorical schema. The presented results and lack of repeated-label/repeated-run analysis leave open the possibility of annotator self-inconsistency and evaluator nondeterminism, which are known concerns of generally less favorable rating-based evaluation schemas. Also,
- **W2.** The mapping of SPUR-style labels to 3/4 is somewhat problematic and might produce an unfair comparison. A reverse mapping to SAT-Neutral-DSAT may establish a better comparison. Additionally, the footnote of Table 2 is a misinterpretation of SPUR, as SPUR does produce neutral label from its pipeline.
- **W3.** The SAT and DSAT terminology in this paper is somewhat misleading, as 3 is actually defined as neutral in the rating rubric. Also, the paper should clarify whether genuine continuation of the conversation is treated as SAT instead of neutral, which may make the rating distribution heavily skewed toward SAT ratings (4/5).
- **W4.** The paper lacks a study on the impacts of varying user history amounts to assess the proposed LLM judge's robustness when historical data is sparse.
- **W5.** The dataset and the evaluator are built within a narrow domain, which limits their generalizability. Also, real world users may express mixed SAT/DSAT signals within the same turn. The proposed approach appears to force each turn into a single 1-5 satisfaction score, which may collapse these mixed reactions into one dominant label. This loss of nuance could limit the approach's usefulness for downstream applications such as preference optimization, reward modeling, and credit assignment.

### Comments, Suggestions And Typos
*(none)*

### Questions
- **Q1:** Are any descriptive statistics, such as error bars around results or summary statistics from sets of experiments, available for the results reported in Table 2? Are they the max, mean, etc., or just from a single run?
- **Q2:** What is the per-score distribution of the user history data and testing data?
- **Q3:** What was the model used as the user-facing assistant at the data curation stage? Was any bias observed when using the same model as the evaluator model?
- **Q4:** I'm a little confused about whether the SPUR rubric used in this paper was re-induced from the historical data or borrowed directly from the SPUR paper. Could you explain and share the rubric?

### Scores & Metadata
- **Confidence:** 4 = Quite sure. I tried to check the important points carefully. It's unlikely, though conceivable, that I missed something that should affect my ratings.
- **Soundness:** 2.5
- **Excitement:** 3 = Interesting: I might mention some points of this paper to others and/or attend its presentation in a conference if there's time.
- **Overall Assessment:** 2.5 = Borderline Findings
- **Ethical Concerns:** There are no concerns with this submission
- **Reproducibility:** 4 = They could mostly reproduce the results, but there may be some variation because of sample variance or minor variations in their interpretation of the protocol or method.
- **Datasets:** 3 = Potentially useful: Someone might find the new datasets useful for their work.
- **Software:** 2 = Documentary: The new software will be useful to study or replicate the reported research, although for other purposes it may have limited interest or limited usability. (Still a positive rating)
- **Knowledge Of Or Educated Guess At Author Identity:** Yes
- **Knowledge Of Paper:** After the review process started
- **Knowledge Of Paper Source:** other (specify)
- **Knowledge Of Paper Source Other:** While I was trying to check a link provided by the authors in the attachment, an accidentally included "<" symbol in the omnibox resulted in a Google search that returned an unanonymized Github repo.
- **Impact Of Knowledge Of Paper:** Not at all
