## Official Review by Reviewer 8qA9
*30 Jun 2026, 17:40 (modified: 08 Jul 2026, 20:25)*

### Paper Summary
The paper studies personalized turn-level satisfaction evaluation: predicting the 1–5 score a specific user would give a specific assistant turn from that user's labeled history in other scenarios. The method builds a comparative user memory (3→4 and 4→5 boundary rubrics) from score-grouped history, runs a rubric-ordered LLM judge (Qwen3-8B), and applies post-hoc calibration. On a new Chinese planning-task corpus, it beats supervised, retrieval, and generic LLM-judge baselines (Table 2). The verified evaluator is then frozen to build PersTurnBench, a replay benchmark scoring seven candidate generators on fixed conversation states.

### Summary Of Strengths
- The contributions of the paper are clear and sound: personalized turn-level user satisfaction evaluation is a useful target for assessing conversational models, and the paper formulates it cleanly against prior work (Table 1).
- The comparative-memory design is a genuine departure from profile/RAG personalization, turning score-grouped labels into 3→4 and 4→5 boundary rubrics that act as an executable per-user scoring guide.
- The proposed evaluator outperforms representative and recent baselines on all four metrics (Table 2), and the ablations separate the effect of memory from calibration across three backbones (Tables 3, 6).
- The proposed benchmark is useful: it provides real-user turn-level satisfaction labels with dissatisfaction reasons, and a replay protocol that compares generators without per-model relabeling.

### Summary Of Weaknesses
- The benchmark ranking is never validated against human judgment. Seven models are ranked by a frozen evaluator whose agreement with human labels is only modest (Pearson 0.360, QWK 0.360; Table 2), and per-turn agreement does not guarantee a correct system-level ordering. The replay design adds risk by scoring counterfactual responses the original users never rated. The authors should validate the ranking on a small stratified sample of candidate responses using human ratings or pairwise preferences.
- The benchmark reports calibrated scores only. All results (Fig. 2, Tables 8–9) use reference-CDF calibration, which maps models through the same gold distribution and can compress real differences, plausibly explaining the narrow 4.43–4.81 spread and the overlapping middle-group CIs. The ablations test calibration only against gold-label agreement, not against model spread or order. The authors should re-score the 300 replay turns with raw and mean-shift outputs and report rank correlation (Kendall's τ) across the leaderboards.

### Comments, Suggestions And Typos
N/A

### Scores & Metadata
- **Confidence:** 3 = Pretty sure, but there's a chance I missed something. Although I have a good feel for this area in general, I did not carefully check the paper's details, e.g., the math or experimental design.
- **Soundness:** 3.5
- **Excitement:** 3 = Interesting: I might mention some points of this paper to others and/or attend its presentation in a conference if there's time.
- **Overall Assessment:** 3 = Findings: I think this paper could be accepted to the Findings of the ACL.
- **Ethical Concerns:** There are no concerns with this submission
- **Needs Ethics Review:** No
- **Reproducibility:** 4 = They could mostly reproduce the results, but there may be some variation because of sample variance or minor variations in their interpretation of the protocol or method.
- **Datasets:** 4 = Useful: I would recommend the new datasets to other researchers or developers for their ongoing work.
- **Software:** 4 = Useful: I would recommend the new software to other researchers or developers for their ongoing work.
- **Knowledge Of Or Educated Guess At Author Identity:** No
- **Knowledge Of Paper:** N/A, I do not know anything about the paper from outside sources
- **Knowledge Of Paper Source:** N/A, I do not know anything about the paper from outside sources
- **Impact Of Knowledge Of Paper:** N/A, I do not know anything about the paper from outside sources
