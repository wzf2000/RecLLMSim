## Official Review by Reviewer dEr8
*01 Jul 2026, 18:23 (modified: 08 Jul 2026, 20:25)*

### Paper Summary
This paper evaluates user satisfaction with an assistant at the level of a single turn, for a specific user rather than on average. The motivation is that generic automatic evaluators tell you whether a response is good in general, but not whether it satisfies this user at this point in the conversation.

There are three pieces. (1) A training-free memory-based evaluator (Section 3): it summarizes a user's labeled history from other scenarios into a compact, comparative user memory (Section 3.2), scores a target turn against it with a rubric-style decision order (Section 3.3), and rescales the raw score to the user's own rating distribution via post-hoc calibration (Section 3.4); a cross-scenario protocol (Eq. 9) keeps the target labels out. (2) A meta-evaluation against human labels (Section 4, Table 2), where the evaluator beats supervised, retrieval, and generic LLM-judge baselines on Pearson/Spearman/QWK and F1-DSAT, with an ablation (Table 3) showing the memory drives the gain. (3) PersTurnBench (Section 5), which freezes the verified evaluator as a judge and compares candidate generation models on fixed conversation states (replay), reporting user-macro satisfaction and a dissatisfied-turn rate (Figure 2, Table 8).

The data (Appendix B, Table 5) come from 115 real Chinese-speaking users on four planning tasks: 8,060 labeled assistant turns on a 1-5 scale with categorical dissatisfaction reasons. Code and anonymized data are released, and the authors frame PersTurnBench as a low-cost screening layer before human studies, not a replacement for them.

### Summary Of Strengths
- The problem is well chosen: judging satisfaction per turn for an individual user is a real gap that generic quality metrics and population-level preference miss.
- The dataset is collected carefully, with an anchored 1-5 scale, a fixed dissatisfaction taxonomy, and clear attention to leakage.
- The comparative memory is a nice idea: instead of a descriptive summary it contrasts adjacent score levels into an actionable rubric, and the ablation shows this is what drives the gain (F1-DSAT 0.046 → 0.351, Table 3).
- The scoping is honest and the work is reproducible: PersTurnBench is framed as a screening layer rather than a replacement for human evaluation, and code, data, and full configurations (Appendix D, H) are released, including a negative result (Appendix G).

### Summary Of Weaknesses
- The agreement scores have no reference point. Table 2 reports Pearson 0.3601, QWK 0.3595, and F1-DSAT 0.3655, but there's no annotator agreement, test-retest reliability, or human baseline to anchor them, and since satisfaction is self-reported the ceiling is well below 1.0. So I can't tell whether 0.36 is near the ceiling or far from it, and only the ranking over baselines is meaningful. A reliability estimate on a small subset would settle this.
- Table 2 also omits the simple baselines the code already implements — majority/mean predictors, and a user-history CDF baseline that maps each turn to the user's score distribution without reading the response. That last one is the key control: it shows how much of the Pearson/Spearman/QWK correlation is genuine judging versus just matching the user's distribution, which matters when 84% of turns are scored 4-5 (Table 5). These belong in Table 2.
- The benchmark leans on a judge that is only weakly verified: a judge agreeing with humans at about 0.36 (Table 2) is used to rank much stronger candidate models (Figure 2) it was never verified on, with no human labels, and the top models' confidence intervals already overlap (e.g. 4.812 vs 4.791, Table 8). The reported scores also mix calibrated and raw values, since the released calibration falls back to identity when a replay record lacks history. The paper frames this as a screening tool with grouped rather than strict rankings, so I would not call it broken — but the comparisons should be read cautiously.

### Comments, Suggestions And Typos
My main suggestions for improving the paper are already covered in the weaknesses above. Beyond those, I have the following additional comments and a typo to report:
- Table 1 lists "Direct" (feedback from original users) as a property of this work, but it only holds for the Section 4 verification — in the benchmark itself the original users never see the candidate responses, which the frozen judge scores. Separating the two would avoid overstating the comparison.
- Typo: the second contribution bullet in Section 1 ("Meta-evaluation between the proposed evaluator ...") reads awkwardly and should be reworded.

### Limitations And Societal Impact
The limitations (Section 7) are honest and appropriately scoped, and the authors should be credited for that. The one thing I would add is the within-user agreement ceiling from weakness 1, since it bounds what the evaluator can ever reach.

### Scores & Metadata
- **Confidence:** 4 = Quite sure. I tried to check the important points carefully. It's unlikely, though conceivable, that I missed something that should affect my ratings.
- **Soundness:** 3 = Acceptable: This study provides sufficient support for its main claims. Some minor points may need extra support or details.
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
