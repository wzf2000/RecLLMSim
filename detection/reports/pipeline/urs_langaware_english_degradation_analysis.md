# URS Language-Aware English Degradation Analysis

## Summary

`urs_v2_calibrated_langaware` does not improve the URS evaluator on the English subset.
Compared with `urs_v2_calibrated`, the English subset drops from Pearson `0.1961` / QWK `0.1750` to Pearson `-0.0021` / QWK `-0.0017`.
The degradation is not caused by coverage or parsing failures; both versions cover the same `220` English sessions.

## Distribution Shift

For English memory-based predictions:

- `147 / 220` sessions keep the same score.
- `49 / 220` sessions are decreased by 1 point.
- `18 / 220` sessions are increased by 1 point.
- The main harmful pattern is high-gold sessions being pushed down to `3`.

By gold score:

- Gold `5`: `19 / 63` sessions worsen, only `3 / 63` improve.
- Gold `4`: `21 / 96` sessions worsen, `12 / 96` improve.
- Gold `3`: `2 / 39` sessions worsen, `7 / 39` improve.
- Gold `2`: `5 / 18` sessions worsen, `2 / 18` improve.

By intent:

- Information Retrieval: `25 / 85` worsen.
- Ask for Advice: `6 / 47` worsen.
- Seek Creativity: `5 / 30` worsen.
- Leisure: `5 / 29` worsen.
- Solve Professional Problem: `3 / 21` worsen.
- Text Assistant: `4 / 8` worsen.

## Main Failure Modes

### 1. Over-penalizing truncated English responses

Many URS English responses are visibly cut off in the source data, but the gold labels still often treat them as satisfactory.
The original Chinese calibrated prompt tends to focus on whether the main intent is mostly addressed.
The English language-aware prompt explicitly reasons about "complete dialogue", "substantive flaw", and "requires follow-up", so it often treats truncation as enough to cross below the 3/4 boundary.

Examples:

- `en_12__retrieval__urs::en::00020.json__turn_0`: gold `5`, calibrated `4`, langaware `3`.
  The answer explains "what is life" from multiple perspectives but is cut off near the end.
  Langaware focuses on the cutoff and says the answer lacks "detailed, practical guidance", which is mismatched for this philosophical query.
- `en_142__creative__urs::en::00250.json__turn_0`: gold `5`, calibrated `5`, langaware `3`.
  The story is long and covers the core birth story details, but the source answer ends at "Mama, Dada, and Na...".
  Langaware treats this as an unresolved story and drops to `3`.
- `en_7__leisure__urs::en::00009.json__turn_0`: gold `5`, calibrated `4`, langaware `3`.
  Langaware says the response is cut off before completing a list, so the main task is not fulfilled.

### 2. Overusing memory-derived "specific / structured / actionable" requirements

The URS memory is built from few cross-intent sessions.
For many English users, it summarizes generic preferences such as detailed, structured, practical, actionable, or step-by-step.
The language-aware prompt follows these requirements more literally than the Chinese prompt, causing unrelated short-answer tasks to be under-scored.

Examples:

- `en_133__retrieval__urs::en::00241.json__turn_0`: gold `5`, calibrated `4`, langaware `3`.
  The assistant identifies the correct bus route from Victoria Station to Lewisham.
  Langaware penalizes missing frequency, timing, and alternatives, even though the core fact is sufficient for a high score.
- `en_132__advice__urs::en::00234.json__turn_0`: gold `5`, calibrated `4`, langaware `3`.
  Langaware judges three stretching suggestions as too generic because the memory prefers detailed practical advice.
- `en_79__professional__urs::en::00156.json__turn_0`: gold `5`, calibrated `4`, langaware `3`.
  Langaware asks for broader coverage such as fire safety, electrical requirements, legal compliance, and emotional support, which exceeds the observed gold standard.

### 3. Prompt wording makes English 3/4 stricter than intended

The English calibration says "Use 3 only when there is a substantive flaw", but the step prompt asks the model to decide whether the response "meets the user's 3-to-4 satisfaction threshold" using personalized requirements.
In practice, the model treats missing extra details as a substantive flaw.
This weakens the intended calibration rule: "if the main task is solved and there is no serious error, prefer 4".

Example:

- `en_74__retrieval__urs::en::00151.json__turn_0`: gold `2`, calibrated `3`, langaware `4`.
  The user asks for the cheapest UK-available non-China-made valve headphone amplifier.
  Langaware correctly notes missing price and manufacturing verification, but still says the main task is addressed and assigns `4`.
  This shows the prompt is not consistently stricter; it is unstable around the 3/4 boundary.

### 4. Language-aware helps some cross-intent memory failures, but not enough

There are cases where English langaware fixes memory overgeneralization.

Example:

- `en_163__retrieval__urs::en::00293.json__turn_0`: gold `4`, calibrated `2`, langaware `4`.
  The old calibrated prompt incorrectly applies leisure/travel memory to an economics definition query.
  Langaware explicitly notes that travel-related requirements are irrelevant and correctly scores the answer as satisfactory.

This is a real benefit, but it is outweighed by the high-score downgrades and unstable boundary decisions.

## Conclusion

The English degradation is mainly a prompt-behavior problem, not a data-processing problem.
The language-aware prompt makes Qwen3-8B pay more attention to literal completeness, cutoff artifacts, and generic memory requirements.
That is poorly aligned with URS English gold labels, where many short or partially truncated answers are still rated `4` or `5` if the main need is mostly met.

The current `urs_v2_calibrated_langaware` should not be used as the default URS evaluator.
If continuing this direction, the English prompt should keep the Chinese calibrated prompt's lenient main-task-first behavior and add explicit rules:

- Do not lower below `4` solely because the answer is short or slightly truncated if the core answer is already present.
- Treat memory-derived preferences as weak evidence unless directly relevant to the current intent.
- For factual short-answer retrieval, prioritize correctness and directness over extra structure.
- Require a serious factual error, clear mismatch, unsafe answer, or missing core answer before assigning `<=3`.
