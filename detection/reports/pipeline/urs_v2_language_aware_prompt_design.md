# URS V2 Language-Aware Prompt Design

## Motivation

Previous URS predictor runs consistently showed weaker English performance than
Chinese performance. For example, with `mem_calibrated`:

- zh: MAE `0.6923`, QWK `0.2347`, F1-DSAT `0.4688`
- en: MAE `0.7818`, QWK `0.1750`, F1-DSAT `0.4412`

The existing URS evaluator prompt is mostly Chinese even when the target
dialogue is English. This may add unnecessary friction for English sessions.

## New Prompt Version

Added:

- `urs_prompt_version=urs_v2_calibrated_langaware`

Behavior:

- Chinese sessions use the same Chinese calibration rubric as
  `urs_v2_calibrated`.
- English sessions use an English session-level rating rubric and English
  step instructions.
- The output schema remains unchanged and still uses the existing reason labels
  for compatibility.

## Language Detection

Language is detected from `task_context` and the first dialogue messages:

- If ASCII alphabetic characters dominate CJK characters, the session is treated
  as English.
- Otherwise it is treated as Chinese.

This is intentionally lightweight and avoids changing the URS data loader.

## Recommended Commands

Memory version:

```bash
cd detection
model=Qwen/Qwen3-8B \
vllm_base_url=http://localhost:8000/v1 \
vllm_api_key=EMPTY \
memory_update_mode=none \
urs_prompt_version=urs_v2_calibrated_langaware \
max_workers=4 \
output_jsonl=outputs/urs/Qwen_Qwen3-8B_test_none_urs_v2_calibrated_langaware.jsonl \
bash scripts/collect_urs.sh
```

No-memory version:

```bash
model=Qwen/Qwen3-8B \
vllm_base_url=http://localhost:8000/v1 \
vllm_api_key=EMPTY \
no_memory=1 \
urs_prompt_version=urs_v2_calibrated_langaware \
max_workers=4 \
output_jsonl=outputs/urs/Qwen_Qwen3-8B_test_no_memory_urs_v2_calibrated_langaware.jsonl \
bash scripts/collect_urs.sh
```

## Evaluation

Compare against:

- `urs_v2_calibrated`
- `urs_v2_memory_guarded`
- `no_memory + urs_v2_calibrated`

Primary focus:

- English subset metrics
- overall QWK / Pearson / Spearman
- F1-DSAT change
- predicted score distribution

## V2 English Adjustment

Initial full-run results showed that the English language-aware branch degraded
English subset correlation sharply. The main issue was not parsing or coverage:
the prompt became too sensitive to short/truncated English answers and applied
generic memory requirements such as "structured", "detailed", or "actionable"
too strongly.

The English branch was therefore adjusted with a main-task-first policy:

- If the current session's core request is answered correctly and usefully,
  assign at least `4` unless there is a serious flaw.
- Do not downgrade below `4` solely because an answer is short, lacks extra
  structure, or is lightly truncated after already giving the core answer.
- For factual short-answer or definition tasks, prioritize correctness and
  directness over broad coverage or step-by-step detail.
- Treat personalized memory as weak evidence and apply it only when directly
  relevant to the current intent.
- Do not import requirements from unrelated intents, such as using travel
  preferences to judge economics definitions.

This keeps the language-aware route available while making it closer to the
original calibrated prompt's URS gold-label behavior.
