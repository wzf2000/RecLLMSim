# Episodic RAG Memory Baseline Design

## Motivation

This baseline avoids per-user/task summarized `UserMemory`. For each
`PersonalizedSample`, it keeps every labeled historical assistant turn from the
same user as an independent retrievable memory item, then predicts target-turn
satisfaction with retrieved raw evidence.

The goal is to test whether user-specific satisfaction can be improved by
retrieving concrete prior cases instead of relying on compressed memory
summaries.

## Memory Item

Each historical assistant turn is converted into one `EpisodicMemoryRecord`:

- `memory_id`: `{user}__{task}__{file}__turn_{idx}`
- `source_task`, `source_file`, `turn_idx`
- `task_context`
- `local_history`
- `last_user_msg`
- `assistant_reply`
- `score`
- `reason`
- `score_side`: `dsat`, `sat_4`, or `sat_5`

No summarization or memory update is performed. The memory corpus is fixed for a
block and only contains cross-task history sessions.

## Retrieval

Implementation: `detection/lib/episodic_rag.py`

The current retriever uses char-level TF-IDF over:

- task context
- last user message, duplicated to increase query weight
- assistant reply

Supported strategies:

- `topk_similar`: pure top-k semantic/textual similarity.
- `boundary_paired`: retrieve both dissatisfied (`<=3`) and satisfied (`>=4`)
  evidence near the 3/4 boundary, then fill remaining slots by similarity.
- `score_balanced`: retrieve from low-score, score-4, and score-5 buckets.
- `nearest`: same retrieval as top-k, intended for the no-LLM weighted-neighbor
  baseline.

## Prediction

Implementation:

- `detection/lib/episodic_rag_prompts.py`
- `detection/trace/episodic_rag_predictions.py`
- `detection/trace/episodic_rag_runner.py`
- `detection/trace/collect_personalized_episodic_rag.py`

The LLM prompt receives:

- user profile
- current task context
- current dialogue window
- candidate assistant reply
- retrieved raw historical memory records with true score and reason

The model must output strict JSON:

```json
{
  "classification": 4,
  "reason": "满意",
  "analysis": "short explanation",
  "boundary_side": "sat",
  "evidence_confidence": "medium"
}
```

Reason consistency is enforced in the prompt and normalized in code:

- `classification >= 4` must map to `reason = "满意"`.
- `classification <= 3` must map to a dissatisfied reason label.

The `episodic_rag_nearest` prompt version is a no-LLM smoke/evaluation baseline.
It predicts a rounded similarity-weighted average score from retrieved memories.

### Boundary-First Prompt Version

`episodic_rag_boundary_first` keeps the same raw memory retrieval, but changes the
LLM decision process:

1. Phase 1 predicts only the 3/4 boundary:
   `boundary_decision = sat | dsat`.
2. Phase 2 refines inside the chosen side:
   `severe_dsat=1`, `clear_dsat=2`, `near_boundary_dsat=3`,
   `qualified_sat=4`, `strong_sat=5`.
3. The schema validates that `boundary_decision`, `score_refinement`, and
   `classification` are mutually consistent.

This variant is intended to avoid the direct 1-5 prompt's observed all-4
collapse by forcing the model to decide the SAT/DSAT boundary before choosing a
score.

## Output Compatibility

The collector writes JSONL records with the same core fields as existing
personalized prediction outputs:

- `sample_id`
- `user`
- `target_task`
- `target_file`
- `turn_idx`
- `model`
- `with_memory`
- `memory_update_mode`
- `memory_version`
- `turn_eval_prompt_version`
- `gold_score`
- `pred_score`
- `gold_reason`
- `reason_prediction`
- `analysis`

Additional retrieval diagnostics:

- `retrieval_strategy`
- `retrieval_top_k`
- `retrieved_memory_ids`
- `retrieved_scores`
- `retrieved_reasons`
- `retrieved_tasks`
- `retrieved_roles`
- `retrieved_similarities`
- `episodic_boundary_side`
- `episodic_evidence_confidence`

## Commands

No-LLM smoke test:

```bash
cd detection
turn_eval_prompt_version=episodic_rag_nearest \
retrieval_strategy=boundary_paired \
top_k=6 \
limit_users=1 \
max_workers=1 \
output_jsonl=outputs/personalized/episodic_rag_nearest_smoke.jsonl \
bash scripts/collect_personalized_episodic_rag.sh
```

Qwen3 via vLLM:

```bash
cd detection
model=Qwen/Qwen3-8B \
vllm_base_url=http://localhost:8000/v1 \
vllm_api_key=EMPTY \
turn_eval_prompt_version=episodic_rag \
retrieval_strategy=boundary_paired \
top_k=6 \
max_workers=4 \
output_jsonl=outputs/personalized/qwen3_8b_test_episodic_rag_boundary_k6.jsonl \
bash scripts/collect_personalized_episodic_rag.sh
```

Boundary-first Qwen3 via vLLM:

```bash
cd detection
model=Qwen/Qwen3-8B \
vllm_base_url=http://localhost:8000/v1 \
vllm_api_key=EMPTY \
turn_eval_prompt_version=episodic_rag_boundary_first \
retrieval_strategy=boundary_paired \
top_k=6 \
limit_users=20 \
max_workers=4 \
output_jsonl=outputs/personalized/qwen3_8b_test_episodic_rag_boundary_first_k6_limit20.jsonl \
bash scripts/collect_personalized_episodic_rag.sh
```

OpenAI-compatible API:

```bash
cd detection
model=gpt-4o-mini \
turn_eval_prompt_version=episodic_rag \
retrieval_strategy=score_balanced \
top_k=6 \
max_workers=8 \
output_jsonl=outputs/personalized/gpt4o_mini_test_episodic_rag_score_balanced_k6.jsonl \
bash scripts/collect_personalized_episodic_rag.sh
```

## Initial Analysis Plan

Recommended first pass:

1. Run `episodic_rag_nearest` on a fixed 20-user subset to verify retrieval
   quality and output compatibility without LLM variance.
2. Compare `boundary_paired` vs `score_balanced` with the same subset.
3. Run Qwen3-8B `episodic_rag` on the best retrieval strategy.
4. Inspect retrieval diagnostics for wrong predictions, especially whether
   retrieved examples are truly comparable or only lexically similar.
