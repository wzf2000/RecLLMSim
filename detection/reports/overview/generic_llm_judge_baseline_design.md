# Generic LLM-as-Judge Baseline Implementation

## Goal

This baseline family evaluates whether a non-personalized LLM judge can predict
turn-level satisfaction on the same cross-task personalized split used by the
main predictor pipeline.

Unlike the personalized memory agent, these baselines do not use:

- user profile
- source-task user history
- summarized user memory
- memory updates during target prediction

The judge only sees the target task context, dialogue history before the target
assistant reply, and the assistant reply to score.

## Implemented Files

- `detection/eval/generic_llm_judge.py`
- `detection/scripts/run_generic_llm_judge.sh`

The output JSONL is compatible with `detection/eval/personalized.py`.

## Variants

| Variant | Description |
|---|---|
| `zero_shot` | Generic 1-5 satisfaction rubric, no examples. |
| `few_shot_global` | Adds balanced labeled examples sampled from train users. Examples are explicitly treated as global demonstrations, not user memory. |
| `task_rubric` | Adds a hand-written task-level rubric for travel planning, gift preparation, recipe planning, or skill learning. |
| `prometheus_rubric` | Uses a Prometheus-style evaluation layout with task description, score rubric, criteria, and output contract. |

All variants output:

- `pred_score`: integer 1-5
- `reason_prediction`: `满意` if `pred_score >= 4`; otherwise one dissatisfied reason
- `parse_ok`: whether the LLM output was parsed successfully

If parsing or calling fails after retries, the fallback prediction is
`pred_score=3`, `reason_prediction=其它`, and `parse_ok=false`.

## Backend Support

The script supports both default API client and OpenAI-compatible custom
endpoints such as vLLM.

If `base_url` is empty, it uses the project default API client from
`detection/lib/llm.py`.

If `base_url` is set, it creates a separate OpenAI-compatible client:

```bash
base_url=http://localhost:8000/v1
api_key=EMPTY
```

## Example Commands

Run Qwen3-8B through local vLLM:

```bash
cd detection
model=Qwen/Qwen3-8B \
base_url=http://localhost:8000/v1 \
api_key=EMPTY \
variant=zero_shot \
max_workers=4 \
output_jsonl=outputs/personalized/generic_judge_qwen3_8b_zero_shot_test.jsonl \
bash scripts/run_generic_llm_judge.sh
```

Run a third-party API model through the default client:

```bash
cd detection
model=gpt-4o-mini \
variant=task_rubric \
max_workers=8 \
output_jsonl=outputs/personalized/generic_judge_gpt4o_mini_task_rubric_test.jsonl \
bash scripts/run_generic_llm_judge.sh
```

Smoke test on a small subset:

```bash
cd detection
model=Qwen/Qwen3-8B \
base_url=http://localhost:8000/v1 \
api_key=EMPTY \
variant=zero_shot \
limit_users=2 \
max_workers=1 \
output_jsonl=outputs/personalized/generic_judge_qwen3_8b_zero_shot_smoke.jsonl \
bash scripts/run_generic_llm_judge.sh
```

Evaluate:

```bash
cd detection
result_file=outputs/personalized/generic_judge_qwen3_8b_zero_shot_test.jsonl \
bash scripts/eval_personalized.sh
```

## Notes

This baseline is intentionally non-personalized. It should be compared against
personalized memory methods to quantify the gain from user-specific modeling
rather than general LLM judging ability.
