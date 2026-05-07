# Turn Content Filter Design

## Motivation

The first turn-position analysis showed that `turn_idx` is only a rough proxy
for whether an assistant reply contains substantive task content. In the U20
subset, turn 0 has the highest gold DSAT rate, so simply filtering early turns
would change the label distribution rather than only remove low-signal setup
turns.

A better split is to annotate each assistant turn by content type:

- content-like turns: the assistant gives concrete task content, or gives both
  content and clarification
- non-content turns: the assistant mostly asks clarifying questions, acknowledges
  the request, or gives meta/process text

## Implementation

New script:

```text
detection/eval/turn_content_filter.py
```

The script has two subcommands:

1. `annotate`: reconstruct target turns from `build_personalized_samples`, call
   an OpenAI-compatible LLM with structured output, and write annotation JSONL.
2. `analyze`: join existing prediction JSONL files with annotation JSONL and
   report metrics by content group and content type.

The default annotation route uses ordinary `chat.completions.create`, asks the
model to emit a compact JSON object, caps generation with `--max_tokens`, and
does local JSON validation. This is intentionally different from SDK schema
parse because vLLM guided schema decoding can be slow or timeout for this simple
annotation task. `--use_schema_parse` remains available as an opt-in debug
route.

Annotation schema:

```json
{
  "has_substantive_content": true,
  "content_type": "substantive_answer | mixed_answer | clarifying_question | ack_or_meta | other",
  "task_relevance": "none | low | medium | high",
  "rationale": "short explanation"
}
```

`has_substantive_content=true` for:

- `substantive_answer`
- `mixed_answer`

Examples:

- A concrete itinerary, recipe, gift list, study plan, explanation, checklist, or
  recommendation is content-like.
- A reply that asks for budget/date/preferences but also gives useful interim
  options is `mixed_answer` and content-like.
- A reply that only asks for more information is `clarifying_question` and
  non-content.
- A reply that only acknowledges or says it will help is `ack_or_meta` and
  non-content.

The annotation prompt includes only the current turn exchange:

```text
用户: ...
当前助手回复: ...
```

It does not include earlier dialogue history and does not truncate the current
exchange. This avoids leaking substantive content from previous turns into a
current clarifying/meta reply, and avoids cutting off long substantive answers.

## Suggested U20 Run

Annotate the current U20 subset once:

```bash
cd detection
python eval/turn_content_filter.py annotate \
  --result_file outputs/personalized/history_prior_delta_v3_none_n3_limit20.jsonl \
  --output_jsonl outputs/personalized/turn_content_annotations_u20.jsonl \
  --vllm_base_url http://localhost:8000/v1 \
  --vllm_api_key EMPTY \
  --model Qwen/Qwen3-8B \
  --timeout 120 \
  --max_tokens 256
```

Then analyze selected routes:

```bash
python eval/turn_content_filter.py analyze \
  --annotations_jsonl outputs/personalized/turn_content_annotations_u20.jsonl \
  --result_files \
    history_median=outputs/personalized/history_baselines/user_history_median_test.jsonl \
    hpd_v2=outputs/personalized/history_prior_delta_v2_none_n3.jsonl \
    hpd_v3=outputs/personalized/history_prior_delta_v3_none_n3_limit20.jsonl \
    hpd_v3_ms=outputs/personalized/history_prior_delta_v3_none_n3_limit20_calMS.jsonl \
    hpd_v1_cdf=outputs/personalized/history_prior_delta_none_n3_limit20_calCDF.jsonl \
  --output_json outputs/personalized/turn_content_compare_u20.json
```

The analysis output contains:

- `groups.all_annotated`
- `groups.content_like`
- `groups.non_content`
- `by_content_type`
- `by_turn`

Each group reports:

- MAE / RMSE
- Pearson / Spearman / QWK
- binary accuracy
- F1-DSAT / precision-DSAT / recall-DSAT
- false SAT / false DSAT
- gold and predicted DSAT rates

## Expected Use

This split should answer a cleaner question than `turn_idx>=2`:

> Does a method still help on turns where the assistant actually gives task
> content, and is its DSAT performance mainly coming from clarifying/meta turns?

For reporting, use both:

- turn-position analysis, because it is deterministic and cheap
- content-type analysis, because it better matches the semantic concern
