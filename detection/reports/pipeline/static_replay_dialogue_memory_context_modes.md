# Static Replay Dialogue Memory Context Modes

## Motivation

The original static replay benchmark uses `replay_context_mode=raw` by default:
candidate LLMs only see the current dialogue prefix and generate the next
assistant reply. This evaluates vanilla next-turn inference, but does not test
whether a model can use long-term user memory.

To support a lightweight memory-augmented replay setting, two new context modes
were added:

- `dialogue_memory_tfidf`
- `dialogue_memory_diverse`

Both modes expose only raw historical dialogues from the same user and other
task scenarios. They do not expose satisfaction scores, dissatisfaction reasons,
or explicit user profiles.

## Relation To Prior Work

The design follows the general memory-agent idea used by prior long-term
memory systems such as Generative Agents and MemoryBank:

- Store past user interactions as episodic memories.
- Retrieve relevant memories dynamically at inference time.
- Use retrieved memories as soft evidence for personalized behavior.

This implementation is not a reproduction of those systems. It is a constrained
static replay variant designed for the current benchmark setting.

## Implementation

Code changes:

- `detection/lib/dialogue_memory.py`
  - Builds an unlabeled per-user dialogue memory index from
    `sample.history_sessions`.
  - Each historical assistant turn is kept as one memory item.
  - Memory fields include source task, source file, turn index, local dialogue
    history, last user message, and assistant reply.
  - Satisfaction labels, dissatisfaction reasons, and user profile fields are
    intentionally excluded.
- `detection/trace/collect_static_replay.py`
  - Adds `replay_context_mode` choices:
    - `dialogue_memory_tfidf`
    - `dialogue_memory_diverse`
  - Adds CLI options:
    - `--dialogue_memory_top_k`
    - `--dialogue_memory_max_chars_per_item`
    - `--dialogue_memory_local_history_size`
  - Writes retrieved memory metadata into each output record under
    `dialogue_memory_records`.
- `detection/scripts/collect_static_replay.sh`
  - Exposes the same dialogue-memory options as environment variables.

## Modes

### `dialogue_memory_tfidf`

This mode retrieves the top-k most similar historical dialogue memories using a
character n-gram TF-IDF index. The retrieval query is built from the current
dialogue prefix only. It does not use the original target assistant reply, so it
does not leak the answer being replayed.

### `dialogue_memory_diverse`

This mode first ranks memories by TF-IDF similarity, then selects memories with
source-task diversity. It is intended to reduce overfitting to one highly
similar historical scenario and expose broader cross-scenario user behavior.

## Candidate-Visible Prompt

For both modes, the candidate model receives:

1. A system message containing retrieved past dialogue memories.
2. The original current dialogue prefix.

The system message instructs the model to use memories only to infer stable
communication preferences and not to mention the memories. It explicitly states
that no satisfaction scores, reasons, or user profile annotations are provided.

## Example Commands

```bash
cd detection
model=gpt-5.5 \
selection_mode=hard \
replay_context_mode=dialogue_memory_tfidf \
dialogue_memory_top_k=4 \
dialogue_memory_max_chars_per_item=700 \
max_tokens=1024 \
max_workers=4 \
output_jsonl=outputs/static_replay/gpt-5.5_test_hard_dialogue_memory_tfidf_responses.jsonl \
bash scripts/collect_static_replay.sh
```

```bash
cd detection
model=gpt-5.5 \
selection_mode=hard \
replay_context_mode=dialogue_memory_diverse \
dialogue_memory_top_k=4 \
dialogue_memory_max_chars_per_item=700 \
max_tokens=1024 \
max_workers=4 \
output_jsonl=outputs/static_replay/gpt-5.5_test_hard_dialogue_memory_diverse_responses.jsonl \
bash scripts/collect_static_replay.sh
```

## Recommended First Evaluation

Run these modes on the existing hard subset before collecting a full replay
set. The first comparison should be:

- `raw`
- `dialogue_memory_tfidf`
- `dialogue_memory_diverse`

Use the same candidate LLM, selection mode, sample set, and downstream
satisfaction predictor so that any difference can be attributed to the replay
context.
