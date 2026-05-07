# User Memory Retrieval Literature and Design Notes

Date: 2026-05-07

## Current Project State

Current personalized satisfaction prediction primarily uses one structured memory object per `user + target_task + model`. The cached memory summarizes historical sessions into:

- numeric calibration: historical average score and 1-5 score distribution
- boundary rubrics: `3/4` satisfaction boundary and `4/5` excellence boundary
- user-specific requirements, preferred response format, and task-specific observations

There is an optional anchor retrieval path (`--n_anchors`) that builds an in-memory TF-IDF index over historical turns and inserts similar labeled turns into the prompt. This is useful, but it is not yet a persistent user memory corpus: it is rebuilt per sample, uses shallow lexical retrieval, is not shared across runs as an indexed artifact, and does not maintain multi-granularity or provenance-aware memory records.

Therefore, the current architecture is best described as:

> summary memory + optional transient anchor retrieval, not a full retrieval-based long-term user memory system.

## Recent Relevant Work

### Memory hierarchy and persistent memory

MemGPT introduced an OS-inspired memory hierarchy where a small in-context memory is paired with external archival memory. The core idea is to page relevant memories into context rather than feeding the entire history. This is relevant because our current `UserMemory` is effectively a compact core memory, but we do not yet have a persistent archival layer for historical evidence. Source: [MemGPT, arXiv:2310.08560](https://arxiv.org/abs/2310.08560).

MemoryBank focuses on long-term user interaction, memory updating, and selective forgetting/reinforcement inspired by the Ebbinghaus forgetting curve. The useful lesson for this project is not the companion-chat setting itself, but the separation between stable user profile, evolving memories, and relevance-based recall. Source: [MemoryBank, arXiv:2305.10250](https://arxiv.org/abs/2305.10250).

### Production-style memory extraction and retrieval

Mem0 proposes extracting, consolidating, and retrieving salient information from conversations, with a graph-memory variant for relational structure. It reports large latency/token savings versus full-context processing while improving long-term dialogue QA. For this project, the closest transferable idea is a memory ingestion stage that turns each historical turn/session into compact, retrievable memory facts while retaining enough provenance to audit predictions. Source: [Mem0, arXiv:2504.19413](https://arxiv.org/abs/2504.19413).

MemMachine argues for preserving episodic ground truth rather than relying only on lossy LLM summaries. It combines short-term, long-term episodic, and profile memory, and uses contextualized retrieval that expands a matched nucleus with surrounding dialogue context. This directly matches a weakness in our current summary memory: the model sees the abstract rule but often lacks concrete evidence for applying the rule to a target turn. Source: [MemMachine, arXiv:2604.04853](https://arxiv.org/abs/2604.04853).

### Adaptive and agentic retrieval

RF-Mem proposes a dual-path user memory retriever: direct top-k retrieval when familiarity is high, and iterative recollection-style expansion when familiarity is uncertain. It specifically targets personalized LLM memory retrieval and argues that one-shot similarity search captures only surface matches. This is highly relevant to our anchor retrieval: easy target turns can use cheap top-k anchors, while ambiguous boundary cases can trigger deeper retrieval over neighboring turns, task-similar sessions, or score-boundary evidence. Source: [RF-Mem, arXiv:2603.09250](https://arxiv.org/abs/2603.09250).

A-MEM organizes memory with Zettelkasten-style notes, dynamic indexing, linking, and memory evolution. For our task, this suggests representing each historical satisfaction event as a note with fields such as task, user request, assistant reply, score, reason, extracted success/failure factors, and links to similar or contrasting memories. Source: [A-MEM, arXiv:2502.12110](https://arxiv.org/abs/2502.12110).

MemInsight uses autonomous memory augmentation to improve semantic representation and retrieval, reporting gains over a RAG baseline on LoCoMo retrieval and recommendation. The useful direction is to enrich raw turns with LLM-generated attributes before retrieval, rather than retrieving only by raw text similarity. Source: [MemInsight, arXiv:2503.21760](https://arxiv.org/abs/2503.21760).

### Graph and structured memory

PersonalAI compares knowledge-graph storage and retrieval approaches for personalized LLM agents, including hybrid graph designs and retrieval mechanisms such as A*, WaterCircles, beam search, and hybrid methods. This is probably too heavy as a first implementation for satisfaction prediction, but it motivates a lightweight graph-like layer: connect user requirements, tasks, score-boundary examples, and contradictory evidence. Source: [PersonalAI, arXiv:2506.17001](https://arxiv.org/abs/2506.17001).

## Implications for This Project

### 1. Add an episodic memory corpus, but keep summary memory

The strongest near-term direction is not to replace `UserMemory`, but to split memory into two layers:

- `profile_summary`: current v2/v3 memory, used as compact prior and scoring rubric
- `episodic_memory`: retrievable labeled historical turns/sessions, used as evidence

Each episodic record should preserve raw provenance:

- `user`, `source_task`, `session_file`, `turn_idx`
- preceding user message and assistant reply
- gold score and dissatisfied reason
- task context and short local dialogue window
- extracted attributes: request type, constraints, response strengths, response failures, format features, boundary tags

This follows the MemMachine lesson: preserve ground-truth episodes and use summaries only as an index/abstraction, not as the sole memory.

### 2. Upgrade anchor retrieval from lexical top-k to typed evidence retrieval

Current anchor retrieval is character n-gram TF-IDF over `user_msg + assistant_reply`. It can be improved without introducing a complex graph system:

- semantic embedding retrieval over `user_msg`, `assistant_reply`, and extracted attributes
- score-aware retrieval: retrieve nearest examples from both sides of the target boundary, especially `3` vs `4` and `4` vs `5`
- task-aware retrieval: boost same-task or same-request-type memories
- evidence expansion: once a turn is retrieved, include its neighboring turns or full session summary when useful
- contradiction retrieval: retrieve examples that look similar but had different scores, because those are most informative for boundary calibration

This is closer to RF-Mem and MemMachine than to the current one-shot anchor retriever.

### 3. Use adaptive retrieval only for difficult cases

Full retrieval on every turn may slow the pipeline and add prompt noise. A practical policy:

- normal cases: use summary memory only
- uncertain cases: retrieve 2-4 episodic anchors
- boundary cases: retrieve paired evidence from both sides of `3/4`
- score-scale cases: retrieve paired evidence from `4/5`

Trigger signals can come from the predictor itself:

- low `boundary_confidence`
- small margin between SAT/DSAT evidence
- `delta_confidence=medium/low`
- inconsistency among `classification`, `delta_score`, and `boundary_score`

This maps naturally onto the existing `history_prior_delta_v*` structured outputs.

### 4. Prefer lightweight structured memory before full graph memory

Graph memory is attractive but likely not the first priority. The dataset is turn-level satisfaction prediction, where the key evidence unit is a labeled assistant turn rather than a complex multi-hop factual graph.

A recommended intermediate form is a JSONL or SQLite memory table:

- one row per historical assistant turn
- optional one row per session summary
- embedding columns or external vector index
- extracted tags for task, constraints, failure modes, and score boundary
- provenance fields for reproducibility

This gives most of the benefit of a memory corpus while keeping experiments simple and auditable.

## Suggested Experimental Roadmap

### R1: Persistent episodic anchor corpus

Build a cached per-user episodic memory file from source sessions. Replace transient `AnchorRetriever(sample.history_sessions)` with a reusable `EpisodicMemoryIndex`.

Compare:

- summary only
- current TF-IDF anchors
- persistent lexical episodic index
- embedding episodic index

Primary metrics:

- MAE / QWK
- user-aware MAE / QWK
- `3/4` F1-DSAT and false-SAT
- latency and prompt token cost

### R2: Boundary-paired retrieval

For each target turn, retrieve:

- top similar examples with score `<=3`
- top similar examples with score `>=4`
- optionally top similar `4` and `5` examples for full-score calibration

Prompt the model to compare current reply against both sides instead of treating anchors as generic examples.

Expected benefit: better DSAT recall without globally shifting predictions toward low scores.

### R3: Retrieval-on-uncertainty

Run first-pass `history_prior_delta_v3_ms` or current best HPD route. Only call retrieval when the first pass is uncertain or internally inconsistent. This controls cost and avoids anchor noise on easy cases.

Expected benefit: improve boundary metrics while preserving strong full-score metrics.

### R4: Memory record augmentation

Use an LLM or rule-based extractor to add compact tags to each historical turn:

- explicit constraints mentioned by the user
- whether the assistant satisfied each constraint
- concreteness / actionability / structure quality
- reason-category-like failure modes
- whether the turn is a clean `3/4` or `4/5` boundary example

Evaluate whether retrieval over these tags beats raw text retrieval. This follows the MemInsight and A-MEM direction.

### R5: Minimal graph memory only if R1-R4 saturate

If structured episodic retrieval still fails on cross-task personalization, add lightweight graph links:

- user requirement -> supporting episodes
- task type -> observed score boundaries
- failure mode -> low-score examples
- contradictory pair -> similar turns with different scores

This should be implemented as an explainable adjacency layer over episodic records, not as a heavy graph database at first.

## Recommended Next Step

The most pragmatic next experiment is:

> Implement `memory_v4_episodic`: keep current `UserMemory` summary, add a persistent per-user episodic memory index, and retrieve boundary-paired labeled examples only when the first-pass predictor is uncertain.

This directly addresses the current weakness: the model has an abstract user rubric but lacks concrete, query-specific evidence for how that user applied the rubric historically.

It is also safer than memory update experiments, because it does not let predicted labels mutate user memory. The memory corpus remains grounded in source-task gold labels, making ablations easier to interpret.

