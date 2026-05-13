# Static Replay Dialogue Memory Results

## Setup

This report summarizes the main-dataset static replay runs with the new
dialogue-memory candidate context modes.

Dataset and selection:

- Dataset: personalized main benchmark.
- Selection: `selection_mode=hard`.
- Samples: `300` replay turns.
- Users: `90`.
- User-task blocks: `180`.
- Gold score distribution of replayed original turns:
  - `1`: `28`
  - `2`: `53`
  - `3`: `133`
  - `4`: `86`

Candidate models:

- `Qwen/Qwen3-8B`
- `deepseek-v4-flash`
- `gemini-3.1-flash-lite-preview`

Replay context modes:

- `raw`
- `dialogue_memory_tfidf`
- `dialogue_memory_diverse`

Scoring:

- Judge: `Qwen/Qwen3-8B`
- Judge prompt: `v2`
- Judge memory: `v2`
- Post-hoc calibration: `reference_cdf`
- Reference: `outputs/personalized/Qwen_Qwen3-8B_test_none.jsonl`

All nine runs contain `300 / 300` scored and refCDF-calibrated records.

## RefCDF Benchmark Scores

| Candidate | Context | Micro Mean | User Macro | Block Macro | SAT Rate | DSAT Rate |
|---|---|---:|---:|---:|---:|---:|
| `Qwen/Qwen3-8B` | `raw` | `4.3333` | `4.4647` | `4.4000` | `0.9200` | `0.0800` |
| `Qwen/Qwen3-8B` | `dialogue_memory_tfidf` | `4.2733` | `4.3711` | `4.3269` | `0.9033` | `0.0967` |
| `Qwen/Qwen3-8B` | `dialogue_memory_diverse` | `4.2600` | `4.3833` | `4.3019` | `0.8967` | `0.1033` |
| `deepseek-v4-flash` | `raw` | `4.3200` | `4.4207` | `4.3630` | `0.8933` | `0.1067` |
| `deepseek-v4-flash` | `dialogue_memory_tfidf` | `4.2500` | `4.4150` | `4.3130` | `0.8833` | `0.1167` |
| `deepseek-v4-flash` | `dialogue_memory_diverse` | `4.2900` | `4.4138` | `4.3454` | `0.9000` | `0.1000` |
| `gemini-3.1-flash-lite-preview` | `raw` | `4.2033` | `4.3392` | `4.2602` | `0.8700` | `0.1300` |
| `gemini-3.1-flash-lite-preview` | `dialogue_memory_tfidf` | `4.1767` | `4.2904` | `4.2167` | `0.8667` | `0.1333` |
| `gemini-3.1-flash-lite-preview` | `dialogue_memory_diverse` | `4.1667` | `4.3099` | `4.2250` | `0.8700` | `0.1300` |

## Paired Comparison Against Raw

The table below compares each memory mode to the same model's `raw` run on the
same `300` replay turns. A win means the memory-mode refCDF score is higher
than the raw score for that replay turn.

| Candidate | Memory Context | Win | Tie | Lose | Win Rate | Tie Rate | Lose Rate | Mean Delta |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `Qwen/Qwen3-8B` | `dialogue_memory_tfidf` | `18` | `253` | `29` | `0.060` | `0.843` | `0.097` | `-0.0600` |
| `Qwen/Qwen3-8B` | `dialogue_memory_diverse` | `20` | `247` | `33` | `0.067` | `0.823` | `0.110` | `-0.0733` |
| `deepseek-v4-flash` | `dialogue_memory_tfidf` | `20` | `240` | `40` | `0.067` | `0.800` | `0.133` | `-0.0700` |
| `deepseek-v4-flash` | `dialogue_memory_diverse` | `26` | `234` | `40` | `0.087` | `0.780` | `0.133` | `-0.0300` |
| `gemini-3.1-flash-lite-preview` | `dialogue_memory_tfidf` | `24` | `243` | `33` | `0.080` | `0.810` | `0.110` | `-0.0267` |
| `gemini-3.1-flash-lite-preview` | `dialogue_memory_diverse` | `29` | `238` | `33` | `0.097` | `0.793` | `0.110` | `-0.0367` |

## Interpretation

### 1. Dialogue memory does not improve the current benchmark score

Across all three candidate models, both dialogue-memory modes are below the
corresponding raw run on micro mean and block macro mean.

The drop is small but consistent:

- `Qwen/Qwen3-8B`: `-0.0600` to `-0.0733` mean paired delta.
- `deepseek-v4-flash`: `-0.0700` for TF-IDF, `-0.0300` for diverse.
- `gemini-3.1-flash-lite-preview`: `-0.0267` to `-0.0367`.

This suggests that simply injecting retrieved cross-scenario dialogue memories
does not make candidate responses more satisfying under the current judge.

### 2. Most examples are unchanged after calibration

Tie rates are high:

- `0.780` to `0.843` across all memory-vs-raw comparisons.

The memory prompt changes some generations, but after judge scoring and
refCDF calibration, most replay turns land on the same integer score as raw.
The changed cases are more often losses than wins.

### 3. Memory context shifts models slightly toward lower satisfaction

Memory modes usually reduce:

- micro mean,
- user macro mean,
- block macro mean,
- and often SAT rate.

For example, `Qwen/Qwen3-8B` drops from SAT rate `0.9200` in raw to `0.9033`
with TF-IDF and `0.8967` with diverse memory. This pattern indicates that the
extra memory context may make responses more constrained, cautious, or less
directly optimized for the current user request.

### 4. Diverse retrieval is slightly better than TF-IDF for DeepSeek, but not enough

For `deepseek-v4-flash`, `dialogue_memory_diverse` is closer to raw than
`dialogue_memory_tfidf`:

- raw micro mean: `4.3200`
- TF-IDF: `4.2500`
- diverse: `4.2900`

However, diverse memory still loses to raw in paired comparison:

- win/tie/lose: `26 / 234 / 40`
- mean delta: `-0.0300`

So source-task diversity may reduce the harm of memory retrieval, but it does
not yet turn memory augmentation into a gain.

## Current Conclusion

Under the current implementation, dialogue-memory candidate context is not a
positive benchmark condition. The strongest setting remains plain `raw` replay
for all three tested candidate models.

The result is still useful: it suggests that adding user history to candidate
generation is not automatically beneficial, at least when the memory is a
retrieved raw-dialogue block prepended to the prompt.

## Suggested Next Steps

1. Keep `raw` as the main static replay benchmark setting.
2. Treat `dialogue_memory_tfidf` and `dialogue_memory_diverse` as diagnostic
   variants rather than primary benchmark results.
3. If continuing this direction, try a more compressed memory instruction:
   - fewer retrieved examples, e.g. `top_k=2`;
   - shorter assistant reply snippets;
   - a stronger instruction to prioritize the current user request over memory.
4. Analyze changed cases where memory wins versus loses, to see whether memory
   helps specific tasks or user types.
5. If memory-augmented candidate generation is included in the paper, frame it
   as an ablation showing that naive raw-dialogue memory is not enough.
