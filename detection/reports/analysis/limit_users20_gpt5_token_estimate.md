# limit_users=20 GPT-5 tokenizer token estimate

## Setup

- Task: estimate predictor trace size for `limit_users=20`.
- Dataset split: `test`, `train_ratio=0.2`, `split_seed=42`.
- Model setting: `Qwen/Qwen3-8B`, memory enabled, `memory_version=v2`, `memory_update_mode=none`.
- Turn prompt: `turn_eval_prompt_version=v2`, `history_window_size=5`, `n_anchors=0`.
- Python environment: `chat`.
- Tokenizer: local `o200k_base` rank table from `/data/wangzhefan/.vscode-server/extensions/github.copilot-1.372.0/dist/resources/o200k_base.tiktoken.noindex`, used as GPT-5/4o-family tokenizer base.
- Chat overhead: approximate `+7` input tokens per two-message call, following the common OpenAI chat wrapper estimate.

## Data Size

Full test split:

- Users: 90
- User-task blocks: 356
- Target sessions: 1470
- Target assistant turns: 6474

`limit_users=20` selects the first 20 users in sample order:

`User_0`, `User_100`, `User_101`, `User_102`, `User_103`, `User_104`, `User_105`, `User_106`, `User_107`, `User_108`, `User_109`, `User_11`, `User_111`, `User_112`, `User_113`, `User_114`, `User_13`, `User_16`, `User_19`, `User_2`.

Subset size:

- Users: 20
- User-task blocks: 80
- Target sessions: 346
- Target assistant turns: 1594
- Avg history sessions per block: 12.975
- Min / max history sessions per block: 9 / 15

## Call Count

Fresh memory run:

- Memory build calls: 80
- Turn prediction calls: 1594
- Total calls: 1674

Cached memory run:

- Turn prediction calls only: 1594

Because `memory_update_mode=none`, there are no post-session memory update calls.

## Token Estimate

Input token counts include system message, user prompt, and approximate chat wrapper overhead.

| Component | Calls | Total tokens | Mean | P50 | P90 | Min | Max |
|---|---:|---:|---:|---:|---:|---:|---:|
| Memory build input | 80 | 616,698 | 7,708.7 | 7,772 | 8,921 | 5,410 | 11,854 |
| Memory output cached JSON | 80 | 39,609 | 495.1 | 479 | 592 | 369 | 789 |
| Turn prediction input | 1594 | 5,553,760 | 3,484.2 | 3,047 | 5,664 | 1,003 | 16,846 |
| Turn prediction output | 1594 | 225,905 | 141.7 | 136 | 188 | 63 | 320 |

Totals:

- Fresh memory input tokens: 6,170,458
- Fresh memory output tokens: 265,514
- Fresh memory total tokens: 6,435,972
- Cached memory input tokens: 5,553,760
- Cached memory output tokens: 225,905
- Cached memory total tokens: 5,779,665

The turn output estimate was computed from existing `detection/outputs/personalized/Qwen_Qwen3-8B_test_none.jsonl` predictions for the same 1594 `sample_id`s, serialized as compact structured JSON with `classification`, `reason`, and `analysis`.

