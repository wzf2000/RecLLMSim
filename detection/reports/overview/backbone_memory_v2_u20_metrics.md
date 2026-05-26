# Backbone comparison on the same 20-user subset

## Setup

- Subset: first 20 test users selected by `limit_users=20`, `user_offset=0`.
- Number of turn-level examples: 1594.
- Prompt/memory setting for memory runs: `memory_version=v2`, `memory_update_mode=none`, `turn_eval_prompt_version=v2`.
- Metrics: Pearson, Spearman, quadratic weighted kappa (QWK), and SAT/DSAT boundary F1-DSAT where DSAT means score `<=3`.

Selected users:

`User_0`, `User_100`, `User_101`, `User_102`, `User_103`, `User_104`, `User_105`, `User_106`, `User_107`, `User_108`, `User_109`, `User_11`, `User_111`, `User_112`, `User_113`, `User_114`, `User_13`, `User_16`, `User_19`, `User_2`.

## Results

| Backbone | Variant | n | Pearson ↑ | Spearman ↑ | QWK ↑ | F1-DSAT ↑ |
|---|---|---:|---:|---:|---:|---:|
| Qwen3-8B | No memory | 1594 | 0.1392 | 0.1069 | 0.0706 | 0.0459 |
| Qwen3-8B | Memory V2 | 1594 | 0.2999 | 0.3012 | 0.2807 | 0.3505 |
| Qwen3-8B | Memory V2 + MS | 1594 | 0.3529 | 0.3737 | 0.3386 | 0.3934 |
| Qwen3-8B | Memory V2 + CDF | 1594 | 0.3384 | 0.3760 | 0.3374 | 0.4050 |
| Qwen3.6-35B-A3B | No memory | - | - | - | - | - |
| Qwen3.6-35B-A3B | Memory V2 | 1594 | 0.3411 | 0.3335 | 0.3083 | 0.4168 |
| Qwen3.6-35B-A3B | Memory V2 + MS | 1594 | 0.3828 | 0.4028 | 0.3758 | 0.4207 |
| Qwen3.6-35B-A3B | Memory V2 + CDF | 1594 | 0.3977 | 0.4238 | 0.3966 | 0.4121 |
| gpt-5.4-mini | No memory | - | - | - | - | - |
| gpt-5.4-mini | Memory V2 | 1594 | 0.3706 | 0.3319 | 0.3393 | 0.3987 |
| gpt-5.4-mini | Memory V2 + MS | 1594 | 0.4093 | 0.4014 | 0.3967 | 0.3993 |
| gpt-5.4-mini | Memory V2 + CDF | 1594 | 0.4212 | 0.4199 | 0.4200 | 0.4156 |

## File Sources

- Qwen3-8B no memory: `detection/outputs/personalized/Qwen_Qwen3-8B_test_no_memory.jsonl`
- Qwen3-8B memory V2: `detection/outputs/personalized/Qwen_Qwen3-8B_test_none.jsonl`
- Qwen3-8B memory V2 + MS: `detection/outputs/personalized/Qwen_Qwen3-8B_test_none_calMS.jsonl`
- Qwen3-8B memory V2 + CDF: `detection/outputs/personalized/Qwen_Qwen3-8B_test_none_calCDF.jsonl`
- Qwen3.6-35B-A3B memory V2: `detection/outputs/personalized/Qwen_Qwen3.6-35B-A3B_test_none_v2_u20.jsonl`
- Qwen3.6-35B-A3B memory V2 + MS: `detection/outputs/personalized/Qwen_Qwen3.6-35B-A3B_test_none_v2_u20_calMS.jsonl`
- Qwen3.6-35B-A3B memory V2 + CDF: `detection/outputs/personalized/Qwen_Qwen3.6-35B-A3B_test_none_v2_u20_calCDF.jsonl`
- gpt-5.4-mini memory V2: `detection/outputs/personalized/gpt-5.4-mini-2026-03-17_test_v2_none_limit20.jsonl`
- gpt-5.4-mini memory V2 + MS: `detection/outputs/personalized/gpt-5.4-mini-2026-03-17_test_v2_none_limit20_calMS.jsonl`
- gpt-5.4-mini memory V2 + CDF: `detection/outputs/personalized/gpt-5.4-mini-2026-03-17_test_v2_none_limit20_calCDF.jsonl`

No 20-user no-memory result file was found for `Qwen3.6-35B-A3B` or `gpt-5.4-mini`.

## Notes

- The gpt-5.4-mini `limit20` MS/CDF files were generated from the completed memory V2 `limit20` output using `detection/eval/calibrate.py`; both calibrated all 80 blocks with no fallback.
- The Qwen3-8B full files were filtered to the same 20-user subset before metric computation.
- On this subset, the best overall row is `gpt-5.4-mini + Memory V2 + CDF` by Pearson and QWK, while `Qwen3.6-35B-A3B + Memory V2 + CDF` is slightly better on Spearman.

