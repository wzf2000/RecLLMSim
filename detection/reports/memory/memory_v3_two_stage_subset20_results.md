# memory_v3_two_stage: subset-20 results

## Setup

- backbone: `Qwen/Qwen3-8B`
- memory update mode: `none`
- memory version: `v3`
- prompt version: `v3_two_stage`
- result file:
  `detection/outputs/personalized/Qwen_Qwen3-8B_test_none_memv3_v3_two_stage_u20.jsonl`
- comparison baselines on the same 20-user subset:
  - `detection/outputs/personalized/Qwen_Qwen3-8B_test_none.jsonl` filtered to the same records
  - `detection/outputs/personalized/Qwen_Qwen3-8B_test_none_memv3_v3_u20.jsonl`
  - `detection/outputs/personalized/Qwen_Qwen3-8B_test_none_memv3_v3_1_u20.jsonl`
  - `detection/outputs/personalized/Qwen_Qwen3-8B_test_none_boundary_34_selective_refute_v2_reasonfix_u20.jsonl`

## Main results

All numbers below are computed on the same `1594` turns from the 20-user subset.

| method | MAE | Pearson | Spearman | QWK | Boundary Acc | F1-DSAT | False SAT | False DSAT |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `Qwen none` | `0.7077` | `0.2999` | `0.3012` | `0.2807` | `0.7629` | `0.3505` | `0.6495` | `0.1450` |
| `memv3` | `0.6349` | `0.3163` | `0.3176` | `0.2687` | `0.8168` | `0.1705` | `0.8969` | `0.0238` |
| `memv3.1` | `0.6418` | `0.2942` | `0.2997` | `0.2475` | `0.8043` | `0.1789` | `0.8832` | `0.0422` |
| `boundary_v2` | `0.7967` | `0.2215` | `0.2112` | `0.1436` | `0.7302` | `0.3524` | `0.5979` | `0.1965` |
| `memv3_two_stage` | `0.7346` | `0.2477` | `0.2362` | `0.2379` | `0.7679` | `0.3440` | `0.6667` | `0.1351` |

## Interpretation

`memv3_two_stage` is a meaningful correction over `memv3` / `memv3.1`, but it is not yet a new best overall pipeline.

What improved relative to `memv3`:

- DSAT detection recovered strongly:
  - `F1-DSAT: 0.1705 -> 0.3440`
  - `false_sat_rate: 0.8969 -> 0.6667`
- boundary behavior is now close to the original `Qwen none` baseline:
  - `boundary acc: 0.8168 -> 0.7679`
  - `F1-DSAT: 0.1705 -> 0.3440`
- user-aware binary metrics also recovered:
  - `PU-bin F1-DSAT: 0.1425 -> 0.2886`
  - `PU-bin Kappa: 0.0910 -> 0.1567`

What regressed relative to `memv3`:

- full 1-5 regression quality dropped:
  - `MAE: 0.6349 -> 0.7346`
  - `Pearson: 0.3163 -> 0.2477`
  - `QWK: 0.2687 -> 0.2379`

Compared with the same-subset `Qwen none`, the picture is mixed:

- slightly better boundary calibration:
  - `boundary acc: 0.7629 -> 0.7679`
  - `false_dsat_rate: 0.1450 -> 0.1351`
- but not better 1-5 prediction:
  - `MAE: 0.7077 -> 0.7346`
  - `Pearson: 0.2999 -> 0.2477`
  - `QWK: 0.2807 -> 0.2379`

Compared with `boundary_v2`, `memv3_two_stage` is clearly stronger as a fullscale 1-5 system:

- `MAE: 0.7967 -> 0.7346`
- `QWK: 0.1436 -> 0.2379`
- boundary performance stays comparable:
  - `F1-DSAT: 0.3524 -> 0.3440`
  - `boundary kappa: 0.1853 -> 0.2031`

## Two-stage diagnostics

### Branch distribution

- total turns: `1594`
- `sat_45`: `1321`
- `dsat_123`: `273`

Gold boundary distribution on this subset:

- SAT (`>=4`): `1303`
- DSAT (`<=3`): `291`

So the router is already much more balanced than `memv3`, but it still under-routes some DSAT samples into the SAT branch.

### Gate quality

- gate boundary accuracy: `0.7679`
- final boundary accuracy: `0.7679`

This means the refinement stages do not correct cross-boundary mistakes. The final SAT/DSAT split is entirely inherited from the first-stage gate.

### Refinement quality inside the correct branch

Among correctly routed SAT examples (`gold >= 4` and branch = `sat_45`):

- exact `4/5` match: `0.5430` (`612 / 1127`)
- confusion:
  - `4 -> 4`: `275`
  - `4 -> 5`: `223`
  - `5 -> 4`: `292`
  - `5 -> 5`: `337`

This branch is usable, but still compresses many gold `5` into `4`.

Among correctly routed DSAT examples (`gold <= 3` and branch = `dsat_123`):

- exact `1/2/3` match: `0.5773` (`56 / 97`)
- confusion:
  - `1 -> 3`: `19`
  - `2 -> 2`: `1`
  - `2 -> 3`: `20`
  - `3 -> 2`: `2`
  - `3 -> 3`: `55`

This branch is conservative and heavily collapses low scores upward into `3`, but that is still preferable to the previous `memv3` behavior of over-predicting SAT.

## Conclusion

The two-stage design validates the main diagnosis from `memv3`:

- separating calibration from the SAT gate is the right direction
- the first-stage `3/4` gate is the main bottleneck
- the branch refiners are not perfect, but they are not the dominant failure source

Current status:

- `memv3_two_stage` is the best memory-v3-family variant so far for restoring DSAT boundary behavior
- but it still does not beat `Qwen none` on overall `1-5` regression quality
- next optimization should focus on making the first-stage SAT gate more accurate without reintroducing the strong SAT bias of `memv3`
