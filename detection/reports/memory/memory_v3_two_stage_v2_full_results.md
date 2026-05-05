# memory_v3_two_stage_v2: full results

## Setup

- backbone: `Qwen/Qwen3-8B`
- memory update mode: `none`
- memory version: `v3`
- prompt version: `v3_two_stage_v2`
- result file:
  `detection/outputs/personalized/Qwen_Qwen3-8B_test_none_memv3_v3_two_stage_v2.jsonl`
- comparison file:
  `detection/outputs/personalized/memv3_two_stage_v2_full_comparison.json`

Direct full-data baselines used here:

- `detection/outputs/personalized/Qwen_Qwen3-8B_test_none.jsonl`
- `detection/outputs/personalized/Qwen_Qwen3-8B_test_none_boundary_34_selective_refute_v2_fullscale.jsonl`

Note:

- there is currently no full-data run for `memv3_v3` or the original `memv3_two_stage`, so the strongest apples-to-apples full comparison is against `Qwen none` and `boundary_34_selective_refute_v2_fullscale`

## Main metrics

| method | MAE | Pearson | Spearman | QWK | Boundary Acc | F1-DSAT | False SAT | False DSAT |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `Qwen none` | `0.7110` | `0.2967` | `0.2820` | `0.2815` | `0.7555` | `0.3312` | `0.6459` | `0.1617` |
| `boundary_fullscale` | `0.8551` | `0.1916` | `0.1717` | `0.1797` | `0.7076` | `0.3309` | `0.5772` | `0.2337` |
| `memv3_two_stage_v2` | `0.7337` | `0.2212` | `0.1983` | `0.2055` | `0.7643` | `0.3064` | `0.6956` | `0.1409` |

## High-level conclusion

`memv3_two_stage_v2` is a meaningful improvement over the earlier boundary-driven fullscale pipeline, but it still does not beat raw `Qwen none`.

Its overall position is:

- clearly better than `boundary_fullscale` as a full `1-5` pipeline
- still worse than `Qwen none` on all main `1-5` regression metrics
- slightly better than `Qwen none` on boundary accuracy and false-DSAT control
- but weaker than `Qwen none` on DSAT recall and `F1-DSAT`

So this version is better interpreted as:

- a more balanced fullscale system than `boundary_fullscale`
- but not yet a new best overall Qwen pipeline

## Detailed comparison

### 1. Compared with `boundary_fullscale`

`memv3_two_stage_v2` is a substantial upgrade:

- `MAE: 0.8551 -> 0.7337`
- `Pearson: 0.1916 -> 0.2212`
- `QWK: 0.1797 -> 0.2055`
- `boundary acc: 0.7076 -> 0.7643`
- `false_dsat_rate: 0.2337 -> 0.1409`

This means the new `memory_v3 + selective gate + branch refine` structure is much healthier than the earlier boundary-v2 fullscale route.

### 2. Compared with `Qwen none`

`memv3_two_stage_v2` still loses on the main 1-5 task:

- `MAE: 0.7110 -> 0.7337`
- `Pearson: 0.2967 -> 0.2212`
- `Spearman: 0.2820 -> 0.1983`
- `QWK: 0.2815 -> 0.2055`

Boundary behavior is mixed:

- slightly better:
  - `boundary acc: 0.7555 -> 0.7643`
  - `false_dsat_rate: 0.1617 -> 0.1409`
- worse:
  - `F1-DSAT: 0.3312 -> 0.3064`
  - `false_sat_rate: 0.6459 -> 0.6956`

This is the clearest signal of the current tradeoff:

- the new gate reduces unnecessary SAT-to-DSAT mistakes
- but it still lets too many true DSAT cases slip into SAT

## User-aware metrics

### Full 1-5 user-aware metrics

| method | PU-Pearson | PU-Spearman | PU-Kappa | WC-Pearson | WC-Spearman |
|---|---:|---:|---:|---:|---:|
| `Qwen none` | `0.1997` | `0.1798` | `0.1646` | `0.2154` | `0.1637` |
| `boundary_fullscale` | `0.1683` | `0.1568` | `0.1390` | `0.1750` | `0.1468` |
| `memv3_two_stage_v2` | `0.1769` | `0.1700` | `0.1401` | `0.1957` | `0.1581` |

Interpretation:

- `memv3_two_stage_v2` recovers a meaningful amount of user-aware signal relative to `boundary_fullscale`
- but it still does not reach `Qwen none`

### Boundary user-aware metrics

| method | PU-bin F1-DSAT | PU-bin Kappa | WC-bin Pearson |
|---|---:|---:|---:|
| `Qwen none` | `0.2734` | `0.1196` | `0.1529` |
| `boundary_fullscale` | `0.2912` | `0.1266` | `0.1434` |
| `memv3_two_stage_v2` | `0.2600` | `0.1267` | `0.1512` |

Interpretation:

- `memv3_two_stage_v2` no longer shows the strong DSAT bias of `boundary_fullscale`
- but that bias reduction also removes some DSAT sensitivity
- user-aware boundary performance ends up in the middle: better calibrated than `boundary_fullscale`, but not stronger than `Qwen none`

## Prediction distribution

Gold distribution:

- `1: 123`
- `2: 251`
- `3: 733`
- `4: 2449`
- `5: 2918`

`memv3_two_stage_v2` prediction distribution:

- `1: 3`
- `2: 25`
- `3: 1065`
- `4: 3803`
- `5: 1578`

Key pattern:

- the system no longer collapses everything into SAT
- but it still strongly compresses:
  - low scores upward into `3`
  - many gold `5` downward into `4`

This is already more usable than the earlier boundary fullscale system, but it still lacks enough score spread.

## Gate and branch diagnostics

### Gate statistics

- total turns: `6474`
- `two_stage_gate_score=4`: `5381`
- `two_stage_gate_score=3`: `1093`
- `two_stage_gate_model_flag=true`: `513`
- `two_stage_gate_triggered=true`: `407`

Boundary accuracy:

- gate boundary accuracy: `0.7643`
- final boundary accuracy: `0.7643`

This means the second-stage branch refiners do not correct cross-boundary mistakes. Final SAT/DSAT behavior is fully inherited from the gate.

### Branch distribution

- `sat_45`: `5381`
- `dsat_123`: `1093`

This is close to the gold SAT/DSAT proportion, which is one reason the overall boundary accuracy improves over `Qwen none`.

### Refinement quality inside correctly routed examples

For correctly routed SAT examples (`gold >= 4` and branch = `sat_45`):

- exact `4/5` match: `0.5155` (`2377 / 4611`)
- confusion:
  - `4 -> 4`: `1488`
  - `4 -> 5`: `536`
  - `5 -> 4`: `1698`
  - `5 -> 5`: `889`

This branch still heavily compresses gold `5` into `4`.

For correctly routed DSAT examples (`gold <= 3` and branch = `dsat_123`):

- exact `1/2/3` match: `0.5312` (`179 / 337`)
- confusion:
  - `1 -> 1`: `1`
  - `1 -> 2`: `8`
  - `1 -> 3`: `57`
  - `2 -> 1`: `1`
  - `2 -> 3`: `90`
  - `3 -> 2`: `2`
  - `3 -> 3`: `178`

This branch remains conservative and mostly collapses severe DSAT into `3`.

## Reason consistency

The score/reason legality rule remains fully satisfied:

- `pred_score >= 4` with non-`满意` reason: `0`
- `pred_score <= 3` with `满意` reason: `0`

## Final takeaways

`memory_v3_two_stage_v2` answers one important question positively:

- using a stronger selective `3/4` gate on top of `memory_v3` does improve the fullscale pipeline compared with the older boundary-based fullscale route

But it also confirms the next bottleneck:

- the gate is now reasonably balanced, but still not accurate enough to beat `Qwen none`
- once the gate makes a boundary mistake, the branch refiners do not recover it
- inside the branches, score compression is still strong:
  - `5 -> 4`
  - `1/2 -> 3`

So the next high-ROI direction is not a full redesign. It is:

1. improve first-stage gate precision, especially DSAT recall without pushing the system back into DSAT bias
2. reduce branch compression:
   - SAT branch: recover more `5`
   - DSAT branch: recover more `2` and `1`
