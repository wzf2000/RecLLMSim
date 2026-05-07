# Reinterpreted Turn Content Prediction Results on U20

## Setup

This report re-analyzes prediction performance using the case-audited
interpretation of the turn-content annotations.

Inputs:

- annotations:
  `detection/outputs/personalized/turn_content_annotations_u20.jsonl`
- output summary:
  `detection/outputs/personalized/turn_content_reinterpreted_compare_u20.json`

Compared methods:

- `history_median`
- `history_mean`
- `nearest3`
- `qwen_v2_5`
- `hpd_v1_cdf`
- `hpd_v2`
- `hpd_v3`
- `hpd_v3_ms`
- `hpd_v3_cdf`
- `hpd_v3_1`

## Updated Group Definitions

The case audit showed that `content_type=other` can mean "substantive but
off-task", so content filtering should use the boolean field rather than
`content_type` alone.

Derived groups:

```text
evaluable_content = has_substantive_content == true
empty_or_process_only = has_substantive_content == false
task_aligned_content = has_substantive_content == true
                       and task_relevance in {medium, high}
off_task_content = has_substantive_content == true
                   and task_relevance in {none, low}
pure_clarification = content_type == clarifying_question
ack_meta = content_type == ack_or_meta
```

Group sizes and gold distributions:

| Group | n | Gold mean | DSAT rate | Score distribution |
|---|---:|---:|---:|---|
| all | `1594` | `4.159` | `0.183` | `{1:38, 2:72, 3:181, 4:611, 5:692}` |
| evaluable_content | `1576` | `4.172` | `0.180` | `{1:32, 2:71, 3:180, 4:604, 5:689}` |
| task_aligned_content | `1562` | `4.173` | `0.179` | `{1:32, 2:69, 3:179, 4:599, 5:683}` |
| off_task_content | `14` | `4.071` | `0.214` | `{2:2, 3:1, 4:5, 5:6}` |
| empty_or_process_only | `18` | `3.000` | `0.444` | `{1:6, 2:1, 3:1, 4:7, 5:3}` |
| pure_clarification | `10` | `3.700` | `0.300` | `{1:1, 3:2, 4:5, 5:2}` |
| ack_meta | `19` | `3.000` | `0.421` | `{1:6, 2:2, 4:8, 5:3}` |

Most annotated turns are evaluable and task-aligned. The off-task and
empty/process groups are too small for stable method ranking, but they reveal
important behavior differences.

## Main Results: Evaluable Content

`evaluable_content` is the best robustness subset for the main benchmark
because it keeps turns where the assistant produced something evaluable, even if
some content is not perfectly task-aligned.

| Method | MAE | QWK | Pearson | Spearman | F1-DSAT | Pred DSAT | False SAT |
|---|---:|---:|---:|---:|---:|---:|---:|
| `history_median` | **`0.5635`** | `0.2741` | `0.3565` | `0.4102` | `0.0000` | `0.0000` | `1.0000` |
| `history_mean` | `0.6060` | `0.2713` | `0.3227` | `0.3532` | `0.1337` | `0.0387` | `0.9187` |
| `nearest3` | `0.6726` | `0.2663` | `0.2746` | `0.3227` | `0.2868` | `0.1478` | `0.7385` |
| `qwen_v2_5` | `0.7005` | `0.2575` | `0.2751` | `0.2702` | `0.3600` | `0.1694` | `0.6502` |
| `hpd_v1_cdf` | `0.6396` | `0.3859` | `0.3866` | `0.4084` | **`0.4343`** | `0.1681` | **`0.5795`** |
| `hpd_v2` | `0.5806` | `0.3155` | `0.3668` | `0.3942` | `0.1966` | `0.0463` | `0.8763` |
| `hpd_v3` | `0.6199` | `0.3831` | `0.4042` | `0.4195` | `0.3788` | `0.1789` | `0.6219` |
| `hpd_v3_ms` | `0.6072` | **`0.3936`** | **`0.4137`** | **`0.4324`** | `0.3674` | `0.1555` | `0.6572` |
| `hpd_v3_cdf` | `0.6580` | `0.3661` | `0.3666` | `0.3881` | `0.4015` | `0.1681` | `0.6113` |
| `hpd_v3_1` | `0.6015` | `0.3588` | `0.3869` | `0.4062` | `0.3368` | `0.1256` | `0.7138` |

Interpretation:

- `history_median` remains the best exact-score MAE baseline, but it misses all
  dissatisfied turns.
- `hpd_v3_ms` is the best full-score/ranking method among model routes on
  evaluable content.
- `hpd_v1_cdf` remains the strongest DSAT-discovery route.
- `hpd_v3` is the best raw balanced route: it matches the gold DSAT rate closely
  and keeps strong QWK/correlation.
- `hpd_v3_1` improves MAE over v3 but suppresses DSAT too much relative to
  v3/v1-CDF.

## Task-Aligned Content

This group removes content that is substantive but off-task or low relevance.
It is almost identical to evaluable content because most turns are task-aligned.

| Method | MAE | QWK | Pearson | Spearman | F1-DSAT | Pred DSAT | False SAT |
|---|---:|---:|---:|---:|---:|---:|---:|
| `history_median` | **`0.5615`** | `0.2781` | `0.3607` | `0.4137` | `0.0000` | `0.0000` | `1.0000` |
| `hpd_v2` | `0.5794` | `0.3197` | `0.3714` | `0.3973` | `0.1983` | `0.0467` | `0.8750` |
| `hpd_v3` | `0.6184` | `0.3864` | `0.4081` | `0.4208` | `0.3799` | `0.1780` | `0.6214` |
| `hpd_v3_ms` | `0.6050` | **`0.3979`** | **`0.4183`** | **`0.4347`** | `0.3716` | `0.1549` | `0.6536` |
| `hpd_v1_cdf` | `0.6357` | `0.3903` | `0.3911` | `0.4100` | **`0.4407`** | `0.1665` | **`0.5750`** |
| `hpd_v3_1` | `0.5992` | `0.3635` | `0.3920` | `0.4093` | `0.3403` | `0.1255` | `0.7107` |

The task-aligned subset confirms the same story as all/evaluable turns. The
method improvements are not caused by off-task side conversations.

## Off-Task but Substantive Content

This group has only `14` turns, so it is not reliable for ranking methods.
However, it helps interpret the annotation labels.

| Method | MAE | QWK | F1-DSAT | Pred DSAT | False SAT | False DSAT |
|---|---:|---:|---:|---:|---:|---:|
| `history_mean` | **`0.7143`** | **`0.2033`** | `0.0000` | `0.0000` | `1.0000` | `0.0000` |
| `nearest3` | **`0.7143`** | `0.0667` | `0.0000` | `0.0000` | `1.0000` | `0.0000` |
| `hpd_v2` | **`0.7143`** | `-0.0476` | `0.0000` | `0.0000` | `1.0000` | `0.0000` |
| `hpd_v3` | `0.7857` | `0.1297` | **`0.2857`** | `0.2857` | `0.6667` | `0.2727` |
| `hpd_v3_ms` | `0.8571` | `0.0508` | `0.0000` | `0.2143` | `1.0000` | `0.2727` |
| `hpd_v3_1` | `0.8571` | `-0.0307` | `0.0000` | `0.1429` | `1.0000` | `0.1818` |

Substantive off-task turns are not automatically low-scored in gold; the gold
mean is `4.071`. This matters for predictor design: an answer can deviate from
the original task taxonomy but still satisfy the user if it responds to the
current local request.

## Empty / Process-Only Replies

This group has only `18` turns, but its behavior is very different:

- gold mean: `3.000`
- gold DSAT rate: `0.444`

| Method | MAE | QWK | F1-DSAT | Pred DSAT | False SAT | False DSAT |
|---|---:|---:|---:|---:|---:|---:|
| `history_median` | `1.7778` | `-0.0612` | `0.0000` | `0.0000` | `1.0000` | `0.0000` |
| `nearest3` | `1.5556` | `-0.0541` | `0.3636` | `0.1667` | `0.7500` | `0.1000` |
| `qwen_v2_5` | `2.0556` | `-0.4918` | `0.3810` | `0.7222` | `0.5000` | `0.9000` |
| `hpd_v2` | `1.5556` | `-0.1613` | `0.2857` | `0.3333` | `0.7500` | `0.4000` |
| `hpd_v3` | `1.6111` | `-0.2745` | `0.4545` | `0.7778` | **`0.3750`** | `0.9000` |
| `hpd_v3_1` | **`1.4444`** | **`-0.1071`** | **`0.4706`** | `0.5000` | `0.5000` | `0.5000` |

Interpretation:

- Empty/process-only turns are genuinely difficult and noisy.
- Aggressive DSAT detectors catch more dissatisfied empty replies but often
  over-penalize satisfied closing/meta turns.
- Because `n=18`, this should be reported qualitatively rather than as a main
  metric table.

## Clarification and Ack/Meta Turns

### Pure clarifications (`n=10`)

| Method | MAE | QWK | F1-DSAT | Pred DSAT | False SAT | False DSAT |
|---|---:|---:|---:|---:|---:|---:|
| `history_median` | `0.7000` | `0.1975` | `0.0000` | `0.0000` | `1.0000` | `0.0000` |
| `hpd_v2` | **`0.6000`** | **`0.5062`** | **`0.5714`** | `0.4000` | `0.3333` | `0.2857` |
| `hpd_v3` | `0.9000` | `0.2857` | `0.5000` | `0.9000` | **`0.0000`** | `0.8571` |
| `hpd_v3_1` | `0.9000` | `0.2857` | `0.5000` | `0.9000` | **`0.0000`** | `0.8571` |

Clarifications are heterogeneous. Some are satisfactory because asking for key
constraints is useful; others are unsatisfactory when the user expected concrete
recommendations. V2 is best on this tiny group because it is less aggressive
than v3 while still detecting some DSAT.

### Ack/meta turns (`n=19`)

| Method | MAE | QWK | F1-DSAT | Pred DSAT | False SAT | False DSAT |
|---|---:|---:|---:|---:|---:|---:|
| `nearest3` | `1.5789` | `-0.0750` | **`0.3636`** | `0.1579` | `0.7500` | `0.0909` |
| `hpd_v2` | `1.6842` | `-0.1282` | `0.0000` | `0.1053` | `1.0000` | `0.1818` |
| `hpd_v3` | `1.7895` | `-0.2424` | `0.3158` | `0.5789` | `0.6250` | `0.7273` |
| `hpd_v3_1` | **`1.5789`** | `-0.0833` | `0.3077` | `0.2632` | `0.7500` | `0.2727` |

Ack/meta turns are especially hard because some are acceptable closing turns
and others are process-only failures. A simple "meta equals dissatisfied" rule
would over-penalize satisfied closings.

## Method-Level Takeaways

### `history_median`

Best or near-best MAE on content-heavy groups, but not a real dissatisfaction
detector:

- evaluable-content MAE: `0.5635`
- evaluable-content F1-DSAT: `0.0000`

Use it as a score-scale baseline, not as a satisfaction boundary predictor.

### `hpd_v2`

Good exact-score compromise but too SAT-conservative:

- evaluable-content MAE: `0.5806`
- evaluable-content F1-DSAT: `0.1966`
- evaluable-content false SAT: `0.8763`

It preserves MAE while missing most dissatisfied cases.

### `hpd_v3`

Best raw balanced semantic route:

- evaluable-content QWK: `0.3831`
- evaluable-content F1-DSAT: `0.3788`
- predicted DSAT rate: `0.1789`, close to gold `0.1796`

It is the cleanest raw method if the goal is balanced 1-5 plus boundary
behavior without post-hoc calibration.

### `hpd_v3_ms`

Best content-aligned ranking / full-score method among HPD routes:

- task-aligned-content QWK: `0.3979`
- task-aligned-content Pearson: `0.4183`
- task-aligned-content Spearman: `0.4347`
- MAE improves over raw v3.

It is the best candidate when the target is robust full 1-5 prediction with
reasonable boundary behavior.

### `hpd_v1_cdf`

Best DSAT discovery:

- task-aligned-content F1-DSAT: `0.4407`
- task-aligned-content false SAT: `0.5750`

But exact MAE is worse than v2/v3-MS. It is useful when the benchmark cares more
about dissatisfied-turn discovery than exact score error.

### `hpd_v3_1`

More conservative than v3:

- evaluable-content MAE: `0.6015`
- evaluable-content F1-DSAT: `0.3368`
- pred DSAT rate: `0.1256`, below gold `0.1796`

It improves exact score relative to v3, but loses DSAT sensitivity.

## Main Conclusions

1. **Content-based filtering does not change the main story.**
   Evaluable and task-aligned content dominate the subset, and method rankings
   are very close to all-turn rankings.

2. **The central tradeoff remains MAE vs DSAT discovery.**
   History and v2-style methods do better on exact score, while v3/CDF-style
   methods do better on dissatisfied-turn detection.

3. **Task-aligned content confirms the predictor is not exploiting off-task or
   meta turns.**
   The gains of `hpd_v3_ms` and `hpd_v1_cdf` persist on task-aligned content.

4. **Empty/process-only turns are difficult but too rare to drive results.**
   They are useful for qualitative error analysis, not for primary metric
   claims.

5. **Best reporting choices:**
   - full-score main: `history_median` as baseline, `hpd_v3_ms` as strongest
     model route
   - boundary main: `hpd_v1_cdf` for DSAT discovery, `hpd_v3` for raw balanced
     behavior
   - robustness table: report all turns and task-aligned content turns side by
     side

## Recommended Paper Table

For the paper, a compact table should compare:

| Method | All MAE | All QWK | All F1-DSAT | Task-aligned MAE | Task-aligned QWK | Task-aligned F1-DSAT |
|---|---:|---:|---:|---:|---:|---:|
| `history_median` | `0.5772` | `0.2548` | `0.0000` | `0.5615` | `0.2781` | `0.0000` |
| `hpd_v2` | `0.5916` | `0.3081` | `0.2000` | `0.5794` | `0.3197` | `0.1983` |
| `hpd_v3` | `0.6311` | **`0.3777`** | `0.3816` | `0.6184` | `0.3864` | `0.3799` |
| `hpd_v3_ms` | `0.6211` | `0.3753` | `0.3603` | `0.6050` | **`0.3979`** | `0.3716` |
| `hpd_v1_cdf` | `0.6531` | `0.3725` | **`0.4298`** | `0.6357` | `0.3903` | **`0.4407`** |

This table directly supports the claim that content characteristics do not
invalidate the main conclusions.
