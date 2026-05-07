# Turn Content Annotation Results on U20

## Setup

This report analyzes the completed turn-content annotations on the 20-user
subset.

Annotation file:

- `detection/outputs/personalized/turn_content_annotations_u20.jsonl`

Comparison file:

- `detection/outputs/personalized/turn_content_compare_u20.json`

Annotated scope:

- turns: `1594`
- users: `20`
- user-task blocks: `80`

Compared prediction routes:

- `history_median`: `outputs/personalized/history_baselines/user_history_median_test.jsonl`
- `hpd_v2`: `outputs/personalized/history_prior_delta_v2_none_n3.jsonl`
- `hpd_v3`: `outputs/personalized/history_prior_delta_v3_none_n3_limit20.jsonl`
- `hpd_v3_ms`: `outputs/personalized/history_prior_delta_v3_none_n3_limit20_calMS.jsonl`
- `hpd_v1_cdf`: `outputs/personalized/history_prior_delta_none_n3_limit20_calCDF.jsonl`

## Annotation Distribution

### Boolean content label

| Label | n | Ratio |
|---|---:|---:|
| `has_substantive_content=true` | `1576` | `98.87%` |
| `has_substantive_content=false` | `18` | `1.13%` |

### Content type

| Content type | n | Ratio |
|---|---:|---:|
| `substantive_answer` | `1543` | `96.80%` |
| `mixed_answer` | `13` | `0.82%` |
| `ack_or_meta` | `19` | `1.19%` |
| `clarifying_question` | `10` | `0.63%` |
| `other` | `9` | `0.56%` |

Using the stricter definition
`content_type in {substantive_answer, mixed_answer}`, content-like turns are:

- strict content-like: `1556 / 1594 = 97.62%`
- strict non-content: `38 / 1594 = 2.38%`

So the subset is overwhelmingly made of substantive answer turns. Filtering out
non-content turns will not materially change the benchmark size or main method
ranking.

## Annotation Consistency Note

There are `20` records where `has_substantive_content` and `content_type` are
not perfectly aligned. Examples include:

- `content_type=clarifying_question` but `has_substantive_content=true`
- `content_type=ack_or_meta` but `has_substantive_content=true`
- `content_type=other` but `has_substantive_content=true`

This happens because the annotation prompt allowed the model to decide both a
boolean and a type independently. For future reporting, the safer deterministic
rule is:

```text
strict_content_like = content_type in {substantive_answer, mixed_answer}
```

The existing comparison JSON uses the boolean field for `content_like`, so its
`non_content` group has only `18` turns. Under strict content type, the
non-content group has `38` turns.

## Gold Distribution by Content Type

| Group | n | Gold mean | DSAT rate | Score distribution |
|---|---:|---:|---:|---|
| strict content-like | `1556` | `4.175` | `0.179` | `{1:31, 2:69, 3:178, 4:597, 5:681}` |
| strict non-content | `38` | `3.500` | `0.342` | `{1:7, 2:3, 3:3, 4:14, 5:11}` |
| boolean content-like | `1576` | `4.172` | `0.180` | close to full distribution |
| boolean non-content | `18` | `3.000` | `0.444` | small and unstable |

Non-content turns are clearly harder and more likely to be dissatisfied, but
they are too rare to explain the overall benchmark behavior.

## Turn Position vs Content Type

The previous turn-position analysis suggested that turn 0 is difficult and has
the highest DSAT rate. The content annotation shows that this is not simply
because turn 0 contains many non-substantive setup replies.

| Turn bucket | n | Strict content-like | Boolean content-like | Gold DSAT |
|---:|---:|---:|---:|---:|
| `0` | `346` | `97.7%` | `99.4%` | `27.5%` |
| `1` | `344` | `99.4%` | `100.0%` | `17.7%` |
| `2` | `328` | `98.2%` | `99.4%` | `18.0%` |
| `3` | `284` | `98.2%` | `99.3%` | `11.6%` |
| `4` | `165` | `94.5%` | `97.0%` | `12.7%` |
| `5+` | `127` | `93.7%` | `94.5%` | `17.3%` |

Turn 0 remains mostly substantive. Its high DSAT rate is therefore likely a real
first-response quality issue, not just a clarifying/meta-turn artifact.

## Method Performance on Boolean Content Groups

The comparison file uses `has_substantive_content` for grouping.

### Boolean content-like turns (`n=1576`)

| Method | MAE | QWK | Pearson | Spearman | F1-DSAT | Pred DSAT | False SAT |
|---|---:|---:|---:|---:|---:|---:|---:|
| `history_median` | **`0.5635`** | `0.2741` | `0.3565` | `0.4102` | `0.0000` | `0.0000` | `1.0000` |
| `hpd_v2` | `0.5806` | `0.3155` | `0.3668` | `0.3942` | `0.1966` | `0.0463` | `0.8763` |
| `hpd_v3` | `0.6199` | `0.3831` | `0.4042` | `0.4195` | `0.3788` | `0.1789` | `0.6219` |
| `hpd_v3_ms` | `0.6072` | **`0.3936`** | **`0.4137`** | **`0.4324`** | `0.3674` | `0.1555` | `0.6572` |
| `hpd_v1_cdf` | `0.6396` | `0.3859` | `0.3866` | `0.4084` | **`0.4343`** | `0.1681` | **`0.5795`** |

Interpretation:

- `history_median` remains best for exact MAE, but predicts no DSAT.
- `hpd_v3_ms` is the strongest ranking/full-score route on content-like turns.
- `hpd_v1_cdf` is strongest for DSAT discovery on content-like turns.
- `hpd_v3` remains a balanced raw route: near-matched DSAT rate and good QWK.

### Boolean non-content turns (`n=18`)

| Method | MAE | QWK | F1-DSAT | Pred DSAT | False SAT | False DSAT |
|---|---:|---:|---:|---:|---:|---:|
| `history_median` | `1.7778` | `-0.0612` | `0.0000` | `0.0000` | `1.0000` | `0.0000` |
| `hpd_v2` | **`1.5556`** | `-0.1613` | `0.2857` | `0.3333` | `0.7500` | `0.4000` |
| `hpd_v3` | `1.6111` | `-0.2745` | **`0.4545`** | `0.7778` | **`0.3750`** | `0.9000` |
| `hpd_v3_ms` | `1.8333` | `-0.4444` | `0.1250` | `0.4444` | `0.8750` | `0.7000` |
| `hpd_v1_cdf` | `1.8333` | `-0.3380` | `0.2667` | `0.3889` | `0.7500` | `0.5000` |

The non-content group is too small for stable conclusions. It mainly confirms
that these turns are hard and that aggressive DSAT prediction can easily
overcorrect.

## Strict Content-Type Grouping Check

Using the deterministic strict definition:

```text
content-like = substantive_answer or mixed_answer
```

the conclusions are almost unchanged.

| Method | Strict content MAE | Strict content QWK | Strict content F1-DSAT |
|---|---:|---:|---:|
| `history_median` | **`0.5611`** | `0.2777` | `0.0000` |
| `hpd_v2` | `0.5784` | `0.3172` | `0.1948` |
| `hpd_v3` | `0.6170` | `0.3853` | `0.3775` |
| `hpd_v3_ms` | `0.6028` | **`0.3995`** | `0.3721` |
| `hpd_v1_cdf` | `0.6279` | `0.3945` | **`0.4390`** |

This confirms that the main result is not an artifact of the boolean/type
inconsistency.

## Main Conclusions

1. **The data is mostly substantive-answer turns.**
   Non-content or pure clarification/meta turns are rare (`1-2%` depending on
   grouping). Filtering them out will not materially change the core evaluation.

2. **Early-turn difficulty is real, not just non-content noise.**
   Turn 0 has high DSAT, but `97.7%` of turn-0 replies are still strict
   content-like. The first response is often substantive but unsatisfactory.

3. **History-only baselines are not enough for DSAT.**
   `history_median` is best for MAE but predicts all turns as SAT, so it has
   zero DSAT recall. It is a strong score-scale baseline, not a satisfaction
   boundary detector.

4. **The prior-delta routes improve meaningful content turns too.**
   On strict content-like turns, `hpd_v3_ms` has the best QWK/Pearson/Spearman,
   while `hpd_v1_cdf` has the best F1-DSAT. The gains are not mainly coming from
   rare clarifying/meta turns.

5. **For paper reporting, content filtering should be a robustness analysis,
   not the main benchmark.**
   Because almost all turns are substantive, the main benchmark should keep all
   turns. A content-like subset table can demonstrate that results persist when
   non-content turns are removed.

## Recommendation

For subsequent reports and paper tables:

- Keep **all turns** as the primary evaluation.
- Add a robustness table on **strict content-like turns**.
- Use `content_type` rather than `has_substantive_content` to define content
  groups:

```text
strict_content_like = content_type in {substantive_answer, mixed_answer}
```

- Avoid claiming that low performance is caused by clarification/meta turns;
  the annotation does not support that.
