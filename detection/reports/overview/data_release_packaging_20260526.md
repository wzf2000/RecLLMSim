# PersTurnBench Data Release Packaging

## Artifact

- Release directory: `detection/assets/releases/persturnbench_data_release_20260526/`
- Compressed archive: `detection/assets/releases/persturnbench_data_release_20260526.tgz`
- SHA256: `12f02837b8f2d516efdcc66d1fe0862a3cfd4af1ce065486f304730dd915f7bb`
- Archive size: 9.2 MB

## Included Content

The archive contains the sanitized main PersTurnBench conversation data used for personalized turn-level user conversation satisfaction evaluation.

Included files:

- `data/User_*/<scenario>/*.json`: anonymized per-user, per-scenario conversation sessions.
- `README.md`: release overview and privacy notes.
- `DATA_SCHEMA.md`: JSON schema description.
- `statistics.json`: aggregate statistics.
- `splits.json`: main personalized evaluator split metadata.
- `MANIFEST.txt`: file list.

## Sanitization

The packaged files are generated from the repository `data/` directory.

The release keeps:

- anonymous user IDs such as `User_26`;
- task context and task type;
- structured user profile fields;
- conversation turns;
- assistant-turn satisfaction scores and low-score dissatisfaction reasons.

The release removes:

- all internal `strategy_*` fields;
- `profile.complete_tid`;
- assistant-turn `hallucination` fields;
- top-level `questionnaire` fields;
- any payment/contact information, which is not present in the paper-facing data files.

A lightweight scan found no `strategy_*`, `complete_tid`, `hallucination`, or `questionnaire` keys in the release.
It also found no phone-number-like 11-digit strings.
Long digit strings in the release are URL or product identifiers in assistant-generated content.

## Dataset Statistics

- Users: 115
- Conversations: 1833
- User turns: 8204
- Assistant turns with satisfaction labels: 8060
- Scenarios:
  - `旅行规划`: 502
  - `礼物准备`: 531
  - `菜谱规划`: 402
  - `技能学习规划`: 398
- Satisfaction distribution:
  - 1: 139
  - 2: 279
  - 3: 843
  - 4: 3102
  - 5: 3697

## Split Metadata

The `splits.json` file records the main personalized evaluator user split.

The packaged split was verified in the chat environment with the released loader and `sklearn`.
It is produced by `GroupShuffleSplit` over usable user-scenario blocks with `train_ratio=0.2`, `test_ratio=0.8`, and `seed=42`.
The full release contains 115 anonymized users, while the main personalized evaluator split uses the 112 users that have at least one usable target block and at least one cross-scenario history session.

Resulting split:

- Usable users: 112
- Train users: 22
- Test users: 90
- Train user-scenario blocks: 85
- Test user-scenario blocks: 356
- Train target turns: 1413
- Test target turns: 6474
