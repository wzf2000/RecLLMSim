# Appendix 7 Data Protocol Update

## Purpose

This note records the information added to `detection/assets/paper-draft/Appendix/7.tex` for ARR responsible research checklist support.

## Added Content

The appendix now describes:

- Recruitment of 115 native Chinese-speaking participants with prior LLM assistant use experience.
- Public recruitment through posts and social media channels.
- Consent through study registration and online task completion after recruitment materials explained research and analysis use.
- Compensation based on completed conversation quantity and quality, averaging about 8--10 RMB per conversation.
- Structured profile fields, including gender, age, occupation, personality, daily interests, travel habits, dining preferences, spending habits, and other preference tags.
- Four planning-oriented task families: recipe planning, gift preparation, travel planning, and skill learning planning.
- Up to 19 tasks per participant across four scenarios, with roughly four to five tasks per scenario on average.
- Instructions to follow real needs and preferences, add additional requirements when appropriate, ask follow-up questions, and continue until the plan or recommendation was realistically usable.
- Turn-level satisfaction annotation with 1--5 anchors and categorical dissatisfaction reasons for scores at most 3.
- Quality handling through compensation adjustment for very short or low-participation conversations rather than direct filtering.
- Separation of payment identifiers from research data.

## Privacy Check

An automatic pattern scan was run over `data/**/*.json` for common phone-number, bank-card-like, identity-card-like, and email patterns.
The matches inspected were public or generated contact information inside assistant responses, such as hotel, restaurant, or organization contact details, rather than participant identity or payment information.
The appendix phrases this as a risk-reduction check rather than a guarantee that no personal information appears in free-form conversation text.
