# ARR Submission Checklist Gap Analysis

## Scope

This note checks `detection/assets/paper-draft/requirements.md` against the current paper draft under `detection/assets/paper-draft/`.
It records which ARR responsible research checklist items are already supported by the draft and which items need additional text before submission.

## Current Status

The draft already contains a dedicated `Limitations` section in `detection/assets/paper-draft/Tex/6.tex`.
The draft also reports main data statistics in `detection/assets/paper-draft/Tex/4.tex`, the replay benchmark setup in `detection/assets/paper-draft/Tex/5.tex`, detailed metrics in `detection/assets/paper-draft/Appendix/1.tex`, baseline details in `detection/assets/paper-draft/Appendix/2.tex`, and the main data collection protocol in `detection/assets/paper-draft/Appendix/7.tex`.

The largest missing item is the currently empty `Ethical Considerations` section in `detection/assets/paper-draft/Tex/6.tex`.
This section should cover potential risks, privacy/anonymization, participant consent/recruitment/payment, ethics review status, and AI-assistant use if applicable.

## Recommended Additions

1. Add a compact but complete `Ethical Considerations` section in `detection/assets/paper-draft/Tex/6.tex`.
2. Expand `detection/assets/paper-draft/Appendix/7.tex` with concrete participant-facing collection details, including instructions, consent, recruitment/payment, and anonymization.
3. Add a small reproducibility or implementation paragraph in `detection/assets/paper-draft/Appendix/2.tex` if the submission checklist answer for experimental setup needs a precise section reference for hyperparameters and decoding settings.
4. Add an AI-assistant use statement either in `Tex/6.tex` or directly in the ARR checklist response.

## Items That Need Author Confirmation

- Whether participants signed or otherwise gave informed consent.
- Whether an IRB/ethics review board approved the protocol, determined it exempt, or was not applicable under local policy.
- Recruitment source and compensation amount or policy.
- Whether any personally identifying information was collected and how it was removed or protected.
- Whether AI assistants were used in research, coding, writing, or editing.
