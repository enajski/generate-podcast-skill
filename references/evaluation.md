# Evaluation limits

## Separate the checks

Report these independently:

- File integrity and complete decoding.
- Encoded loudness, true peak, clipping, sample rate and duration.
- Transcript agreement from each raw recognizer.
- Critical facts, including amounts, currencies, dates and negation.
- Human pronunciation and voice preference.
- Whole-episode listening status.

ASR agreement is not audio fidelity or naturalness. Two recognizers can agree on the same error. They can also mishear a correctly spoken name.

Never use filler removal or repeated-word cleanup for fidelity evaluation. Keep raw recognizer text and mismatch spans.

The bundled comparison normalizes Unicode, case and whitespace. It ignores most punctuation while preserving numbers, currency symbols, decimal separators and signs, including hyphens. ASR spelling such as `three-cut` versus canonical `three cut` can therefore count an inserted hyphen despite identical words. Inspect the retained spans before interpreting an aggregate error rate; do not clean the raw transcript to improve the score.

It does not silently equate proper names, currency amounts or year verbalizations. This conservative policy can create false mismatches for manual review.

If you add equivalence rules, document and test them symmetrically. Do not forgive a different amount, magnitude, sign or currency.

## Timing and acoustic measures

Words per minute describe pace. Pause duration and pitch spread describe acoustic behavior. None is a naturalness score.

Record trimming and inserted gaps before interpreting pace. Do not treat an intentional section break as a synthesis stall.

Use matched-loudness excerpts for human comparison. Keep the same passage and disclose listening conditions.

## Review and repair

Sample approval applies to that sample, not every later occurrence or the full episode. Keep accepted pronunciation cues separate from canonical text. Repair affected segments with targeted retakes and retain unchanged audio hashes.

Use the root run manifest to assess current ASR and review status. Delivery reports describe checks at mastering and are not rewritten after later reviews. A passing acoustic report does not imply a human-approved episode.
