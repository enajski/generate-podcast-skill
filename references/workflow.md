# Editorial workflow

## Research and claims

Start with a brief. Record intended audience, question, length, language and allowed source types.

For each source, record its title, URL, access level and relevant claim. Separate the author's evidence from your interpretation.

Name uncertainty in the narration when it changes the conclusion. An organizer account is not a representative survey.

Do not invent budgets, source independence, field recordings or research participants. Mark a fictional example as fictional.

## Canonical text and pronunciation

Canonical paragraphs are the approved words. Each section has a stable, unique identifier.

The scripts group complete paragraphs within the word limit. An oversized paragraph fails validation. Split it deliberately before rendering.

Pronunciation entries match exact, case-sensitive whole terms. Use separate entries for different spellings or plurals. Replacements do not chain.

A pronunciation change can affect chunk size. The scripts reject synthesis text that exceeds `max_words`. Shorten the replacement or split source paragraphs.

Use a new short recipe for each audition candidate. Preserve audio and recipe hashes when the user makes a selection.

## Review evidence

Record who reviewed what, when, and with which result. Use the `review` command only to record an actual human statement.

Carry forward existing authorization for production and setup. Script or voice-direction approval is not audition listening approval. A routine test with disclosed defaults does not require a new script approval; ask when an unresolved preference or material change affects the requested outcome.

A selected audition supports a pronunciation recipe. It does not prove that the whole episode has correct pronunciation.

Do not mark whole-episode listening as complete unless someone listened to the delivered encoded file.

Check numbers, currencies, negation and evidence limits separately from transcript agreement. Keep unresolved disagreements visible.

## Revision and recovery

Use the same output directory for resume and targeted retakes. The manifest identifies the current selected takes.

Rendering saves checkpoints after each successful take. Previous manifests and audio remain available. Failed work cannot become a selected completed take.

A changed recipe invalidates incompatible selections. Unchanged chunks retain their audio hashes.

A killed process can leave a lock. Confirm that its owner has stopped before removing that lock. Do not bypass a live lock.

Run mastering again after changes. Keep each previous delivery directory. Use the root run manifest's `delivery` object to identify the current output. Its `verification`, `asr_reports` and `reviews` contain later updates; the manifest and report inside each delivery directory preserve the state at mastering.

## Licensing and permissions

Verify model and voice licenses before distribution. Record the model identifier and exact revision in production notes.

The tested Qwen model family uses Apache-2.0. This statement does not cover arbitrary replacement models, inputs or voice clones.

Keep source attribution and required model notices. Do not assume all voices in another backend share one license.

Downloads, package installation, external publishing and service changes need the user's applicable permission. Honor permission already granted for the same scope. A skill never grants that permission.
