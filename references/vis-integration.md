# Vis integration findings

Research checked Vis revision `ba9cf9a5d4adc8beb6dc7d72e1f9ce96a2e811c9` on 2026-10-05. Recheck these contracts before implementing an integration.

## Recommendation

In Vis, attach completed files with the available sandbox attachment tool and reuse filenames for revisions. Do not change regular Vis TTS or install an extension as part of episode production.

Keep the skill and portable renderer independent of Vis. A future extension should wrap the same scripts or library.

Use native changes for shared speech capabilities, model readiness, resource limits and client controls. Do not move the editorial workflow into Bridge.

## Verified boundaries

- Vis discovers standard agent skills, including bundled scripts and references.
- Extensions can declare tools, slash commands, settings and activity presentation.
- The inspected SDK has no declaration for a speech engine.
- The native speech registry contains Piper and Pocket, not Qwen or Kokoro.
- Current synthesis progress events are not playable-audio streaming.
- Speech jobs are process-local, with limited retention. They are not durable episode checkpoints.
- The sandbox attachment tool stores durable files. No direct public attachment method was found in the extension SDK.
- Bridge coordinates project-owned verification. It is not an audio scheduler.

A first extension can return validated file paths and a manifest. The agent can attach them. Do not invent `vis.attach` or `vis.tools.attach`.

## Voice evidence

On one 198-word passage, Qwen Aiden took 31.65 seconds to produce 80.96 seconds of audio. Piper Kristin took 2.28 seconds for 70.97 seconds.

These are local batch timings on an M1 Max with 64 GiB memory. They are not gateway latency or first-audio measurements.

Raw Parakeet agreement was 9.80/10 for Qwen and 9.70/10 for Piper. Independent Whisper agreement was 9.80/10 for both.

These scores do not establish a naturalness winner. The user selected one pronunciation sample, not a universal voice ranking.

Keep Piper as the regular default until an end-to-end interactive comparison justifies a change. Qwen is an optional narration candidate.

## Source locations in Vis

- `src/com/blockether/vis/internal/speech/synthesis.clj`: engine descriptors.
- `src/com/blockether/vis/internal/speech/tts.clj`: model cache, native generation and controls.
- `src/com/blockether/vis/internal/gateway/server/speech.clj`: inline and job responses.
- `src/com/blockether/vis/internal/speech/core.clj`: job lifecycle.
- `src/com/blockether/vis/internal/speech/engine.clj`: ASR cleanup.
- `packages/vis-agent/src/blockether/vis/extension.py`: extension declarations.
- `resources/vis-docs/skills.md`: discovery and skill format.
- `resources/vis-docs/extension-packages.md`: managed package and bundled-skill contracts.

## Before a future integration

Test registration, progress, cancellation, recovery, output handoff and resource limits. Verify native-image behavior when JNI or interop changes.

Measure first playable audio at the client. Compare short acknowledgements, readbacks and paragraphs. Separate downloads, cold starts, warm runs and voice changes.

Use human listening for timbre, intelligibility, stress and preference. Keep factual correctness separate from aggregate ASR agreement.
