# Generate Podcast

A Codex skill for researched, synthetic single-narrator podcasts. It preserves the canonical script, selected audio takes, source notes, chapter times and verification records.

Rendering uses a complete local Qwen3-TTS 1.7B CustomVoice model through MLX on Apple Silicon macOS. Mastering needs FFmpeg and FFprobe. Recipe validation and transcript generation use Python 3.11 or later without MLX. Raw transcription through a local Whisper model is optional.

## Use the skill

Place this folder under your agent's skills directory as `generate-podcast`, or reference its `SKILL.md` directly. For example:

> Use the generate-podcast skill to create a three-minute English podcast about tree pruning for beginner gardeners, with a calm narrator.

The agent follows [SKILL.md](SKILL.md). It reuses existing task authorization and asks only for setup or preferences still unresolved. Production approval does not claim that anyone listened to the audio.

For standalone execution, follow [setup, preflight and commands](references/commands.md). It includes an isolated environment, the tested model identifier and revision, validation, rendering, targeted retakes, mastering and optional ASR. Dependencies and model downloads require applicable permission; the renderer and transcription helper do not download models.

## Outputs and verification

Each run contains `manifest.json`, `transcript.md`, immutable selected takes and timestamped delivery directories. Each delivery includes an MP3, sample-derived chapters in `assembly.json`, an encoded-audio report and production notes.

Use the **root run manifest** for current delivery, ASR and human review state. Delivery-directory manifests and mastering reports are historical snapshots. Preserve their relative paths when moving or packaging a run.

The checks cover decoding, sample count, sample rate, loudness, true peak and clipping. Optional raw ASR reports preserve recognizer text and mismatch spans. Human listening and pronunciation remain unverified until someone actually reviews the audio. See [evaluation limits](references/evaluation.md).

## Development and release

[Test commands](references/commands.md#test-the-copied-skill) exercise the portable workflow without model downloads. A release smoke test should also render, master, resume and listen to a real short episode; automated checks alone do not establish voice quality.

Vis-specific notes are in [optional integration findings](references/vis-integration.md). No Vis installation is needed for the scripts.

## License status

This repository does not yet declare a license for its own code and instructions. Before releasing it for reuse, the repository owner should choose a license and add its `LICENSE` file. The model card declares Apache-2.0 for the tested Qwen snapshot; that declaration does not license this repository. Model weights are not bundled. Keep applicable model notices and source attribution with distributed artifacts.
