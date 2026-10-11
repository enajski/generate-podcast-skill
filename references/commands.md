# Setup and commands

## Requirements

The portable scripts use Python 3.11 or later. Rendering requires Apple Silicon macOS, GPU access, compatible MLX dependencies and a complete local Qwen3-TTS 1.7B CustomVoice model. Mastering requires FFmpeg and FFprobe. Validation and transcript generation use the standard library.

The successful smoke test used Python 3.11, `mlx-audio==0.5.7`, MLX 0.32.3, NumPy 2.4.6 and Transformers 5.18.0. Other rendering environments are not established by the standard-library Python requirement. `constraints-render-tested.txt` records these transitive versions; it is not a guarantee for every hardware or macOS version.

The renderer and optional transcription helper do not download models. Prepare dependencies and model files only with applicable setup permission; honor permission already granted.

## Standalone quick start

Start in the skill checkout. Reuse a compatible environment supplied by the user, or create an isolated one:

```sh
SKILL="$(pwd)"
PRODUCTION="$SKILL/tmp/quickstart"
mkdir -p "$PRODUCTION"
python3.11 -m venv "$PRODUCTION/.venv"
PYTHON="$PRODUCTION/.venv/bin/python"
"$PYTHON" -m pip install -r "$SKILL/requirements-render.txt" \
  -c "$SKILL/constraints-render-tested.txt"
cp "$SKILL/examples/episode.json" "$PRODUCTION/episode.json"
RUN="$PRODUCTION/episode-run"
```

An existing `uv` installation can use `uv venv` and `uv pip install --python "$PYTHON"` instead. Use `--offline` when all packages are cached; do not assume a cache is complete.

The copied example is fictional. Replace its text, sources and voice direction with the requested material. Keep inputs and `.venv` outside `$RUN`. On its first render, `$RUN` must be absent or empty; later renders reuse its manifest.

The tested model is `mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit`, revision `41d3337e8b7f2843a75841595fc14e4b9a7a4b96`. Set `$MODEL` to a complete local snapshot directory and `$REVISION` to its actual source revision. A snapshot needs text-tokenizer and speech-tokenizer assets as well as weights. If that snapshot is missing and download is authorized, this explicit preparation step returns its local path:

```sh
MODEL="$("$PYTHON" - <<'PY'
from huggingface_hub import snapshot_download
print(snapshot_download(
    repo_id="mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit",
    revision="41d3337e8b7f2843a75841595fc14e4b9a7a4b96",
))
PY
)"
REVISION="41d3337e8b7f2843a75841595fc14e4b9a7a4b96"
```

Do not pass a repository identifier to the renderer or label different assets with the tested revision.

## Preflight before drafting

Check the explicit interpreter, dependencies and tools without loading model weights:

```sh
"$PYTHON" - <<'PY'
import importlib.metadata
import platform
import shutil
import sys
print("Interpreter:", sys.executable)
print("Platform:", platform.system(), platform.machine())
missing = []
if (platform.system(), platform.machine()) != ("Darwin", "arm64"):
    missing.append("Apple Silicon macOS rendering host")
for package in ("mlx-audio", "mlx", "numpy", "transformers"):
    try:
        print(package, importlib.metadata.version(package))
    except importlib.metadata.PackageNotFoundError:
        missing.append(package)
for tool in ("ffmpeg", "ffprobe"):
    path = shutil.which(tool)
    if path is None:
        missing.append(tool)
    print(tool, path)
if missing:
    raise RuntimeError("Missing prerequisites: " + ", ".join(missing))
PY
"$PYTHON" - "$SKILL/scripts" "$MODEL" "$REVISION" <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from podcast_qwen import backend_info
info = backend_info(Path(sys.argv[2]), sys.argv[3])
print("Complete model assets:", info["model_fingerprint"])
PY
```

On Apple Silicon, check GPU access in the actual execution context:

```sh
"$PYTHON" - <<'PY'
import mlx.core as mx
from mlx_audio.tts.utils import load_model
if not mx.metal.is_available():
    raise RuntimeError("Metal GPU is unavailable")
with mx.stream(mx.gpu):
    result = mx.array([1.0]) + 1.0
    mx.eval(result)
print("GPU check:", result.item())
PY
```

A sandbox can deny GPU access even when the host has Metal. Use the host's permitted GPU execution route; do not mistake this for a missing model or reinstall packages to repair it. Prefer explicit environment paths and conventional caches over broad filesystem searches. Report missing prerequisites together and request only setup permission still needed.

One 409-word English test produced 162.27 seconds of audio, with about 9.1 GB process-wide MLX peak memory and 65 seconds of segment synthesis, excluding loading and hashing. These are observations from one run, not minimum memory requirements or a throughput guarantee. A short audition can estimate pace for the selected voice; include assembly gaps in duration estimates.

## Recipe controls and validation

The `voice` object supports `voice`, `lang_code`, `instruct`, `temperature`, `top_p`, `top_k`, `repetition_penalty` and `max_tokens`. Unknown settings are rejected. Qwen's apparent `speed` argument is not an implemented rate control in the tested route.

```sh
"$PYTHON" "$SKILL/scripts/podcast.py" validate "$PRODUCTION/episode.json"
```

This prints canonical and synthesis chunks without loading MLX. Empty narration and oversized paragraphs fail. Keep each paragraph within `max_words`, which defaults to 95. This is an editorial bound, not a language-independent token estimate. Split oversized paragraphs at sentence boundaries without changing their words.

## Audition and render

When auditioning, use a short recipe and a separate output directory. Ask for sample selection only when voice or pronunciation is unresolved; a routine test with an authorized default can proceed. Production authorization is not listening approval.

```sh
"$PYTHON" "$SKILL/scripts/podcast.py" render "$PRODUCTION/episode.json" \
  --model "$MODEL" --revision "$REVISION" --out "$RUN"
```

The command writes `manifest.json`, `transcript.md` and selected WAV takes. Model assets and package versions are recorded in provenance. It stays in the foreground; wait for its exit and inspect the manifest. Interrupt with Ctrl-C if necessary; completed takes remain on disk.

## Resume and retake

Repeat the render command to resume compatible work. File hashes are checked before reuse. Changed inputs cannot reuse incompatible takes. To replace one chunk, copy its identifier from `validate` or the manifest:

```sh
"$PYTHON" "$SKILL/scripts/podcast.py" render "$PRODUCTION/episode.json" \
  --model "$MODEL" --revision "$REVISION" --out "$RUN" \
  --chunks opening-001 --retake --seed-offset 100
```

Previous takes remain intact. Unselected compatible chunks retain their audio. Changing a pronunciation mapping invalidates only affected synthesis text. A subset render can leave an incomplete episode; mastering rejects missing selections. Run the normal render command to finish it.

## Record actual human review

Use this command only for an actual listening statement about the audition in the specified run:

```sh
"$PYTHON" "$SKILL/scripts/podcast.py" review "$AUDITION_RUN" \
  --scope audition --reviewer "User" --result approved \
  --evidence "The user selected this sample after listening."
```

Supported scopes are `audition`, `episode` and `critical-facts`; results are `approved` or `rejected`. An audition record is not whole-episode approval. Episode review requires a current encoded delivery. Changing output makes older reviews historical. Do not record script or setup authorization as a listening review.

## Master and inspect

```sh
"$PYTHON" "$SKILL/scripts/podcast.py" master "$RUN"
```

Defaults are mono MP3 at the source sample rate, 128 kbit/s, -19 LUFS, -1.5 dBTP and LRA target 11. Assembly trims outer silence with an 80 ms margin and adds 0.35 seconds between chunks and 1.2 seconds between sections. It does not time-stretch or overlap narration. FFmpeg and FFprobe paths can be passed explicitly with `--ffmpeg` and `--ffprobe`.

Each master creates a directory under `deliveries/` with chapter samples, trimming, inserted gaps, metadata and encoded-audio checks. Full decoding, final loudness and sample clipping checks run after encoding.

Inspect `report.json` and `notes.md` in the selected delivery for mastering checks. Use `$RUN/manifest.json` for current delivery, ASR and review state: delivery-directory manifests and reports are snapshots from mastering. Later comparisons and reviews update the root manifest. Do not call an unverified check a pass.

## Compare raw transcripts

Use raw ASR on the final encoded audio. Do not pass filler-cleaned or repetition-cleaned dictation as a fidelity check. The input JSON has `model`, `audio_sha256` and `segments`; each segment has `id` and `text`, matching all canonical chunk identifiers. Use `assembly.json` sample offsets to transcribe aligned chunks from the decoded delivery. The audio hash must match.

```sh
"$PYTHON" "$SKILL/scripts/podcast.py" compare-asr "$RUN" "$RAW_ASR"
```

The report retains raw text, normalization rules, edit counts and mismatch spans. One recognizer is useful evidence; a second is optional. Scores are transcript agreement, not naturalness or proof of correct facts. Missing ASR remains unverified and need not prevent delivery unless the user requires it.

### Optional local Whisper transcription

The helper uses `mlx-whisper==0.4.3` and an existing local Whisper model. Install optional dependencies only with applicable setup permission:

```sh
"$PYTHON" -m pip install -r "$SKILL/requirements-asr.txt"
```

The tested recognizer is `mlx-community/whisper-large-v3-turbo`, revision `a4aaeec0636e6fef84abdcbe3544cb2bf7e9f6fb`. Set `$ASR_MODEL` to its local snapshot, which needs `config.json` and `weights.safetensors` or `weights.npz`. This backend supplies its own tokenizer; no Hugging Face processor files are needed. The helper does not download or install anything.

```sh
"$PYTHON" "$SKILL/scripts/podcast_transcribe.py" "$RUN" \
  --model "$ASR_MODEL" --revision "a4aaeec0636e6fef84abdcbe3544cb2bf7e9f6fb" \
  --language en --ffmpeg ffmpeg
```

Use the actual model revision and narration language for other snapshots. The command prints timestamped `raw_transcript` and recognizer `details` paths in the current delivery directory. Set `$RAW_ASR` to the printed transcript path and run `compare-asr` above.

It locks the run, verifies the current delivery hash and decoded sample count, and uses assembly boundaries for each chunk. Only recognizer input is resampled to sixteen kilohertz. Raw text, fillers and repetitions are retained; no canonical script is supplied as a prompt. Completed chunks are checkpointed if a later chunk fails. Rerunning creates new files rather than overwriting prior transcripts. Transcription alone does not mark ASR as compared or record a human review.

Read [evaluation limits](evaluation.md), including the treatment of hyphen symbols, before interpreting mismatches.

## Test the copied skill

From the checkout, use a separate test environment; installation needs applicable permission:

```sh
python3 -m venv tmp/test-env
TEST_PYTHON="$SKILL/tmp/test-env/bin/python"
"$TEST_PYTHON" -m pip install -r "$SKILL/requirements-test.txt"
"$TEST_PYTHON" -m pytest "$SKILL/tests"
"$TEST_PYTHON" -m ruff check "$SKILL/scripts" "$SKILL/tests"
"$TEST_PYTHON" -m ruff format --check "$SKILL/scripts" "$SKILL/tests"
```

Tests use deterministic TTS and recognizer doubles; FFmpeg integration checks run when tools are available. They do not establish real voice quality or GPU compatibility. For a release smoke test, render and master a short sample with a local model, verify resume preserves its takes, and listen to the encoded delivery. Record the listening outcome separately.

When testing through Vis, run `vis-agent python -m pytest "$SKILL/tests"` from your normal workspace, outside the skill directory, to avoid selecting an incomplete project environment.
