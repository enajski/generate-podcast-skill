#!/usr/bin/env python3
"""Produce raw, aligned ASR from an encoded delivery using local Whisper assets."""

import argparse
import importlib.metadata
import json
import math
import os
import sys
import tempfile
import time
import wave
from pathlib import Path

from podcast import _manifest, output_lock, safe_file, selection_hash
from podcast_core import atomic_json, file_sha256
from podcast_media import _run, _tool


def _load_runtime():
    # Set offline mode before importing libraries that cache it at import time.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    try:
        import mlx_whisper
        import numpy as np
        from scipy.signal import resample_poly
    except ImportError as exc:
        raise RuntimeError("Install requirements-asr.txt in the selected interpreter") from exc
    return np, resample_poly, mlx_whisper.transcribe, importlib.metadata.version("mlx-whisper")


def transcribe_episode(output, model, revision, *, language="en", ffmpeg="ffmpeg"):
    """Checkpoint raw recognizer results without changing takes or review status."""
    if not isinstance(revision, str) or not revision.strip():
        raise ValueError("An explicit model revision is required")
    if not isinstance(language, str) or not language.strip():
        raise ValueError("An explicit narration language is required")
    model = Path(model).expanduser().resolve()
    config = model / "config.json"
    weights = model / "weights.safetensors"
    if not weights.is_file():
        weights = model / "weights.npz"
    if not all(p.is_file() and p.stat().st_size for p in (config, weights)):
        raise ValueError("Whisper requires an existing local config.json and weights file")
    model_config = json.loads(config.read_text())
    if not isinstance(model_config, dict) or model_config.get("model_type", "whisper") != "whisper":
        raise ValueError("The local model must be Whisper")
    ffmpeg = _tool(ffmpeg)
    output = Path(output).resolve()
    if not (output / "manifest.json").is_file():
        raise ValueError("Transcription requires an existing podcast run")
    with output_lock(output):
        state = _manifest(output)
        delivery = state.get("delivery") if state else None
        if not delivery or delivery.get("selection_hash") != selection_hash(state):
            raise ValueError("Transcription requires a current encoded delivery")
        audio = safe_file(output, delivery["audio_path"])
        digest = file_sha256(audio)
        if digest != delivery["audio_sha256"]:
            raise ValueError("Encoded audio changed since mastering")
        assembly_path = safe_file(
            output, (audio.parent / "assembly.json").relative_to(output).as_posix()
        )
        assembly = json.loads(assembly_path.read_text())
        rate, frames = assembly["sample_rate"], assembly["samples"]
        if type(rate) is not int or not 8000 <= rate <= 192000:
            raise ValueError("Invalid assembly sample rate")
        if type(frames) is not int or frames <= 0:
            raise ValueError("Invalid assembly sample count")
        chunks = assembly["chunks"]
        if not chunks or [c["id"] for c in chunks] != [c["id"] for c in state["chunks"]]:
            raise ValueError("Assembly chunks differ from the current manifest")
        cursor = 0
        for chunk in chunks:
            start, count = chunk["start_sample"], chunk["samples"]
            if (
                type(start) is not int
                or type(count) is not int
                or start < cursor
                or count <= 0
                or start + count > frames
            ):
                raise ValueError("Invalid or overlapping assembly chunk boundaries")
            cursor = start + count
        np, resample_poly, recognize, version = _load_runtime()
        raw = {
            "model": f"Local Whisper@{revision.strip()}; mlx-whisper=={version}",
            "audio_sha256": digest,
            "segments": [],
        }
        details = {
            "model": raw["model"],
            "audio_sha256": digest,
            "model_assets": [
                {"path": p.name, "sha256": file_sha256(p), "bytes": p.stat().st_size}
                for p in (config, weights)
            ],
            "settings": {
                "language": language,
                "temperature": 0.0,
                "condition_on_previous_text": False,
                "initial_prompt": None,
                "cleanup": "none",
            },
            "status": "incomplete",
            "segments": [],
        }
        token = time.time_ns()
        raw_path = audio.parent / f"raw-asr-{token}.json"
        details_path = audio.parent / f"raw-asr-details-{token}.json"
        with tempfile.TemporaryDirectory(prefix="podcast-asr-") as temporary:
            decoded = Path(temporary) / "decoded.wav"
            _run(
                [
                    ffmpeg,
                    "-nostdin",
                    "-v",
                    "error",
                    "-i",
                    str(audio),
                    "-ac",
                    "1",
                    "-ar",
                    str(rate),
                    "-c:a",
                    "pcm_s16le",
                    str(decoded),
                ]
            )
            with wave.open(str(decoded), "rb") as source:
                if (
                    source.getnchannels(),
                    source.getsampwidth(),
                    source.getframerate(),
                    source.getnframes(),
                ) != (1, 2, rate, frames):
                    raise ValueError("Decoded audio metadata differs from assembly")
                for chunk in chunks:
                    source.setpos(chunk["start_sample"])
                    pcm = source.readframes(chunk["samples"])
                    if len(pcm) != chunk["samples"] * 2:
                        raise ValueError("Truncated decoded chunk")
                    waveform = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
                    factor = math.gcd(16000, rate)
                    waveform = resample_poly(waveform, 16000 // factor, rate // factor).astype(
                        np.float32
                    )
                    result = recognize(
                        waveform,
                        path_or_hf_repo=str(model),
                        language=language,
                        temperature=0.0,
                        condition_on_previous_text=False,
                        initial_prompt=None,
                        verbose=None,
                        word_timestamps=False,
                    )
                    if not isinstance(result.get("text"), str):
                        raise ValueError("Recognizer returned no raw text")
                    raw["segments"].append({"id": chunk["id"], "text": result["text"]})
                    details["segments"].append({"id": chunk["id"], "result": result})
                    atomic_json(details_path, details)
                    atomic_json(raw_path, raw)
                    print(json.dumps(raw["segments"][-1], ensure_ascii=False), flush=True)
        if file_sha256(audio) != digest:
            raise ValueError("Encoded audio changed during transcription")
        details["status"] = "complete"
        atomic_json(details_path, details)
        return {"raw_transcript": str(raw_path), "details": str(details_path)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--language", default="en")
    parser.add_argument("--ffmpeg", default="ffmpeg")
    args = parser.parse_args(argv)
    try:
        print(
            json.dumps(
                transcribe_episode(
                    args.output,
                    args.model,
                    args.revision,
                    language=args.language,
                    ffmpeg=args.ffmpeg,
                )
            )
        )
    except (ValueError, OSError, RuntimeError, ImportError, KeyError, wave.Error) as error:
        print(f"podcast-asr: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("podcast-asr: interrupted; completed raw chunks remain on disk", file=sys.stderr)
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
