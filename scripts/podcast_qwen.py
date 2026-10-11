"""Offline Qwen3-TTS 1.7B CustomVoice rendering for the podcast skill.

Importing this module uses only the standard library. Rendering requires MLX on
Apple Silicon. Model files must already be present; this module never downloads.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import platform
import tempfile
import time
import wave
from pathlib import Path

BACKEND_ID = "mlx-qwen3-tts-customvoice"
BACKEND_VERSION = 1
VOICE_KEYS = frozenset(
    {
        "voice",
        "lang_code",
        "instruct",
        "temperature",
        "top_p",
        "top_k",
        "repetition_penalty",
        "max_tokens",
    }
)
ASSET_SUFFIXES = frozenset({".json", ".safetensors", ".txt", ".model"})


def _read_json(path):
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read model JSON: {path}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"Model JSON must be an object: {path}")
    return data


def _require_file(path):
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"Missing or empty local model asset: {path}")


def _model_assets(model_dir):
    path = Path(model_dir).expanduser()
    if not path.is_dir():
        raise ValueError(f"Model must be an existing local directory: {path}")
    path = path.resolve()
    _require_file(path / "config.json")
    config = _read_json(path / "config.json")
    required = {
        "model_type": "qwen3_tts",
        "tts_model_type": "custom_voice",
        "tts_model_size": "1b7",
    }
    if any(config.get(key) != value for key, value in required.items()):
        raise ValueError("Only Qwen3-TTS 1.7B CustomVoice local models are supported")
    _require_file(path / "tokenizer_config.json")
    if not (path / "tokenizer.json").is_file():
        _require_file(path / "vocab.json")
        _require_file(path / "merges.txt")
    _require_file(path / "speech_tokenizer" / "config.json")
    for directory in (path, path / "speech_tokenizer"):
        weights = sorted(directory.glob("*.safetensors"))
        if not weights:
            raise ValueError(f"Missing local safetensors weights: {directory}")
        for weight in weights:
            _require_file(weight)
        index = directory / "model.safetensors.index.json"
        if index.exists():
            weight_map = _read_json(index).get("weight_map")
            if not isinstance(weight_map, dict) or not weight_map:
                raise ValueError(f"Invalid model weight index: {index}")
            for name in weight_map.values():
                if not isinstance(name, str) or Path(name).name != name:
                    raise ValueError(f"Unsafe model weight filename in {index}")
                if not name.endswith(".safetensors"):
                    raise ValueError(f"Invalid model weight filename in {index}")
                _require_file(directory / name)
    assets = sorted(
        p
        for directory in (path, path / "speech_tokenizer")
        for p in directory.iterdir()
        if p.suffix in ASSET_SUFFIXES and not p.name.startswith(".")
    )
    for asset in assets:
        _require_file(asset)
        if asset.suffix == ".json":
            _read_json(asset)
    return path, assets, config


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        before = os.fstat(stream.fileno())
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
        after = os.fstat(stream.fileno())
    current = path.stat()

    def identity(stat):
        return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns

    if identity(before) != identity(after) or identity(after) != identity(current):
        raise RuntimeError(f"File changed while hashing: {path}")
    return digest.hexdigest()


def backend_info(model_dir, model_revision):
    """Fingerprint local assets without importing or loading the model.

    Missing optional packages appear as null versions. The render operation
    reports missing runtime packages before loading weights.
    """
    if not isinstance(model_revision, str) or not model_revision.strip():
        raise ValueError("model_revision must be an explicit nonempty string")
    path, assets, _ = _model_assets(model_dir)
    files = [
        {
            "path": asset.relative_to(path).as_posix(),
            "sha256": _sha256(asset),
            "bytes": asset.stat().st_size,
        }
        for asset in assets
    ]
    fingerprint = hashlib.sha256(
        json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    versions = {}
    for package in ("mlx-audio", "mlx", "numpy", "transformers"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return {
        "backend": BACKEND_ID,
        "backend_version": BACKEND_VERSION,
        "implementation_sha256": _sha256(Path(__file__)),
        "model_revision": model_revision.strip(),
        "model_fingerprint": fingerprint,
        "package_versions": versions,
        "model_assets": files,
    }


def _voice_settings(voice, config):
    if not isinstance(voice, dict):
        raise ValueError("voice must be a settings object")
    unsupported = set(voice) - VOICE_KEYS
    if unsupported:
        raise ValueError(f"Unsupported Qwen controls: {sorted(unsupported)}")
    for key in ("voice", "lang_code"):
        if not isinstance(voice.get(key), str) or not voice[key].strip():
            raise ValueError(f"voice.{key} must be a nonempty string")
    if "instruct" in voice and not isinstance(voice["instruct"], str):
        raise ValueError("voice.instruct must be a string")
    for key in ("temperature", "top_p", "repetition_penalty"):
        if key in voice:
            value = voice[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"voice.{key} must be a finite number")
            if not math.isfinite(value):
                raise ValueError(f"voice.{key} must be a finite number")
            minimum = 0 if key == "temperature" else 0.000001
            maximum = 1 if key == "top_p" else math.inf
            if not minimum <= value <= maximum:
                raise ValueError(f"voice.{key} is outside its supported range")
    for key, minimum in (("top_k", 0), ("max_tokens", 1)):
        if key in voice and (type(voice[key]) is not int or voice[key] < minimum):
            raise ValueError(f"voice.{key} must be an integer >= {minimum}")
    talker = config.get("talker_config", {})
    speakers = talker.get("spk_id", {})
    languages = talker.get("codec_language_id", {})
    if speakers and voice["voice"].lower() not in speakers:
        raise ValueError(f"Unsupported speaker: {voice['voice']}")
    if languages and voice["lang_code"].lower() not in {*languages, "auto"}:
        raise ValueError(f"Unsupported language: {voice['lang_code']}")
    return dict(voice)


def _load_runtime(model_dir):
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise RuntimeError("Qwen MLX rendering requires Apple Silicon macOS")
    # Set these before importing libraries that cache offline state at import.
    # Keep them set for the worker's lifetime; there is no online fallback.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    try:
        import mlx.core as mx
        import numpy as np
        from mlx_audio.tts.utils import load_model
    except ImportError as exc:
        raise RuntimeError(
            "Install the documented MLX rendering dependencies in this interpreter"
        ) from exc
    # load_model receives an existing resolved Path, never a repository ID.
    model = load_model(model_dir)
    if getattr(model, "tokenizer", None) is None:
        raise RuntimeError("The local text tokenizer did not load")
    if getattr(model, "speech_tokenizer", None) is None:
        raise RuntimeError("The local speech tokenizer did not load")
    return mx, np, model


def _publish_wav(target, pcm, sample_rate):
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".partial", dir=target.parent
    )
    temporary = Path(temporary)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            with wave.open(stream, "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(sample_rate)
                wav.writeframes(pcm)
            stream.flush()
            os.fsync(stream.fileno())
        # Hard-link publication is atomic and refuses existing names, including
        # dangling symlinks. Temporary and target share one filesystem.
        os.link(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


class QwenRenderer:
    """Validate a local model now, and load it on the first render only.

    Use one instance serially within an owned worker process. MLX's RNG is
    process-global, so concurrent calls in one process are not supported.
    """

    def __init__(self, model_dir):
        self.model_dir, _, self.config = _model_assets(model_dir)
        self._runtime = None

    def render(self, text, voice, seed, target_path):
        """Render native-rate mono PCM16 audio without overwriting a take.

        Samples outside [-1, 1] fail. A positive full-scale sample maps to
        PCM16's maximum 32767; this is quantization, not normalization.
        Peak memory is the process-wide MLX high-water mark, not per-take RSS.
        """
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Narration must be a nonempty string")
        if type(seed) is not int or not 0 <= seed <= 0xFFFFFFFF:
            raise ValueError("seed must be an integer between 0 and 4294967295")
        settings = _voice_settings(voice, self.config)
        target = Path(target_path).expanduser()
        if target.exists() or target.is_symlink():
            raise FileExistsError(f"Refusing to overwrite audio take: {target}")
        if not target.parent.is_dir():
            raise ValueError(f"Output directory does not exist: {target.parent}")
        if self._runtime is None:
            self._runtime = _load_runtime(self.model_dir)
        mx, np, model = self._runtime
        mx.random.seed(seed)
        started = time.monotonic()
        arrays = []
        sample_rate = None
        for result in model.generate(text=text, **settings):
            mx.eval(result.audio)
            audio = np.asarray(result.audio, dtype=np.float32)
            rate = result.sample_rate
            if isinstance(rate, bool) or not isinstance(rate, (int, np.integer)):
                raise ValueError("Generated sample rate must be an integer")
            rate = int(rate)
            if not 8000 <= rate <= 192000:
                raise ValueError("Generated sample rate is outside 8000..192000 Hz")
            if sample_rate is not None and sample_rate != rate:
                raise ValueError("Sample rate changed within one audio take")
            if audio.ndim != 1 or not audio.size:
                raise ValueError("Generated audio must be nonempty mono samples")
            if not np.isfinite(audio).all():
                raise ValueError("Generated audio contains nonfinite samples")
            if float(np.max(np.abs(audio))) > 1.0:
                raise ValueError("Generated audio clips; render another take")
            sample_rate = rate
            arrays.append(audio.copy())
        if not arrays:
            raise ValueError("The model generated no audio")
        audio = np.concatenate(arrays)
        peak = float(np.max(np.abs(audio)))
        pcm = np.rint(np.clip(audio * 32768.0, -32768, 32767)).astype("<i2")
        _publish_wav(target, pcm.tobytes(), sample_rate)
        return {
            "sample_rate": sample_rate,
            "samples": int(audio.size),
            "duration_seconds": int(audio.size) / sample_rate,
            "render_seconds": time.monotonic() - started,
            "peak_memory_bytes": int(mx.get_peak_memory()),
            "sha256": _sha256(target),
            "sample_peak": peak,
            "pcm_format": "s16le-mono",
        }
