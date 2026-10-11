"""Backend tests use small local fixtures and a fake synthesis model."""

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import types
import wave
from pathlib import Path

import numpy as np
import pytest

MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "podcast_qwen.py"
SPEC = importlib.util.spec_from_file_location("podcast_qwen", MODULE_PATH)
backend = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(backend)

VOICE = {
    "voice": "Aiden",
    "lang_code": "English",
    "instruct": "Read clearly.",
    "temperature": 0.65,
    "top_p": 0.9,
    "top_k": 50,
    "repetition_penalty": 1.05,
    "max_tokens": 1800,
}


@pytest.fixture
def model_dir(tmp_path):
    path = tmp_path / "model"
    (path / "speech_tokenizer").mkdir(parents=True)
    config = {
        "model_type": "qwen3_tts",
        "tts_model_type": "custom_voice",
        "tts_model_size": "1b7",
        "talker_config": {
            "spk_id": {"aiden": 1},
            "codec_language_id": {"english": 2},
        },
    }
    (path / "config.json").write_text(json.dumps(config))
    for name in ("tokenizer_config.json", "vocab.json", "speech_tokenizer/config.json"):
        (path / name).write_text("{}")
    (path / "merges.txt").write_text("a b\n")
    (path / "model.safetensors").write_bytes(b"model-weights")
    (path / "speech_tokenizer/model.safetensors").write_bytes(b"codec-weights")
    return path


class FakeModel:
    def __init__(self, results, error=None):
        self.results = results
        self.error = error
        self.calls = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        yield from self.results
        if self.error:
            raise self.error


def install_runtime(monkeypatch, results, error=None):
    seeds = []
    mx = types.SimpleNamespace(
        random=types.SimpleNamespace(seed=seeds.append),
        eval=lambda audio: None,
        get_peak_memory=lambda: 123456,
    )
    model = FakeModel(results, error)
    loads = []

    def load(path):
        loads.append(path)
        return mx, np, model

    monkeypatch.setattr(backend, "_load_runtime", load)
    return model, loads, seeds


def result(audio=None, rate=24000):
    if audio is None:
        audio = np.array([0.0, 0.25, -0.25, 1.0, -1.0], dtype=np.float32)
    return types.SimpleNamespace(audio=audio, sample_rate=rate)


def test_module_import_is_lightweight():
    code = (
        "import runpy,sys; runpy.run_path(sys.argv[1]); "
        "assert 'numpy' not in sys.modules; assert 'mlx.core' not in sys.modules; "
        "assert 'mlx_audio' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code, str(MODULE_PATH)], check=True)


def test_fingerprint_tracks_bytes_not_only_size_or_mtime(model_dir):
    first = backend.backend_info(model_dir, "test-revision")
    asset = model_dir / "model.safetensors"
    old_stat = asset.stat()
    asset.write_bytes(b"MODEL-weights")
    os.utime(asset, ns=(old_stat.st_atime_ns, old_stat.st_mtime_ns))
    changed = backend.backend_info(model_dir, "test-revision")
    assert first["model_fingerprint"] != changed["model_fingerprint"]
    assert first["model_revision"] == "test-revision"
    assert first["backend"] == "mlx-qwen3-tts-customvoice"
    assert set(first["package_versions"]) == {"mlx-audio", "mlx", "numpy", "transformers"}
    assert not any(Path(asset["path"]).is_absolute() for asset in first["model_assets"])


def test_huggingface_symlink_assets_are_hashed(model_dir):
    asset = model_dir / "model.safetensors"
    blob = model_dir.parent / "cached-blob"
    asset.rename(blob)
    asset.symlink_to(blob)
    info = backend.backend_info(model_dir, "cached-revision")
    weight = next(a for a in info["model_assets"] if a["path"] == asset.name)
    assert weight["sha256"] == hashlib.sha256(blob.read_bytes()).hexdigest()


@pytest.mark.parametrize("revision", [None, "", "  ", 123])
def test_revision_is_explicit(model_dir, revision):
    with pytest.raises(ValueError, match="model_revision"):
        backend.backend_info(model_dir, revision)


@pytest.mark.parametrize(
    "missing",
    [
        "config.json",
        "model.safetensors",
        "merges.txt",
        "speech_tokenizer/config.json",
        "speech_tokenizer/model.safetensors",
    ],
)
def test_incomplete_model_fails_before_load(model_dir, missing):
    (model_dir / missing).unlink()
    with pytest.raises(ValueError, match="Missing"):
        backend.QwenRenderer(model_dir)


def test_model_id_does_not_become_download(tmp_path):
    with pytest.raises(ValueError, match="existing local directory"):
        backend.QwenRenderer(tmp_path / "remote-org" / "remote-model")


@pytest.mark.parametrize(
    "key,value", [("model_type", "other"), ("tts_model_type", "base"), ("tts_model_size", "0b6")]
)
def test_other_routes_are_rejected(model_dir, key, value):
    path = model_dir / "config.json"
    config = json.loads(path.read_text())
    config[key] = value
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="Only Qwen3-TTS"):
        backend.QwenRenderer(model_dir)


@pytest.mark.parametrize(
    "filename",
    ["../escape.safetensors", "/absolute.safetensors", "missing.safetensors", "wrong.bin"],
)
def test_weight_index_cannot_reference_unavailable_assets(model_dir, filename):
    (model_dir / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"layer": filename}})
    )
    with pytest.raises(ValueError):
        backend.QwenRenderer(model_dir)


def test_lazy_serial_render_preserves_rate_and_metadata(monkeypatch, model_dir, tmp_path):
    model, loads, seeds = install_runtime(monkeypatch, [result(), result([0.5, -0.5])])
    renderer = backend.QwenRenderer(model_dir)
    assert not loads
    target = tmp_path / "first.wav"
    metadata = renderer.render("Example narration.", VOICE, 123, target)
    with wave.open(str(target), "rb") as audio:
        assert audio.getframerate() == 24000
        assert audio.getnchannels() == 1
        assert audio.getsampwidth() == 2
        assert audio.getnframes() == 7
        pcm = np.frombuffer(audio.readframes(7), dtype="<i2")
    assert pcm.tolist() == [0, 8192, -8192, 32767, -32768, 16384, -16384]
    assert metadata["samples"] == 7
    assert metadata["duration_seconds"] == 7 / 24000
    assert metadata["sample_rate"] == 24000
    assert metadata["peak_memory_bytes"] == 123456
    assert metadata["sha256"] == hashlib.sha256(target.read_bytes()).hexdigest()
    assert metadata["pcm_format"] == "s16le-mono"
    assert metadata["render_seconds"] >= 0
    renderer.render("Second narration.", VOICE, 124, tmp_path / "second.wav")
    assert len(loads) == 1
    assert seeds == [123, 124]
    assert model.calls[0] == {"text": "Example narration.", **VOICE}


@pytest.mark.parametrize(
    "setting,value",
    [
        ("speed", 1.0),
        ("stream", True),
        ("pitch", 1),
        ("temperature", float("nan")),
        ("temperature", -1),
        ("top_p", 0),
        ("top_p", 1.1),
        ("top_k", True),
        ("max_tokens", 0),
        ("repetition_penalty", 0),
        ("voice", "Missing"),
        ("lang_code", "Missing"),
    ],
)
def test_invalid_controls_fail_before_loading(monkeypatch, model_dir, tmp_path, setting, value):
    _, loads, _ = install_runtime(monkeypatch, [result()])
    with pytest.raises(ValueError):
        backend.QwenRenderer(model_dir).render(
            "Text", {**VOICE, setting: value}, 1, tmp_path / "out.wav"
        )
    assert not loads
    assert not (tmp_path / "out.wav").exists()


@pytest.mark.parametrize("seed", [True, -1, 4294967296, 0.5])
def test_invalid_seed_fails_before_loading(monkeypatch, model_dir, tmp_path, seed):
    _, loads, _ = install_runtime(monkeypatch, [result()])
    with pytest.raises(ValueError, match="seed"):
        backend.QwenRenderer(model_dir).render("Text", VOICE, seed, tmp_path / "out.wav")
    assert not loads


@pytest.mark.parametrize("text", [None, "", " "])
def test_empty_text_fails_before_loading(monkeypatch, model_dir, tmp_path, text):
    _, loads, _ = install_runtime(monkeypatch, [result()])
    with pytest.raises(ValueError, match="Narration"):
        backend.QwenRenderer(model_dir).render(text, VOICE, 1, tmp_path / "out.wav")
    assert not loads


@pytest.mark.parametrize(
    "results",
    [
        [],
        [result([])],
        [result([[0.2], [0.3]])],
        [result([float("nan")])],
        [result([float("inf")])],
        [result([1.001])],
        [result(rate=0)],
        [result(rate=24000.0)],
        [result(rate=True)],
        [result(), result(rate=44100)],
    ],
)
def test_invalid_audio_never_publishes(monkeypatch, model_dir, tmp_path, results):
    install_runtime(monkeypatch, results)
    target = tmp_path / "out.wav"
    with pytest.raises(ValueError):
        backend.QwenRenderer(model_dir).render("Text", VOICE, 1, target)
    assert not target.exists()
    assert not list(tmp_path.glob("*.partial"))


@pytest.mark.parametrize("error", [KeyboardInterrupt(), RuntimeError("worker failed")])
def test_generation_failure_preserves_no_partial_take(monkeypatch, model_dir, tmp_path, error):
    install_runtime(monkeypatch, [result()], error)
    target = tmp_path / "out.wav"
    with pytest.raises(type(error)):
        backend.QwenRenderer(model_dir).render("Text", VOICE, 1, target)
    assert not target.exists()
    assert not list(tmp_path.glob(".*.partial"))


def test_existing_and_dangling_targets_are_never_replaced(monkeypatch, model_dir, tmp_path):
    _, loads, _ = install_runtime(monkeypatch, [result()])
    target = tmp_path / "out.wav"
    target.write_bytes(b"original")
    with pytest.raises(FileExistsError):
        backend.QwenRenderer(model_dir).render("Text", VOICE, 1, target)
    assert target.read_bytes() == b"original"
    target.unlink()
    target.symlink_to(tmp_path / "does-not-exist")
    with pytest.raises(FileExistsError):
        backend.QwenRenderer(model_dir).render("Text", VOICE, 1, target)
    assert target.is_symlink()
    assert not loads


def test_publication_race_preserves_the_other_writer(monkeypatch, model_dir, tmp_path):
    install_runtime(monkeypatch, [result()])
    target = tmp_path / "out.wav"
    real_link = os.link

    def racing_link(source, destination):
        target.write_bytes(b"another-writer")
        real_link(source, destination)

    monkeypatch.setattr(backend.os, "link", racing_link)
    with pytest.raises(FileExistsError):
        backend.QwenRenderer(model_dir).render("Text", VOICE, 1, target)
    assert target.read_bytes() == b"another-writer"
    assert not list(tmp_path.glob(".*.partial"))


def test_interrupted_publication_cleans_temporary_file(monkeypatch, model_dir, tmp_path):
    install_runtime(monkeypatch, [result()])
    target = tmp_path / "out.wav"

    def interrupt(*args):
        raise KeyboardInterrupt

    monkeypatch.setattr(backend.os, "link", interrupt)
    with pytest.raises(KeyboardInterrupt):
        backend.QwenRenderer(model_dir).render("Text", VOICE, 1, target)
    assert not target.exists()
    assert not list(tmp_path.glob(".*.partial"))


@pytest.mark.parametrize(
    "filename", ["tokenizer_config.json", "speech_tokenizer/config.json", "generation_config.json"]
)
def test_malformed_tokenizer_json_fails_before_load(model_dir, filename):
    (model_dir / filename).write_text("not JSON")
    with pytest.raises(ValueError, match="model JSON"):
        backend.QwenRenderer(model_dir)


def test_missing_output_directory_fails_before_load(monkeypatch, model_dir, tmp_path):
    _, loads, _ = install_runtime(monkeypatch, [result()])
    with pytest.raises(ValueError, match="Output directory"):
        backend.QwenRenderer(model_dir).render("Text", VOICE, 1, tmp_path / "missing" / "out.wav")
    assert not loads


def install_fake_imports(monkeypatch, model):
    mx = types.ModuleType("mlx.core")
    mlx = types.ModuleType("mlx")
    mlx.core = mx
    mlx.__path__ = []
    audio = types.ModuleType("mlx_audio")
    audio.__path__ = []
    tts = types.ModuleType("mlx_audio.tts")
    tts.__path__ = []
    utils = types.ModuleType("mlx_audio.tts.utils")
    calls = []

    def load(path):
        calls.append(
            (path, os.environ.get("HF_HUB_OFFLINE"), os.environ.get("TRANSFORMERS_OFFLINE"))
        )
        return model

    utils.load_model = load
    for module in (mlx, mx, audio, tts, utils):
        monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(backend.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(backend.platform, "machine", lambda: "arm64")
    monkeypatch.setenv("HF_HUB_OFFLINE", "0")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "0")
    monkeypatch.setenv("HF_HUB_DISABLE_TELEMETRY", "0")
    return calls


def test_runtime_forces_offline_before_local_load(monkeypatch, model_dir):
    model = types.SimpleNamespace(tokenizer=object(), speech_tokenizer=object())
    calls = install_fake_imports(monkeypatch, model)
    _, _, loaded = backend._load_runtime(model_dir)
    assert calls == [(model_dir, "1", "1")]
    assert os.environ["HF_HUB_DISABLE_TELEMETRY"] == "1"
    assert loaded is model


@pytest.mark.parametrize("missing", ["tokenizer", "speech_tokenizer"])
def test_partial_loader_success_is_not_a_ready_model(monkeypatch, model_dir, missing):
    model = types.SimpleNamespace(tokenizer=object(), speech_tokenizer=object())
    setattr(model, missing, None)
    install_fake_imports(monkeypatch, model)
    with pytest.raises(RuntimeError, match="did not load"):
        backend._load_runtime(model_dir)


def test_unsupported_platform_fails_clearly(monkeypatch, model_dir):
    monkeypatch.setattr(backend.platform, "system", lambda: "Linux")
    with pytest.raises(RuntimeError, match="Apple Silicon"):
        backend._load_runtime(model_dir)
