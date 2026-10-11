"""Check aligned raw transcription without a GPU or recognizer model."""

import json
import shutil
import types
import wave
from pathlib import Path

import numpy as np
import podcast_transcribe as transcribe
import pytest
from podcast import selection_hash
from podcast_core import file_sha256


@pytest.fixture
def production(tmp_path):
    run = tmp_path / "run"
    delivery = run / "deliveries" / "test"
    delivery.mkdir(parents=True)
    audio = delivery / "encoded.wav"
    with wave.open(str(audio), "wb") as target:
        target.setnchannels(1)
        target.setsampwidth(2)
        target.setframerate(8000)
        target.writeframes(np.array([1000, 2000, 0, 0, -1000, -2000, 0, 0], dtype="<i2"))
    state = {
        "schema_version": 1,
        "status": "rendered",
        "recipe_hash": "test",
        "selected": {},
        "chunks": [{"id": "one"}, {"id": "two"}],
        "verification": {"asr": "unverified", "human_listening": "unverified"},
    }
    state["delivery"] = {
        "audio_path": audio.relative_to(run).as_posix(),
        "audio_sha256": file_sha256(audio),
        "selection_hash": selection_hash(state),
    }
    (run / "manifest.json").write_text(json.dumps(state))
    assembly = {
        "sample_rate": 8000,
        "samples": 8,
        "chunks": [
            {"id": "one", "start_sample": 0, "samples": 2},
            {"id": "two", "start_sample": 4, "samples": 2},
        ],
    }
    (delivery / "assembly.json").write_text(json.dumps(assembly))
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text('{"model_type": "whisper"}')
    (model / "weights.safetensors").write_bytes(b"local test weights")
    return types.SimpleNamespace(run=run, delivery=delivery, audio=audio, model=model)


@pytest.fixture
def recognizer(monkeypatch):
    calls = []

    def recognize(audio, **settings):
        calls.append((audio.copy(), settings))
        return {"text": " um um, three-cut.", "segments": [{"text": " um um, three-cut."}]}

    def resample(audio, up, down):
        assert (up, down) == (2, 1)
        return np.repeat(audio, up)

    monkeypatch.setattr(transcribe, "_load_runtime", lambda: (np, resample, recognize, "test"))
    return calls


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="FFmpeg unavailable")
def test_aligned_audio_raw_text_and_immutable_delivery(production, recognizer):
    before = (production.run / "manifest.json").read_bytes()
    result = transcribe.transcribe_episode(production.run, production.model, "revision")
    raw = json.loads(Path(result["raw_transcript"]).read_text())
    details = json.loads(Path(result["details"]).read_text())
    assert raw["audio_sha256"] == file_sha256(production.audio)
    assert raw["segments"] == [
        {"id": "one", "text": " um um, three-cut."},
        {"id": "two", "text": " um um, three-cut."},
    ]
    np.testing.assert_array_equal(recognizer[0][0], np.array([1000, 1000, 2000, 2000]) / 32768)
    np.testing.assert_array_equal(recognizer[1][0], -recognizer[0][0])
    assert all(c[1]["initial_prompt"] is None for c in recognizer)
    assert all(c[1]["condition_on_previous_text"] is False for c in recognizer)
    assert details["status"] == "complete"
    assert details["model_assets"][1]["sha256"] == file_sha256(
        production.model / "weights.safetensors"
    )
    assert (production.run / "manifest.json").read_bytes() == before
    assert not (production.run / ".podcast.lock").exists()
    second = transcribe.transcribe_episode(production.run, production.model, "revision")
    assert second["raw_transcript"] != result["raw_transcript"]
    assert json.loads(Path(result["raw_transcript"]).read_text()) == raw


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="FFmpeg unavailable")
def test_partial_results_survive_failure(production, recognizer, monkeypatch):
    runtime = transcribe._load_runtime()

    def fail_second(*args, **kwargs):
        if recognizer:
            raise RuntimeError("recognizer failed")
        return runtime[2](*args, **kwargs)

    monkeypatch.setattr(transcribe, "_load_runtime", lambda: (*runtime[:2], fail_second, "test"))
    with pytest.raises(RuntimeError, match="recognizer failed"):
        transcribe.transcribe_episode(production.run, production.model, "revision")
    details_path = next(production.delivery.glob("raw-asr-details-*.json"))
    details = json.loads(details_path.read_text())
    raw = json.loads(Path(str(details_path).replace("raw-asr-details-", "raw-asr-")).read_text())
    assert details["status"] == "incomplete"
    assert raw["segments"] == [{"id": "one", "text": " um um, three-cut."}]
    assert not (production.run / ".podcast.lock").exists()


def test_modified_encoded_audio_is_rejected_before_model_load(production, monkeypatch):
    production.audio.write_bytes(b"changed")
    monkeypatch.setattr(transcribe, "_load_runtime", lambda: pytest.fail("model loaded"))
    with pytest.raises(ValueError, match="audio changed"):
        transcribe.transcribe_episode(production.run, production.model, "revision")


@pytest.mark.parametrize("start,count", [(1, 2), (7, 2), (4, 0), (True, 2), (4, 1.5)])
def test_bad_assembly_boundaries_are_rejected(production, monkeypatch, start, count):
    path = production.delivery / "assembly.json"
    assembly = json.loads(path.read_text())
    assembly["chunks"][1].update(start_sample=start, samples=count)
    path.write_text(json.dumps(assembly))
    monkeypatch.setattr(transcribe, "_load_runtime", lambda: pytest.fail("model loaded"))
    with pytest.raises(ValueError, match="boundaries"):
        transcribe.transcribe_episode(production.run, production.model, "revision")


def test_missing_model_never_attempts_recognition(production, monkeypatch):
    monkeypatch.setattr(transcribe, "_load_runtime", lambda: pytest.fail("model loaded"))
    with pytest.raises(ValueError, match="existing local"):
        transcribe.transcribe_episode(production.run, "remote/model-id", "revision")


def test_invalid_model_config_is_rejected(production, monkeypatch):
    (production.model / "config.json").write_text("[]")
    monkeypatch.setattr(transcribe, "_load_runtime", lambda: pytest.fail("model loaded"))
    with pytest.raises(ValueError, match="must be Whisper"):
        transcribe.transcribe_episode(production.run, production.model, "revision")


def test_stale_delivery_is_rejected(production, monkeypatch):
    path = production.run / "manifest.json"
    state = json.loads(path.read_text())
    state["recipe_hash"] = "changed recipe"
    path.write_text(json.dumps(state))
    monkeypatch.setattr(transcribe, "_load_runtime", lambda: pytest.fail("model loaded"))
    with pytest.raises(ValueError, match="current encoded delivery"):
        transcribe.transcribe_episode(production.run, production.model, "revision")


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="FFmpeg unavailable")
def test_decoded_sample_count_must_match_assembly(production, recognizer):
    path = production.delivery / "assembly.json"
    assembly = json.loads(path.read_text())
    assembly["samples"] = 9
    path.write_text(json.dumps(assembly))
    with pytest.raises(ValueError, match="metadata differs"):
        transcribe.transcribe_episode(production.run, production.model, "revision")
    assert not recognizer
