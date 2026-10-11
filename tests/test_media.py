"""Verify assembly and real FFmpeg encoded-audio checks without a TTS model."""

import array
import json
import math
import shutil
import struct
import wave
from pathlib import Path

import pytest
from podcast_core import chunk_recipe, file_sha256, fingerprint, load_recipe
from podcast_media import assemble, inspect_float_audio, master_episode, read_pcm, trim_bounds


@pytest.fixture
def rendered(tmp_path):
    recipe = load_recipe(Path(__file__).parents[1] / "examples/episode.json")
    output = tmp_path / "run"
    (output / "takes").mkdir(parents=True)
    chunks = chunk_recipe(recipe)
    state = {
        "schema_version": 1,
        "status": "rendered",
        "recipe": recipe,
        "recipe_hash": fingerprint(recipe),
        "chunks": chunks,
        "selected": {},
        "verification": {
            "human_listening": "unverified",
            "asr": "unverified",
            "critical_facts": "unverified",
            "pronunciation": "unverified",
        },
    }
    for chunk in chunks:
        path = output / "takes" / (chunk["id"] + ".wav")
        rate = 24000
        samples = (
            [0] * 2400
            + [round(4000 * math.sin(2 * math.pi * 220 * n / rate)) for n in range(72000)]
            + [0] * 2400
        )
        with wave.open(str(path), "wb") as target:
            target.setnchannels(1)
            target.setsampwidth(2)
            target.setframerate(rate)
            target.writeframes(struct.pack("<" + "h" * len(samples), *samples))
        state["selected"][chunk["id"]] = {
            "path": str(path.relative_to(output)),
            "sha256": file_sha256(path),
            "samples": len(samples),
            "sample_rate": rate,
        }
    (output / "manifest.json").write_text(json.dumps(state))
    return output, state


def test_assembly_records_real_sample_boundaries(rendered, tmp_path):
    output, state = rendered
    settings = {"trim_db": -60, "edge_margin": 0.08, "paragraph_gap": 0.35, "section_gap": 1.2}
    result = assemble(output, state, tmp_path / "assembled.wav", settings)
    rate, samples = read_pcm(tmp_path / "assembled.wav")
    assert len(samples) == result["samples"]
    assert result["chapters"][1]["start_sample"] == result["chunks"][1]["start_sample"]
    assert result["chunks"][0]["following_gap_samples"] == round(1.2 * rate)
    assert result["chunks"][1]["following_gap_samples"] == 0
    assert result["time_stretch"] is False
    assert result["overlap"] is False


def test_silent_audio_is_rejected():
    with pytest.raises(ValueError, match="silent"):
        trim_bounds(array.array("h", [0] * 4800), 24000)


def test_decoded_float_integrity(tmp_path):
    path = tmp_path / "audio.f32"
    path.write_bytes(struct.pack("<ffff", 0, 0.5, -1, 1.1))
    result = inspect_float_audio(path)
    assert result["decoded_samples"] == 4
    assert result["clipped_samples"] == 2
    path.write_bytes(struct.pack("<f", float("nan")))
    with pytest.raises(ValueError, match="nonfinite"):
        inspect_float_audio(path)


def test_missing_tool_fails_before_delivery(rendered):
    output, _ = rendered
    with pytest.raises(ValueError, match="unavailable"):
        master_episode(output, ffmpeg="missing-podcast-ffmpeg")
    assert not (output / "deliveries").exists()


@pytest.mark.parametrize(
    "option,value",
    [("section_gap", -1), ("paragraph_gap", float("nan")), ("target_lufs", 0), ("true_peak", 0)],
)
def test_invalid_mastering_settings(rendered, option, value):
    with pytest.raises(ValueError, match="Invalid mastering"):
        master_episode(rendered[0], **{option: value})


@pytest.mark.skipif(
    not shutil.which("ffmpeg") or not shutil.which("ffprobe"), reason="FFmpeg tools absent"
)
def test_real_encoded_mp3_and_review_binding(rendered):
    from podcast import record_review

    output, state = rendered
    delivery = master_episode(output)
    assert delivery["report"]["status"] == "passed"
    assert all(delivery["report"]["checks"].values())
    assert delivery["report"]["human_listening"] == "unverified"
    assert (output / delivery["path"] / "notes.md").is_file()
    assert not (output / delivery["path"] / "decoded.f32").exists()
    review = record_review(output, "episode", "Test fixture", "approved", "Simulated review only")
    assert review["audio_sha256"] == delivery["audio_sha256"]
    newer = master_episode(output)
    assert newer["path"] != delivery["path"]
    assert (output / delivery["audio_path"]).exists()
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["verification"]["human_listening"] == "unverified"
    assert manifest["reviews"][0] == review


def test_incomplete_state_cannot_master(rendered):
    output, state = rendered
    state["status"] = "incomplete"
    (output / "manifest.json").write_text(json.dumps(state))
    with pytest.raises(ValueError, match="fully rendered"):
        master_episode(output)


def test_tampered_audio_cannot_master(rendered):
    output, state = rendered
    path = output / state["selected"]["opening-001"]["path"]
    path.write_bytes(b"bad audio")
    settings = {"trim_db": -60, "edge_margin": 0.08, "paragraph_gap": 0.35, "section_gap": 1.2}
    with pytest.raises(ValueError, match="hash changed"):
        assemble(output, state, output / "bad.wav", settings)


def test_malformed_wav_has_clear_error(tmp_path):
    path = tmp_path / "bad.wav"
    path.write_bytes(b"not a wave file")
    with pytest.raises(ValueError, match="WAV"):
        read_pcm(path)


@pytest.mark.skipif(
    not shutil.which("ffmpeg") or not shutil.which("ffprobe"), reason="FFmpeg tools absent"
)
def test_remastered_snapshot_does_not_inherit_old_listening_approval(rendered):
    from podcast import record_review

    output, _ = rendered
    master_episode(output)
    record_review(output, "episode", "Test fixture", "approved", "Simulated review only")
    delivery = master_episode(output)
    snapshot = json.loads((output / delivery["path"] / "manifest.json").read_text())
    assert snapshot["verification"]["human_listening"] == "unverified"
