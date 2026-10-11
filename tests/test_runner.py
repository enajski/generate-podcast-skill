"""Exercise the command workflow without loading a speech model."""

import json
import sys
import types
from pathlib import Path

import podcast
import pytest


@pytest.fixture
def recipe_file(tmp_path):
    source = Path(__file__).parents[1] / "examples" / "episode.json"
    path = tmp_path / "recipe.json"
    path.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    return path


@pytest.fixture
def fake_backend(monkeypatch):
    class Renderer:
        def __init__(self, model):
            self.model = model

        def render(self, text, voice, seed, target):
            target.write_bytes(text.encode())
            from podcast_core import file_sha256

            return {
                "sha256": file_sha256(target),
                "sample_rate": 24000,
                "samples": 24000,
                "duration_seconds": 1.0,
                "render_seconds": 0.1,
                "peak_memory_bytes": 1,
            }

    module = types.SimpleNamespace(
        QwenRenderer=Renderer,
        backend_info=lambda *_: {"backend": "test", "model_revision": "test"},
    )
    monkeypatch.setitem(sys.modules, "podcast_qwen", module)
    return module


def test_validate_does_not_import_backend(recipe_file, monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "podcast_qwen", None)
    assert podcast.main(["validate", str(recipe_file)]) == 0
    assert len(json.loads(capsys.readouterr().out)["chunks"]) == 2


def test_render_checkpoints_and_canonical_transcript(recipe_file, fake_backend, tmp_path):
    output = tmp_path / "episode"
    result = podcast.render_episode(recipe_file, tmp_path, "test", output)
    assert result["status"] == "rendered"
    assert len(result["selected"]) == 2
    assert result["verification"]["human_listening"] == "unverified"
    assert "fictional" in (output / "transcript.md").read_text()
    assert not (output / ".podcast.lock").exists()
    assert json.loads((output / "manifest.json").read_text()) == result


def test_existing_directory_is_not_overwritten(recipe_file, fake_backend, tmp_path):
    output = tmp_path / "episode"
    output.mkdir()
    (output / "mine.txt").write_text("keep")
    with pytest.raises(ValueError, match="not empty"):
        podcast.render_episode(recipe_file, tmp_path, "test", output)
    assert (output / "mine.txt").read_text() == "keep"


def test_failure_records_interrupted_state(recipe_file, fake_backend, tmp_path, monkeypatch):
    def fail(*_):
        raise RuntimeError("test failure")

    monkeypatch.setattr(fake_backend.QwenRenderer, "render", fail)
    output = tmp_path / "episode"
    with pytest.raises(RuntimeError, match="test failure"):
        podcast.render_episode(recipe_file, tmp_path, "test", output)
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["status"] == "interrupted"
    assert manifest["selected"] == {}
    assert not (output / ".podcast.lock").exists()


def test_concurrent_writer_cannot_acquire_lock(tmp_path):
    with podcast.output_lock(tmp_path):
        with pytest.raises(FileExistsError):
            with podcast.output_lock(tmp_path):
                pytest.fail("second writer acquired lock")


def test_recheck_directory_after_slow_preflight(recipe_file, fake_backend, tmp_path):
    output = tmp_path / "episode"

    def preflight(*_):
        output.mkdir()
        (output / "mine.txt").write_text("keep")
        return {"backend": "test"}

    fake_backend.backend_info = preflight
    with pytest.raises(ValueError, match="not empty"):
        podcast.render_episode(recipe_file, tmp_path, "test", output)
    assert not (output / "manifest.json").exists()


def test_resume_reuses_all_takes_without_loading_model(recipe_file, fake_backend, tmp_path):
    output = tmp_path / "episode"
    first = podcast.render_episode(recipe_file, tmp_path, "test", output)

    def unexpected(*_):
        pytest.fail("compatible resume loaded a model")

    fake_backend.QwenRenderer = unexpected
    second = podcast.render_episode(recipe_file, tmp_path, "test", output)
    assert second["selected"] == first["selected"]
    assert len(list((output / "takes").iterdir())) == 2


def test_targeted_retake_preserves_other_audio(recipe_file, fake_backend, tmp_path):
    output = tmp_path / "episode"
    first = podcast.render_episode(recipe_file, tmp_path, "test", output)
    second = podcast.render_episode(
        recipe_file, tmp_path, "test", output, chunks=["opening-001"], retake=True, seed_offset=10
    )
    assert second["selected"]["limits-001"] == first["selected"]["limits-001"]
    assert second["selected"]["opening-001"]["seed"] != first["selected"]["opening-001"]["seed"]
    assert second["selected"]["opening-001"]["path"] != first["selected"]["opening-001"]["path"]
    assert (output / first["selected"]["opening-001"]["path"]).exists()
    assert len(list((output / "history").iterdir())) == 1


def test_pronunciation_change_invalidates_only_affected_chunk(recipe_file, fake_backend, tmp_path):
    output = tmp_path / "episode"
    first = podcast.render_episode(recipe_file, tmp_path, "test", output)
    recipe = json.loads(recipe_file.read_text())
    recipe["pronunciations"] = [{"text": "cafe", "replacement": "kaffay"}]
    recipe_file.write_text(json.dumps(recipe))
    second = podcast.render_episode(recipe_file, tmp_path, "test", output)
    assert second["selected"]["limits-001"] == first["selected"]["limits-001"]
    assert second["selected"]["opening-001"]["sha256"] != first["selected"]["opening-001"]["sha256"]
    assert "kaffay" not in (output / "transcript.md").read_text()


def test_tampered_take_is_not_reused(recipe_file, fake_backend, tmp_path):
    output = tmp_path / "episode"
    first = podcast.render_episode(recipe_file, tmp_path, "test", output)
    old = output / first["selected"]["opening-001"]["path"]
    old.write_bytes(b"damaged")
    second = podcast.render_episode(recipe_file, tmp_path, "test", output)
    assert second["selected"]["opening-001"]["path"] != first["selected"]["opening-001"]["path"]
    assert old.read_bytes() == b"damaged"


def test_subset_is_incomplete_until_resumed(recipe_file, fake_backend, tmp_path):
    output = tmp_path / "episode"
    partial = podcast.render_episode(recipe_file, tmp_path, "test", output, chunks=["opening-001"])
    assert partial["status"] == "incomplete"
    with pytest.raises(ValueError, match="fully rendered"):
        podcast.record_review(output, "audition", "Tester", "approved", "Test fixture only")
    complete = podcast.render_episode(recipe_file, tmp_path, "test", output)
    assert complete["status"] == "rendered"
    assert complete["selected"]["opening-001"] == partial["selected"]["opening-001"]


def test_audition_review_never_sets_whole_episode_approval(recipe_file, fake_backend, tmp_path):
    output = tmp_path / "episode"
    podcast.render_episode(recipe_file, tmp_path, "test", output)
    review = podcast.record_review(
        output, "audition", "Test fixture", "approved", "Not a real review"
    )
    assert review["scope"] == "audition"
    state = json.loads((output / "manifest.json").read_text())
    assert state["verification"]["human_listening"] == "unverified"
    assert state["verification"]["pronunciation"] == "unverified"
    with pytest.raises(ValueError, match="encoded delivery"):
        podcast.record_review(output, "episode", "Test fixture", "approved", "Not a real review")


def test_voice_change_invalidates_all_takes(recipe_file, fake_backend, tmp_path):
    output = tmp_path / "episode"
    first = podcast.render_episode(recipe_file, tmp_path, "test", output)
    recipe = json.loads(recipe_file.read_text())
    recipe["voice"]["instruct"] = "Read evenly."
    recipe_file.write_text(json.dumps(recipe))
    second = podcast.render_episode(recipe_file, tmp_path, "test", output)
    assert all(
        second["selected"][key]["path"] != take["path"] for key, take in first["selected"].items()
    )


def test_seed_offset_without_retake_is_rejected(recipe_file, fake_backend, tmp_path):
    with pytest.raises(ValueError, match="requires --retake"):
        podcast.render_episode(recipe_file, tmp_path, "test", tmp_path / "run", seed_offset=5)


def test_safe_file_rejects_escape(tmp_path):
    outside = tmp_path / "outside.txt"
    outside.write_text("keep")
    output = tmp_path / "episode"
    output.mkdir()
    with pytest.raises(ValueError, match="unsafe"):
        podcast.safe_file(output, "../outside.txt")


def test_resume_does_not_follow_transcript_symlink(recipe_file, fake_backend, tmp_path):
    output = tmp_path / "episode"
    podcast.render_episode(recipe_file, tmp_path, "test", output)
    outside = tmp_path / "private.txt"
    outside.write_text("keep outside")
    (output / "transcript.md").unlink()
    (output / "transcript.md").symlink_to(outside)
    with pytest.raises(ValueError, match="symlink"):
        podcast.render_episode(recipe_file, tmp_path, "test", output)
    assert outside.read_text() == "keep outside"


def test_resume_does_not_preserve_missing_encoded_delivery(recipe_file, fake_backend, tmp_path):
    output = tmp_path / "episode"
    state = podcast.render_episode(recipe_file, tmp_path, "test", output)
    state["delivery"] = {
        "audio_path": "deliveries/missing.mp3",
        "audio_sha256": "0" * 64,
        "selection_hash": podcast.selection_hash(state),
    }
    state["verification"]["human_listening"] = "approved"
    (output / "manifest.json").write_text(json.dumps(state))
    resumed = podcast.render_episode(recipe_file, tmp_path, "test", output)
    assert "delivery" not in resumed
    assert resumed["verification"]["human_listening"] == "unverified"


def test_resume_preserves_asr_evidence_links(recipe_file, fake_backend, tmp_path):
    output = tmp_path / "episode"
    state = podcast.render_episode(recipe_file, tmp_path, "test", output)
    destination = output / "deliveries" / "first"
    destination.mkdir(parents=True)
    audio = destination / "episode.mp3"
    audio.write_bytes(b"test encoded audio")
    from podcast_core import file_sha256

    audio_hash = file_sha256(audio)
    state["delivery"] = {
        "audio_path": str(audio.relative_to(output)),
        "audio_sha256": audio_hash,
        "selection_hash": podcast.selection_hash(state),
    }
    report = destination / "asr.json"
    report.write_text(json.dumps({"audio_sha256": audio_hash, "status": "compared-unadjudicated"}))
    state["asr_reports"] = [
        {"path": str(report.relative_to(output)), "sha256": file_sha256(report)}
    ]
    state["verification"]["asr"] = "compared-unadjudicated"
    (output / "manifest.json").write_text(json.dumps(state))
    resumed = podcast.render_episode(recipe_file, tmp_path, "test", output)
    assert resumed["asr_reports"] == state["asr_reports"]
    assert resumed["verification"]["asr"] == "compared-unadjudicated"


def test_pronunciation_expansion_cannot_bypass_chunk_limit(recipe_file):
    recipe = json.loads(recipe_file.read_text())
    recipe["pronunciations"] = [{"text": "cafe", "replacement": " ".join(["word"] * 100)}]
    with pytest.raises(ValueError, match="synthesis.*max_words"):
        podcast.chunk_recipe(recipe)
