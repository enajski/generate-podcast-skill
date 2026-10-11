"""Portable recipe invariants; no model, network, audio package or Vis imports."""

import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "podcast_core", Path(__file__).parents[1] / "scripts" / "podcast_core.py"
)
core = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(core)


@pytest.fixture
def recipe():
    return {
        "schema_version": 1,
        "title": "A neutral example",
        "sections": [
            {
                "id": "opening",
                "heading": "Start here",
                "paragraphs": ["This is the first paragraph.", "This is the second paragraph."],
            }
        ],
        "sources": [
            {
                "id": "s1",
                "title": "Example source",
                "url": "https://example.org/source",
                "note": "Example only; not research evidence.",
            }
        ],
    }


def test_defaults_are_explicit_neutral_and_detached(recipe):
    before = copy.deepcopy(recipe)
    validated = core.validate_recipe(recipe)
    assert recipe == before
    assert validated["max_words"] == 95
    assert validated["seed_base"] == 1000
    assert validated["pronunciations"] == []
    assert validated["critical_facts"] == []
    assert validated["voice"] == core.VOICE_DEFAULTS
    assert "Transatlantic" not in validated["voice"]["instruct"]
    validated["sections"][0]["paragraphs"].append("New paragraph.")
    validated["sources"][0]["note"] = "Changed."
    validated["voice"]["voice"] = "Other"
    assert recipe == before
    assert core.VOICE_DEFAULTS["voice"] == "Aiden"


@pytest.mark.parametrize("bad", [None, [], "recipe", 0, True])
def test_recipe_requires_an_object(bad):
    with pytest.raises(core.RecipeError, match="recipe must be an object"):
        core.validate_recipe(bad)


@pytest.mark.parametrize("key", ["schema_version", "title", "sections", "sources"])
def test_required_fields(recipe, key):
    del recipe[key]
    with pytest.raises(core.RecipeError, match="missing"):
        core.validate_recipe(recipe)


@pytest.mark.parametrize("value", [True, False, 1.0, "1", 0, 2, None])
def test_schema_version_is_exact_integer_one(recipe, value):
    recipe["schema_version"] = value
    with pytest.raises(core.RecipeError, match="schema_version"):
        core.validate_recipe(recipe)


@pytest.mark.parametrize("value", ["", " \n\t", 0, False, None, []])
def test_title_is_nonblank_text(recipe, value):
    recipe["title"] = value
    with pytest.raises(core.RecipeError, match="title"):
        core.validate_recipe(recipe)


def test_unknown_fields_are_rejected_at_each_level(recipe):
    for target in [recipe, recipe["sections"][0], recipe["sources"][0]]:
        target["typo"] = "unexpected"
        with pytest.raises(core.RecipeError, match="unsupported fields: typo"):
            core.validate_recipe(recipe)
        del target["typo"]
    recipe[7] = "bad key"
    with pytest.raises(core.RecipeError, match="keys must be strings"):
        core.validate_recipe(recipe)


@pytest.mark.parametrize("sections", [[], None, {}, [None]])
def test_empty_or_invalid_sections_fail(recipe, sections):
    recipe["sections"] = sections
    with pytest.raises(core.RecipeError, match="sections"):
        core.validate_recipe(recipe)


@pytest.mark.parametrize("paragraphs", [[], [""], [" \n"], [None], [2], "text", None])
def test_empty_narration_is_rejected_before_rendering(recipe, paragraphs):
    # The original session helper emitted a zero-word chunk for an empty section.
    recipe["sections"][0]["paragraphs"] = paragraphs
    with pytest.raises(core.RecipeError, match="paragraphs"):
        core.chunk_recipe(recipe)


def test_oversized_paragraph_is_rejected_with_repair_instruction(recipe):
    # The original 95-word grouping helper silently accepted this 120-word paragraph.
    recipe["sections"][0]["paragraphs"] = [" ".join(["word"] * 120)]
    with pytest.raises(core.RecipeError, match="120 words.*max_words=95; split the paragraph"):
        core.validate_recipe(recipe)


@pytest.mark.parametrize(
    "field,value",
    [
        ("max_words", 0),
        ("max_words", 201),
        ("max_words", True),
        ("max_words", 5.0),
        ("max_words", "95"),
        ("max_words", float("inf")),
        ("seed_base", -1),
        ("seed_base", False),
        ("seed_base", 1.0),
    ],
)
def test_integer_bounds(recipe, field, value):
    recipe[field] = value
    with pytest.raises(core.RecipeError, match=field):
        core.validate_recipe(recipe)


@pytest.mark.parametrize("section_id", ["../outside", "a/b", "a\\b", "UPPER", "two words", "", "é"])
def test_section_ids_are_path_safe(recipe, section_id):
    recipe["sections"][0]["id"] = section_id
    with pytest.raises(core.RecipeError, match=r"sections\[0\].id"):
        core.validate_recipe(recipe)


def test_duplicate_section_and_source_ids_fail(recipe):
    recipe["sections"].append(copy.deepcopy(recipe["sections"][0]))
    with pytest.raises(core.RecipeError, match="duplicate section id"):
        core.validate_recipe(recipe)
    recipe["sections"].pop()
    recipe["sources"].append(copy.deepcopy(recipe["sources"][0]))
    with pytest.raises(core.RecipeError, match="duplicate source id"):
        core.validate_recipe(recipe)


@pytest.mark.parametrize(
    "url",
    [
        "",
        "ftp://example.org/a",
        "file:///tmp/a",
        "https:///a",
        "https://",
        "https://example.org/has space",
        "https://example.org/line\nbreak",
        "https://[bad",
        "https://example.org:bad/",
        "https://user:secret@example.org/",
    ],
)
def test_source_urls_are_http_and_safe(recipe, url):
    recipe["sources"][0]["url"] = url
    with pytest.raises(core.RecipeError, match="url"):
        core.validate_recipe(recipe)


@pytest.mark.parametrize("value", [None, {}, "sources", [None]])
def test_source_collection_type(recipe, value):
    recipe["sources"] = value
    with pytest.raises(core.RecipeError, match="sources"):
        core.validate_recipe(recipe)


def test_empty_sources_are_allowed_and_disclosed(recipe):
    recipe["sources"] = []
    text = core.transcript(recipe)
    assert "No sources were supplied" in text
    assert "not been source-verified" in text


@pytest.mark.parametrize(
    "field,value",
    [
        ("temperature", 0),
        ("temperature", -0.1),
        ("temperature", True),
        ("temperature", float("nan")),
        ("temperature", float("inf")),
        ("temperature", 10**1000),
        ("top_p", 0),
        ("top_p", 1.01),
        ("top_p", "0.9"),
        ("top_k", 0),
        ("top_k", 1.5),
        ("top_k", True),
        ("max_tokens", 0),
        ("max_tokens", False),
        ("repetition_penalty", 0),
        ("repetition_penalty", float("-inf")),
        ("voice", ""),
        ("lang_code", None),
        ("instruct", "  "),
    ],
)
def test_invalid_voice_controls_fail(recipe, field, value):
    recipe["voice"] = {field: value}
    with pytest.raises(core.RecipeError, match=field):
        core.validate_recipe(recipe)


@pytest.mark.parametrize("field", ["speed", "pitch", "unknown", "verbose"])
def test_unsupported_controls_are_not_accepted_as_noops(recipe, field):
    recipe["voice"] = {field: 1}
    with pytest.raises(core.RecipeError, match="unsupported fields"):
        core.validate_recipe(recipe)


def test_valid_voice_overrides_preserve_other_defaults(recipe):
    recipe["voice"] = {"voice": "Ryan", "top_p": 1, "top_k": 1, "max_tokens": 42}
    actual = core.validate_recipe(recipe)["voice"]
    assert actual["voice"] == "Ryan"
    assert actual["max_tokens"] == 42
    assert actual["temperature"] == 0.65


@pytest.mark.parametrize(
    "maps",
    [
        None,
        {},
        [None],
        [{"text": "", "replacement": "sound"}],
        [{"text": "term", "replacement": " "}],
        [{"text": "term"}],
        [{"text": "term", "replacement": "sound", "case_sensitive": False}],
        [{"text": "term", "replacement": "one"}, {"text": "term", "replacement": "two"}],
    ],
)
def test_invalid_pronunciation_maps_fail(recipe, maps):
    recipe["pronunciations"] = maps
    with pytest.raises(core.RecipeError):
        core.validate_recipe(recipe)


def test_chunks_group_complete_paragraphs_without_changing_text(recipe):
    recipe["max_words"] = 10
    recipe["sections"][0]["paragraphs"] = [
        "  One two.  ",
        "Three four.",
        "Five six seven eight nine ten eleven.",
    ]
    chunks = core.chunk_recipe(recipe)
    assert [item["id"] for item in chunks] == ["opening-001", "opening-002"]
    assert chunks[0]["canonical_text"] == "  One two.  \n\nThree four."
    assert chunks[1]["canonical_text"] == "Five six seven eight nine ten eleven."
    assert [item["section_end"] for item in chunks] == [False, True]
    assert all(item["synthesis_text"] == item["canonical_text"] for item in chunks)
    assert set(chunks[0]) == {
        "id",
        "section_id",
        "heading",
        "canonical_text",
        "synthesis_text",
        "section_end",
    }


def test_section_local_chunk_ids_survive_an_unrelated_section_edit(recipe):
    recipe["sections"].append({"id": "closing", "heading": "End", "paragraphs": ["Last words."]})
    before = core.chunk_recipe(recipe)[-1]
    recipe["sections"][0]["paragraphs"].extend(["More words."] * 100)
    after = core.chunk_recipe(recipe)[-1]
    assert before == after
    assert after["id"] == "closing-001"


def test_pronunciation_is_case_sensitive_whole_term_longest_first_and_non_chaining(recipe):
    original = "New York, York, york, Yorkshire, éYork, Yorké; C++ and A."
    recipe["sections"][0]["paragraphs"] = [original]
    recipe["pronunciations"] = [
        {"text": "York", "replacement": "YORK"},
        {"text": "New York", "replacement": "newyork"},
        {"text": "C++", "replacement": "see plus plus"},
        {"text": "A", "replacement": "York"},
    ]
    chunk = core.chunk_recipe(recipe)[0]
    assert chunk["canonical_text"] == original
    assert (
        chunk["synthesis_text"]
        == "newyork, YORK, york, Yorkshire, éYork, Yorké; see plus plus and York."
    )
    assert original in core.transcript(recipe)
    assert "newyork" not in core.transcript(recipe)


def test_unrelated_pronunciation_does_not_change_chunk_inputs(recipe):
    before = core.chunk_recipe(recipe)
    recipe["pronunciations"] = [{"text": "absent", "replacement": "missing"}]
    assert core.chunk_recipe(recipe) == before
    assert core.fingerprint(core.chunk_recipe(recipe)[0]) == core.fingerprint(before[0])


def test_transcript_has_canonical_source_and_verification_context(recipe):
    recipe["description"] = "An explicit example, not a factual report."
    recipe["critical_facts"] = ["The example must not become claimed research."]
    text = core.transcript(recipe)
    for fragment in [
        recipe["title"],
        recipe["description"],
        recipe["sections"][0]["heading"],
        recipe["sections"][0]["paragraphs"][0],
        "https://example.org/source",
        recipe["sources"][0]["note"],
        recipe["critical_facts"][0],
        "one synthetic narrator",
        "does not establish audio accuracy",
    ]:
        assert fragment in text
    assert text.endswith("\n")


@pytest.mark.parametrize("facts", ["fact", [""], [3], None])
def test_invalid_critical_facts_fail(recipe, facts):
    recipe["critical_facts"] = facts
    with pytest.raises(core.RecipeError, match="critical_facts"):
        core.validate_recipe(recipe)


def test_load_recipe_utf8_and_strict_json(tmp_path, recipe):
    recipe["title"] = "Récit"
    path = tmp_path / "recipe.json"
    path.write_text(json.dumps(recipe, ensure_ascii=False), encoding="utf-8")
    assert core.load_recipe(path) == core.validate_recipe(recipe)
    for bad in [
        '{"title":"one","title":"two"}',
        '{"nested":{"x":1,"x":2}}',
        '{"value":NaN}',
        '{"value":Infinity}',
        "{bad",
        "[]",
    ]:
        path.write_text(bad, encoding="utf-8")
        with pytest.raises(core.RecipeError):
            core.load_recipe(path)
    path.write_bytes(b"\xff")
    with pytest.raises(core.RecipeError, match="cannot load"):
        core.load_recipe(path)
    with pytest.raises(core.RecipeError, match="cannot load"):
        core.load_recipe(tmp_path / "missing.json")


def test_fingerprint_is_json_order_independent_and_content_sensitive():
    assert core.fingerprint({"b": [1, "é"], "a": True}) == core.fingerprint(
        {"a": True, "b": [1, "é"]}
    )
    assert core.fingerprint("canonical") != core.fingerprint("synthesis")
    assert core.fingerprint([1, 2]) != core.fingerprint([2, 1])
    with pytest.raises(ValueError):
        core.fingerprint(float("nan"))
    with pytest.raises(TypeError):
        core.fingerprint({"set"})


def test_file_hash_is_streamed_content_hash(tmp_path):
    path = tmp_path / "take.wav"
    content = b"not audio" * 200_000
    path.write_bytes(content)
    assert core.file_sha256(path) == hashlib.sha256(content).hexdigest()


def test_atomic_json_round_trip_and_no_orphan_temp(tmp_path):
    path = tmp_path / "output" / "manifest.json"
    core.atomic_json(path, {"title": "Récit", "complete": False})
    assert json.loads(path.read_text(encoding="utf-8")) == {"title": "Récit", "complete": False}
    assert path.read_text(encoding="utf-8").endswith("\n")
    core.atomic_json(path, {"complete": True})
    assert json.loads(path.read_text()) == {"complete": True}
    assert list(path.parent.iterdir()) == [path]


def test_invalid_json_does_not_replace_existing_checkpoint(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text('{"original":true}\n')
    with pytest.raises(ValueError):
        core.atomic_json(path, {"bad": float("nan")})
    assert path.read_text() == '{"original":true}\n'
    assert list(tmp_path.iterdir()) == [path]


def test_failed_atomic_replace_preserves_checkpoint_and_cleans_temp(tmp_path, monkeypatch):
    path = tmp_path / "manifest.json"
    path.write_text('{"original":true}\n')

    def refuse_replace(source, destination):
        assert Path(source).parent == path.parent
        assert Path(destination) == path
        raise OSError("simulated replace failure")

    monkeypatch.setattr(core.os, "replace", refuse_replace)
    with pytest.raises(OSError, match="simulated replace failure"):
        core.atomic_json(path, {"replacement": True})
    assert path.read_text() == '{"original":true}\n'
    assert list(tmp_path.iterdir()) == [path]
