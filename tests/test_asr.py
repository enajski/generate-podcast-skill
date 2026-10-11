"""Raw transcript comparison must not hide factual or pronunciation uncertainty."""

import copy
import itertools

import podcast_asr as asr
import pytest

HASH = "a" * 64


def inputs(reference, hypothesis):
    return (
        {
            "chunks": [{"id": "one", "canonical_text": reference}],
            "delivery": {"audio_sha256": HASH},
        },
        {
            "model": "raw-example-model",
            "audio_sha256": HASH,
            "segments": [{"id": "one", "text": hypothesis}],
        },
    )


def compare(reference, hypothesis):
    return asr.compare_manifest(*inputs(reference, hypothesis))


def test_exact_agreement_is_not_human_or_factual_approval():
    result = compare("The lamp costs 12 euros.", "the lamp costs 12 euros")
    assert result["status"] == "compared-unadjudicated"
    assert result["totals"]["wer"] == 0
    assert result["totals"]["reference_tokens"] == 5
    assert result["verification"] == {
        "critical_facts": "unverified",
        "human_listening": "unverified",
    }
    assert "quality" not in result


@pytest.mark.parametrize(
    ("reference", "hypothesis"),
    [
        ("It cost €9712.01", "It cost €9712.10"),
        ("It cost €9712.01", "It cost $9712.01"),
        ("The change was -12", "The change was +12"),
        ("It cost 10.5 euros", "It cost 105 euros"),
        ("It cost .5 euros", "It cost 5 euros"),
        ("It cost ,5 euros", "It cost 5 euros"),
        ("It cost 1e3 euros", "It cost 1 e3 euros"),
        ("They did not refuse", "They did refuse"),
        ("They refuse", "They they refuse"),
        ("Petiau explains", "Petio explains"),
        ("The year was 2012", "The year was twenty twelve"),
        ("They said you know", "They said"),
        ("The cost was 12%", "The cost was 12"),
    ],
)
def test_semantic_changes_remain_mismatches(reference, hypothesis):
    result = compare(reference, hypothesis)
    assert result["totals"]["errors"] > 0
    assert result["segments"][0]["mismatches"]
    assert result["segments"][0]["canonical_text"] == reference
    assert result["segments"][0]["hypothesis_text"] == hypothesis


def test_numeric_separators_symbols_and_apostrophes_are_retained():
    assert asr.normalize_tokens("€9,712.01 -3.5 +20% can't") == [
        "€",
        "9,712.01",
        "-",
        "3.5",
        "+",
        "20",
        "%",
        "can't",
    ]


def test_unicode_compatibility_case_and_whitespace_are_symmetric():
    result = compare("ＣＡＦÉ １２\u00a0euros\nStraße", "café 12 euros strasse")
    assert result["totals"]["wer"] == 0
    assert result["normalization"]["version"] == 1


def test_original_full_text_is_retained():
    canonical = "  The café cost €3.50.\nIt was not free. "
    hypothesis = "THE CAFE COST $3.50 it was free"
    result = compare(canonical, hypothesis)
    segment = result["segments"][0]
    assert segment["canonical_text"] == canonical
    assert segment["hypothesis_text"] == hypothesis
    assert result["model"] == "raw-example-model"
    assert result["audio_sha256"] == HASH
    assert all(
        "canonical_text" in span and "hypothesis_text" in span for span in segment["mismatches"]
    )


def test_substitutions_deletions_insertions_and_spans():
    result = compare("a b c d e f g", "a x c e f g z")
    assert result["totals"] == {
        "reference_tokens": 7,
        "hypothesis_tokens": 7,
        "substitutions": 1,
        "deletions": 1,
        "insertions": 1,
        "errors": 3,
        "wer": 3 / 7,
    }
    spans = result["segments"][0]["mismatches"]
    assert [(s["canonical_text"], s["hypothesis_text"]) for s in spans] == [
        ("b", "x"),
        ("d", ""),
        ("", "z"),
    ]


def test_empty_hypothesis_reports_all_deletions():
    result = compare("A real amount", "")
    assert result["totals"]["deletions"] == 3
    assert result["totals"]["wer"] == 1
    assert result["segments"][0]["mismatches"][0]["hypothesis_text"] == ""


def test_wer_can_exceed_one_without_clamping():
    result = compare("one", "one two three four")
    assert result["totals"]["wer"] == 3


def test_low_wer_with_wrong_amount_is_not_a_pass():
    reference = " ".join(["word"] * 99 + ["€100"])
    hypothesis = " ".join(["word"] * 99 + ["€1000"])
    result = compare(reference, hypothesis)
    assert 0 < result["totals"]["wer"] < 0.02
    assert result["status"] == "compared-unadjudicated"
    assert result["verification"]["critical_facts"] == "unverified"
    assert result["segments"][0]["mismatches"][0]["hypothesis_text"] == "1000"


def test_chunk_order_comes_from_manifest_and_totals_are_weighted():
    manifest, raw = inputs("one two", "one")
    manifest["chunks"].append({"id": "two", "canonical_text": "three four five"})
    raw["segments"].insert(0, {"id": "two", "text": "three four five"})
    result = asr.compare_manifest(manifest, raw)
    assert [r["id"] for r in result["segments"]] == ["one", "two"]
    assert result["totals"]["wer"] == 0.2


@pytest.mark.parametrize("value", [None, [], "", 1, True])
def test_wrong_manifest_types_fail(value):
    _, raw = inputs("one", "one")
    with pytest.raises(ValueError):
        asr.compare_manifest(value, raw)


@pytest.mark.parametrize("value", [None, "", "abc", 3, True, "g" * 64])
def test_bad_audio_hash_fails(value):
    manifest, raw = inputs("one", "one")
    raw["audio_sha256"] = value
    with pytest.raises(ValueError, match="SHA256"):
        asr.compare_manifest(manifest, raw)


def test_different_audio_hash_is_rejected():
    manifest, raw = inputs("one", "one")
    raw["audio_sha256"] = "b" * 64
    with pytest.raises(ValueError, match="does not match"):
        asr.compare_manifest(manifest, raw)


@pytest.mark.parametrize("value", [None, "", "  ", False, []])
def test_model_must_be_named(value):
    manifest, raw = inputs("one", "one")
    raw["model"] = value
    with pytest.raises(ValueError, match="model"):
        asr.compare_manifest(manifest, raw)


@pytest.mark.parametrize("change", ["missing", "extra", "duplicate", "wrong-id"])
def test_exact_segment_coverage_is_required(change):
    manifest, raw = inputs("one", "one")
    if change == "missing":
        raw["segments"] = []
    elif change == "extra":
        raw["segments"].append({"id": "extra", "text": "two"})
    elif change == "duplicate":
        manifest["chunks"].append({"id": "two", "canonical_text": "two"})
        raw["segments"].append({"id": "one", "text": "two"})
    else:
        raw["segments"][0]["id"] = "other"
    with pytest.raises(ValueError):
        asr.compare_manifest(manifest, raw)


@pytest.mark.parametrize("location", ["root", "segment"])
def test_unknown_raw_keys_are_rejected(location):
    manifest, raw = inputs("one", "one")
    target = raw if location == "root" else raw["segments"][0]
    target["cleaned"] = True
    with pytest.raises(ValueError, match="exactly"):
        asr.compare_manifest(manifest, raw)


def test_production_manifest_extra_fields_are_allowed_without_mutation():
    manifest, raw = inputs("one", "one")
    manifest["recipe"] = {"title": "Production"}
    manifest["chunks"][0]["synthesis_text"] = "won"
    before = copy.deepcopy((manifest, raw))
    assert asr.compare_manifest(manifest, raw)["totals"]["wer"] == 0
    assert (manifest, raw) == before


@pytest.mark.parametrize(
    ("reference", "hypothesis", "message"),
    [
        (" ".join(["word"] * 201), "word", "200 words"),
        ("word", " ".join(["word"] * 5001), "5000 tokens"),
        ("word", "x" * 50001, "characters"),
        ("...", "word", "comparison-token count"),
        ("+" * 1001, "word", "comparison-token count"),
    ],
)
def test_alignment_input_bounds(reference, hypothesis, message):
    with pytest.raises(ValueError, match=message):
        compare(reference, hypothesis)


def test_alignment_work_is_bounded_before_dynamic_programming(monkeypatch):
    def unexpected(*_):
        pytest.fail("alignment ran before all bounds were checked")

    monkeypatch.setattr(asr, "_alignment", unexpected)
    with pytest.raises(ValueError, match="alignment budget"):
        compare("+" * 500, " ".join(["word"] * 5000))


def test_duplicate_manifest_ids_fail():
    manifest, raw = inputs("one", "one")
    manifest["chunks"].append({"id": "one", "canonical_text": "two"})
    with pytest.raises(ValueError, match="Duplicate manifest"):
        asr.compare_manifest(manifest, raw)


def test_small_sequences_match_independent_edit_distance():
    sequences = [list(items) for n in range(1, 4) for items in itertools.product("ab", repeat=n)]
    for reference in sequences:
        for hypothesis in [[]] + sequences:
            table = [[0] * (len(hypothesis) + 1) for _ in range(len(reference) + 1)]
            for i in range(len(reference) + 1):
                table[i][0] = i
            for j in range(len(hypothesis) + 1):
                table[0][j] = j
            for i, a in enumerate(reference, 1):
                for j, b in enumerate(hypothesis, 1):
                    table[i][j] = min(
                        table[i - 1][j] + 1, table[i][j - 1] + 1, table[i - 1][j - 1] + (a != b)
                    )
            counts, spans = asr._alignment(reference, hypothesis)
            assert counts["errors"] == table[-1][-1]
            assert counts["errors"] == sum(
                s[k] for s in spans for k in ["substitutions", "deletions", "insertions"]
            )
