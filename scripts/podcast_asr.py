"""Compare raw ASR with canonical narration without treating agreement as approval."""

from __future__ import annotations

import re
import unicodedata

MAX_TEXT_CHARACTERS = 50_000
MAX_REFERENCE_WORDS = 200
MAX_REFERENCE_TOKENS = 1_000
MAX_HYPOTHESIS_TOKENS = 5_000
MAX_SEGMENTS = 1_000
MAX_SEGMENT_CELLS = 2_000_000
MAX_TOTAL_CELLS = 20_000_000
NORMALIZATION = {
    "version": 1,
    "description": (
        "NFKC, casefold and whitespace collapse. Keep words, numbers with decimal or "
        "grouping separators, interior apostrophes, currency symbols, numeric signs "
        "and percent signs. Ignore other punctuation. Do not expand numbers or years, "
        "alias names, or remove fillers and repetitions. Mismatch text contains "
        "normalized tokens; original full texts are retained."
    ),
    "count_unit": "normalized tokens, including retained symbols",
    "wer_definition": "(substitutions + deletions + insertions) / reference_tokens",
}
_NUMBER = r"(?:\d+(?:[.,]\d+)*|[.,]\d+)(?:e[+-]?\d+)?"
_TOKEN = re.compile(_NUMBER + r"|[^\W\d_]+(?:['’][^\W\d_]+)*|[^\s]", re.UNICODE)
_NUMERIC = re.compile(_NUMBER)
_HASH = re.compile(r"[0-9a-fA-F]{64}\Z")


def _text(value, location, *, nonblank=True):
    if not isinstance(value, str) or (nonblank and not value.strip()):
        raise ValueError(f"{location} must be a {'nonblank ' if nonblank else ''}string")
    if len(value) > MAX_TEXT_CHARACTERS:
        raise ValueError(f"{location} exceeds {MAX_TEXT_CHARACTERS} characters")
    return value


def _digest(value, location):
    if not isinstance(value, str) or _HASH.fullmatch(value) is None:
        raise ValueError(f"{location} must be a 64-character hexadecimal SHA256")
    return value.lower()


def _object(value, fields, location):
    if not isinstance(value, dict):
        raise ValueError(f"{location} must be an object")
    if set(value) != set(fields):
        raise ValueError(f"{location} must contain exactly: {', '.join(fields)}")


def normalize_tokens(text):
    """Keep semantic symbols without forgiving number, name or repetition changes."""
    normalized = " ".join(unicodedata.normalize("NFKC", text).casefold().split())
    tokens = []
    for match in _TOKEN.finditer(normalized):
        token = match.group()
        if (
            token[0].isalnum()
            or _NUMERIC.fullmatch(token) is not None
            or token in {"+", "-", "−", "%", "‰", "‱"}
            or unicodedata.category(token[0]) == "Sc"
        ):
            tokens.append(token)
    return tokens


def _alignment(reference, hypothesis):
    """Return minimal edit counts and contiguous mismatch spans using bounded memory."""
    previous = list(range(len(hypothesis) + 1))
    trace = [bytearray([3]) * (len(hypothesis) + 1)]
    for row, token in enumerate(reference, start=1):
        current = [row] + [0] * len(hypothesis)
        directions = bytearray(len(hypothesis) + 1)
        directions[0] = 2
        for column, other in enumerate(hypothesis, start=1):
            if token == other:
                current[column] = previous[column - 1]
            else:
                choices = (
                    previous[column - 1] + 1,
                    previous[column] + 1,
                    current[column - 1] + 1,
                )
                current[column] = min(choices)
                directions[column] = choices.index(current[column]) + 1
        trace.append(directions)
        previous = current
    row, column = len(reference), len(hypothesis)
    operations = []
    while row or column:
        direction = trace[row][column]
        if direction in (0, 1):
            operations.append((direction, row - 1, row, column - 1, column))
            row -= 1
            column -= 1
        elif direction == 2:
            operations.append((direction, row - 1, row, column, column))
            row -= 1
        else:
            operations.append((direction, row, row, column - 1, column))
            column -= 1
    operations.reverse()
    counts = {"substitutions": 0, "deletions": 0, "insertions": 0}
    names = {1: "substitutions", 2: "deletions", 3: "insertions"}
    spans = []
    active = None
    for direction, ref_start, ref_end, hyp_start, hyp_end in operations:
        if direction == 0:
            if active is not None:
                spans.append(active)
                active = None
            continue
        counts[names[direction]] += 1
        if active is None:
            active = {
                "reference_start": ref_start,
                "reference_end": ref_end,
                "hypothesis_start": hyp_start,
                "hypothesis_end": hyp_end,
                "substitutions": 0,
                "deletions": 0,
                "insertions": 0,
            }
        active["reference_end"] = ref_end
        active["hypothesis_end"] = hyp_end
        active[names[direction]] += 1
    if active is not None:
        spans.append(active)
    for span in spans:
        span["canonical_text"] = " ".join(
            reference[span["reference_start"] : span["reference_end"]]
        )
        span["hypothesis_text"] = " ".join(
            hypothesis[span["hypothesis_start"] : span["hypothesis_end"]]
        )
    errors = sum(counts.values())
    return {
        "reference_tokens": len(reference),
        "hypothesis_tokens": len(hypothesis),
        **counts,
        "errors": errors,
        "wer": errors / len(reference),
    }, spans


def compare_manifest(manifest, raw_transcript):
    """Compare one raw recognizer with the exact encoded delivery and all its chunks.

    Extra manifest fields are allowed because the production manifest owns them.
    The raw-transcript object and its segment objects have strict schemas.
    WER counts normalized tokens, including retained signs and currency symbols.
    It can exceed one. All comparisons remain unadjudicated, even with zero edits.
    """
    if not isinstance(manifest, dict):
        raise ValueError("manifest must be an object")
    delivery = manifest.get("delivery")
    if not isinstance(delivery, dict):
        raise ValueError("manifest.delivery must be an object")
    audio_hash = _digest(delivery.get("audio_sha256"), "manifest.delivery.audio_sha256")
    chunks = manifest.get("chunks")
    if not isinstance(chunks, list) or not 1 <= len(chunks) <= MAX_SEGMENTS:
        raise ValueError(f"manifest.chunks must contain 1..{MAX_SEGMENTS} segments")
    references = {}
    for chunk in chunks:
        if not isinstance(chunk, dict):
            raise ValueError("manifest chunk must be an object")
        identifier = _text(chunk.get("id"), "manifest chunk id")
        if identifier in references:
            raise ValueError(f"Duplicate manifest chunk id: {identifier}")
        text = _text(chunk.get("canonical_text"), f"canonical text for {identifier}")
        if len(text.split()) > MAX_REFERENCE_WORDS:
            raise ValueError(f"Canonical chunk {identifier} exceeds {MAX_REFERENCE_WORDS} words")
        references[identifier] = text
    _object(raw_transcript, ("model", "audio_sha256", "segments"), "raw transcript")
    model = _text(raw_transcript["model"], "raw transcript model")
    if _digest(raw_transcript["audio_sha256"], "raw transcript audio_sha256") != audio_hash:
        raise ValueError("Raw transcript audio SHA256 does not match the encoded delivery")
    raw_segments = raw_transcript["segments"]
    if not isinstance(raw_segments, list) or len(raw_segments) != len(references):
        raise ValueError("Raw segments must cover every manifest chunk exactly once")
    hypotheses = {}
    for segment in raw_segments:
        _object(segment, ("id", "text"), "raw segment")
        identifier = _text(segment["id"], "raw segment id")
        if identifier in hypotheses:
            raise ValueError(f"Duplicate raw segment id: {identifier}")
        hypotheses[identifier] = _text(
            segment["text"], f"raw text for {identifier}", nonblank=False
        )
    if set(hypotheses) != set(references):
        raise ValueError("Raw segment IDs do not match the manifest chunk IDs")
    prepared = []
    total_cells = 0
    for identifier, canonical in references.items():
        reference = normalize_tokens(canonical)
        hypothesis = normalize_tokens(hypotheses[identifier])
        if not reference or len(reference) > MAX_REFERENCE_TOKENS:
            raise ValueError(f"Canonical chunk {identifier} has an invalid comparison-token count")
        if len(hypothesis) > MAX_HYPOTHESIS_TOKENS:
            raise ValueError(f"Raw chunk {identifier} exceeds {MAX_HYPOTHESIS_TOKENS} tokens")
        cells = (len(reference) + 1) * (len(hypothesis) + 1)
        total_cells += cells
        if cells > MAX_SEGMENT_CELLS or total_cells > MAX_TOTAL_CELLS:
            raise ValueError("Transcript exceeds the bounded alignment budget")
        prepared.append((identifier, canonical, hypotheses[identifier], reference, hypothesis))
    segments = []
    totals = {
        "reference_tokens": 0,
        "hypothesis_tokens": 0,
        "substitutions": 0,
        "deletions": 0,
        "insertions": 0,
        "errors": 0,
    }
    for identifier, canonical, raw, reference, hypothesis in prepared:
        counts, mismatches = _alignment(reference, hypothesis)
        segments.append(
            {
                "id": identifier,
                "canonical_text": canonical,
                "hypothesis_text": raw,
                **counts,
                "mismatches": mismatches,
            }
        )
        for name in totals:
            totals[name] += counts[name]
    totals["wer"] = totals["errors"] / totals["reference_tokens"]
    return {
        "status": "compared-unadjudicated",
        "normalization": dict(NORMALIZATION),
        "model": model,
        "audio_sha256": audio_hash,
        "segments": segments,
        "totals": totals,
        "verification": {"critical_facts": "unverified", "human_listening": "unverified"},
    }
