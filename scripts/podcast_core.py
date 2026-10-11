"""Validate podcast recipes and preserve canonical text without host dependencies."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from pathlib import Path
from urllib.parse import urlsplit


class RecipeError(ValueError):
    """The recipe cannot be used without changing or clarifying its inputs."""


RECIPE_FIELDS = {
    "schema_version",
    "title",
    "description",
    "sections",
    "sources",
    "voice",
    "pronunciations",
    "max_words",
    "seed_base",
    "critical_facts",
}
VOICE_DEFAULTS = {
    "voice": "Aiden",
    "lang_code": "English",
    "instruct": "Read in a clear, neutral, conversational voice.",
    "temperature": 0.65,
    "top_p": 0.9,
    "top_k": 50,
    "repetition_penalty": 1.05,
    "max_tokens": 1800,
}


def _object(value, allowed, required, location):
    if not isinstance(value, dict):
        raise RecipeError(f"{location} must be an object")
    if any(not isinstance(key, str) for key in value):
        raise RecipeError(f"{location} keys must be strings")
    unknown = set(value) - allowed
    if unknown:
        raise RecipeError(f"{location} has unsupported fields: {', '.join(sorted(unknown))}")
    missing = required - set(value)
    if missing:
        raise RecipeError(f"{location} is missing: {', '.join(sorted(missing))}")


def _text(value, location, *, allow_empty=False):
    if not isinstance(value, str) or (not allow_empty and not value.strip()):
        adjective = "a string" if allow_empty else "a nonblank string"
        raise RecipeError(f"{location} must be {adjective}")
    return value


def _integer(value, location, minimum, maximum=None):
    if type(value) is not int or value < minimum:
        raise RecipeError(f"{location} must be an integer >= {minimum}")
    if maximum is not None and value > maximum:
        raise RecipeError(f"{location} must be an integer <= {maximum}")
    return value


def _positive(value, location, *, maximum=None):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RecipeError(f"{location} must be a finite positive number")
    try:
        finite = math.isfinite(value)
    except OverflowError:
        finite = False
    if not finite or value <= 0:
        raise RecipeError(f"{location} must be a finite positive number")
    if maximum is not None and value > maximum:
        raise RecipeError(f"{location} must be <= {maximum}")
    return value


def _list(value, location, *, nonempty=False):
    if not isinstance(value, list) or (nonempty and not value):
        adjective = "a nonempty list" if nonempty else "a list"
        raise RecipeError(f"{location} must be {adjective}")
    return value


def _voice(value):
    _object(value, set(VOICE_DEFAULTS), set(), "voice")
    result = {**VOICE_DEFAULTS, **value}
    for key in ("voice", "lang_code", "instruct"):
        _text(result[key], f"voice.{key}")
    for key in ("temperature", "repetition_penalty"):
        _positive(result[key], f"voice.{key}")
    _positive(result["top_p"], "voice.top_p", maximum=1)
    for key in ("top_k", "max_tokens"):
        _integer(result[key], f"voice.{key}", 1)
    return result


def _sections(value, max_words):
    sections = []
    seen = set()
    for index, section in enumerate(_list(value, "sections", nonempty=True)):
        location = f"sections[{index}]"
        _object(section, {"id", "heading", "paragraphs"}, {"id", "heading", "paragraphs"}, location)
        section_id = _text(section["id"], f"{location}.id")
        if re.fullmatch(r"[a-z0-9-]+", section_id) is None:
            raise RecipeError(f"{location}.id must match [a-z0-9-]+")
        if section_id in seen:
            raise RecipeError(f"duplicate section id: {section_id}")
        seen.add(section_id)
        heading = _text(section["heading"], f"{location}.heading")
        paragraphs = []
        for paragraph_index, paragraph in enumerate(
            _list(section["paragraphs"], f"{location}.paragraphs", nonempty=True)
        ):
            paragraph_location = f"{location}.paragraphs[{paragraph_index}]"
            _text(paragraph, paragraph_location)
            words = len(paragraph.split())
            if words > max_words:
                raise RecipeError(
                    f"{paragraph_location} has {words} words, exceeding max_words="
                    f"{max_words}; split the paragraph before rendering"
                )
            paragraphs.append(paragraph)
        sections.append({"id": section_id, "heading": heading, "paragraphs": paragraphs})
    return sections


def _sources(value):
    sources = []
    seen = set()
    for index, source in enumerate(_list(value, "sources")):
        location = f"sources[{index}]"
        _object(source, {"id", "title", "url", "note"}, {"id", "title", "url", "note"}, location)
        source_id = _text(source["id"], f"{location}.id")
        if source_id in seen:
            raise RecipeError(f"duplicate source id: {source_id}")
        seen.add(source_id)
        title = _text(source["title"], f"{location}.title")
        url = _text(source["url"], f"{location}.url")
        try:
            parsed = urlsplit(url)
            valid_url = (
                parsed.scheme in {"http", "https"}
                and parsed.hostname
                and not any(character.isspace() or ord(character) < 32 for character in url)
                and parsed.username is None
                and parsed.password is None
            )
            # Accessing port rejects malformed port numbers without a network request.
            _ = parsed.port
        except ValueError:
            valid_url = False
        if not valid_url:
            raise RecipeError(f"{location}.url must be an HTTP(S) URL without credentials")
        note = _text(source["note"], f"{location}.note", allow_empty=True)
        sources.append({"id": source_id, "title": title, "url": url, "note": note})
    return sources


def _pronunciations(value):
    mappings = []
    seen = set()
    for index, mapping in enumerate(_list(value, "pronunciations")):
        location = f"pronunciations[{index}]"
        _object(mapping, {"text", "replacement"}, {"text", "replacement"}, location)
        text = _text(mapping["text"], f"{location}.text")
        replacement = _text(mapping["replacement"], f"{location}.replacement")
        if text in seen:
            raise RecipeError(f"duplicate pronunciation text: {text}")
        seen.add(text)
        mappings.append({"text": text, "replacement": replacement})
    return mappings


def validate_recipe(value):
    """Return a detached, validated recipe with explicit neutral defaults.

    This function does not change input strings, import a speech backend, read a
    model, download anything, or change the supplied dictionary.
    """
    _object(value, RECIPE_FIELDS, {"schema_version", "title", "sections", "sources"}, "recipe")
    if type(value["schema_version"]) is not int or value["schema_version"] != 1:
        raise RecipeError("schema_version must be the integer 1")
    max_words = _integer(value.get("max_words", 95), "max_words", 1, 200)
    result = {
        "schema_version": 1,
        "title": _text(value["title"], "title"),
        "description": _text(value.get("description", ""), "description", allow_empty=True),
        "sections": _sections(value["sections"], max_words),
        "sources": _sources(value["sources"]),
        "voice": _voice(value.get("voice", {})),
        "pronunciations": _pronunciations(value.get("pronunciations", [])),
        "max_words": max_words,
        "seed_base": _integer(value.get("seed_base", 1000), "seed_base", 0),
        "critical_facts": [
            _text(fact, f"critical_facts[{index}]")
            for index, fact in enumerate(_list(value.get("critical_facts", []), "critical_facts"))
        ],
    }
    return result


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise RecipeError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def _reject_constant(value):
    raise RecipeError(f"non-finite JSON value: {value}")


def load_recipe(path):
    """Read a strict UTF-8 JSON recipe, rejecting duplicate keys and NaN values."""
    try:
        content = Path(path).read_text(encoding="utf-8")
        value = json.loads(
            content, object_pairs_hook=_unique_object, parse_constant=_reject_constant
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RecipeError(f"cannot load recipe {path}: {error}") from error
    return validate_recipe(value)


def chunk_recipe(recipe):
    """Group complete paragraphs and apply exact, non-chaining speech cues.

    Chunk identifiers are local to their section. Canonical strings remain
    unchanged. Synthesis text is the chunk-level input for render cache keys;
    unrelated entries in the pronunciation map must not invalidate a take.
    """
    recipe = validate_recipe(recipe)
    replacements = {item["text"]: item["replacement"] for item in recipe["pronunciations"]}
    pattern = None
    if replacements:
        terms = sorted(replacements, key=lambda term: (-len(term), term))
        pattern = re.compile(
            r"(?<!\w)(?:" + "|".join(re.escape(term) for term in terms) + r")(?!\w)"
        )
    chunks = []
    for section in recipe["sections"]:
        groups = []
        current = []
        word_count = 0
        for paragraph in section["paragraphs"]:
            words = len(paragraph.split())
            if current and word_count + words > recipe["max_words"]:
                groups.append(current)
                current = []
                word_count = 0
            current.append(paragraph)
            word_count += words
        groups.append(current)
        for index, paragraphs in enumerate(groups, start=1):
            canonical_text = "\n\n".join(paragraphs)
            synthesis_text = (
                pattern.sub(lambda match: replacements[match.group(0)], canonical_text)
                if pattern is not None
                else canonical_text
            )
            if len(synthesis_text.split()) > recipe["max_words"]:
                raise RecipeError(
                    f"{section['id']}: synthesis text exceeds max_words after pronunciation "
                    "replacement; split the source paragraphs or shorten the replacement"
                )
            chunks.append(
                {
                    "id": f"{section['id']}-{index:03d}",
                    "section_id": section["id"],
                    "heading": section["heading"],
                    "canonical_text": canonical_text,
                    "synthesis_text": synthesis_text,
                    "section_end": index == len(groups),
                }
            )
    return chunks


def fingerprint(value):
    """Hash canonical UTF-8 JSON, refusing non-finite and non-JSON values."""
    encoded = json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path):
    """Hash file contents without loading the entire file into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    """Replace a JSON checkpoint only after a complete same-directory write."""
    encoded = (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def transcript(recipe):
    """Return canonical narration, source notes and explicit verification limits."""
    recipe = validate_recipe(recipe)
    parts = [f"# {recipe['title']}"]
    if recipe["description"]:
        parts.append(recipe["description"])
    parts.append(
        "Narration: one synthetic narrator. This transcript preserves the canonical "
        "script, not synthesis-only pronunciation spellings."
    )
    for section in recipe["sections"]:
        parts.append(f"## {section['heading']}")
        parts.extend(section["paragraphs"])
    parts.append("## Sources and evidence limits")
    if recipe["sources"]:
        for source in recipe["sources"]:
            parts.append(f"- {source['id']}: {source['title']} — {source['url']}")
            if source["note"]:
                parts.append(f"  Note: {source['note']}")
    else:
        parts.append("No sources were supplied. Factual claims have not been source-verified.")
    if recipe["critical_facts"]:
        parts.append("## Critical facts to verify")
        parts.extend(f"- {fact}" for fact in recipe["critical_facts"])
    parts.append(
        "Verification: this transcript does not establish audio accuracy, pronunciation "
        "approval or human listening review. Consult the output manifest and audit."
    )
    return "\n\n".join(parts) + "\n"
