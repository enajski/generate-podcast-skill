#!/usr/bin/env python3
"""Run the portable podcast skill without importing Vis."""

import argparse
import contextlib
import json
import os
import sys
import time
from pathlib import Path

from podcast_core import (
    atomic_json,
    chunk_recipe,
    file_sha256,
    fingerprint,
    load_recipe,
    transcript,
)


@contextlib.contextmanager
def output_lock(output):
    """Reject concurrent writers and leave stale locks for explicit inspection."""
    output.mkdir(parents=True, exist_ok=True)
    lock = output / ".podcast.lock"
    descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(json.dumps({"pid": os.getpid(), "created": time.time()}))
        yield
    finally:
        lock.unlink(missing_ok=True)


def _manifest(output):
    path = output / "manifest.json"
    if not path.exists():
        if any(p.name != ".podcast.lock" for p in output.iterdir()):
            raise ValueError("Output directory is not empty and has no podcast manifest.")
        return None
    state = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(state, dict) or state.get("schema_version") != 1:
        raise ValueError("Unsupported podcast manifest")
    if not isinstance(state.get("selected"), dict):
        raise ValueError("Invalid selected-take manifest")
    return state


def safe_file(output, relative):
    """Resolve an artifact without allowing an escape from the run directory."""
    if not isinstance(relative, str) or Path(relative).is_absolute():
        raise ValueError("Artifact path must be relative")
    path = (output / relative).resolve()
    if not path.is_relative_to(output.resolve()) or not path.is_file():
        raise ValueError(f"Missing or unsafe artifact path: {relative}")
    return path


def selection_hash(manifest):
    """Bind review and delivery state to the exact script and selected takes."""
    return fingerprint({"recipe": manifest["recipe_hash"], "selected": manifest["selected"]})


def render_episode(
    recipe_path, model, revision, output, *, chunks=None, retake=False, seed_offset=0
):
    """Resume compatible takes and preserve every replaced take and checkpoint."""
    from podcast_qwen import QwenRenderer, backend_info

    recipe = load_recipe(recipe_path)
    items = chunk_recipe(recipe)
    ids = {item["id"] for item in items}
    requested = set(chunks) if chunks is not None else ids
    if not requested or not requested <= ids:
        raise ValueError("Requested chunks must be nonempty known chunk identifiers")
    if type(seed_offset) is not int or not 0 <= seed_offset <= 0xFFFFFFFF:
        raise ValueError("seed-offset must be an integer from 0 to 4294967295")
    if seed_offset and not retake:
        raise ValueError("seed-offset requires --retake")
    output = Path(output).resolve()
    # Refuse foreign output directories before model hashing, then recheck under lock.
    if output.exists() and any(output.iterdir()) and not (output / "manifest.json").exists():
        raise ValueError("Output directory is not empty and has no podcast manifest.")
    info = backend_info(Path(model), revision)
    with output_lock(output):
        if (output / "transcript.md").is_symlink():
            raise ValueError("Transcript path must not be a symlink")
        previous = _manifest(output)
        for name in ("takes", "history"):
            directory = output / name
            if directory.is_symlink():
                raise ValueError(f"Output subdirectory must not be a symlink: {name}")
            directory.mkdir(exist_ok=True)
        if previous is not None:
            atomic_json(output / "history" / f"{time.time_ns()}.json", previous)
        manifest = {
            "schema_version": 1,
            "status": "rendering",
            "recipe": recipe,
            "recipe_hash": fingerprint(recipe),
            "backend": info,
            "chunks": items,
            "selected": {},
            "reviews": (previous or {}).get("reviews", []),
            "verification": {
                "human_listening": "unverified",
                "pronunciation": "unverified",
                "critical_facts": "unverified",
                "asr": "unverified",
            },
        }
        keys = {}
        for item in items:
            key = fingerprint(
                {
                    "synthesis_text": item["synthesis_text"],
                    "voice": recipe["voice"],
                    "seed_base": recipe["seed_base"],
                    "backend": info,
                    "renderer_schema": 1,
                }
            )
            keys[item["id"]] = key
            old = (previous or {}).get("selected", {}).get(item["id"])
            if old and old.get("synthesis_key") == key:
                try:
                    compatible = file_sha256(safe_file(output, old["path"])) == old["sha256"]
                except (ValueError, OSError, KeyError):
                    compatible = False
                if compatible:
                    manifest["selected"][item["id"]] = old
        atomic_json(output / "manifest.json", manifest)
        (output / "transcript.md").write_text(transcript(recipe), encoding="utf-8")
        renderer = None
        try:
            for item in items:
                identifier = item["id"]
                if identifier not in requested:
                    continue
                if identifier in manifest["selected"] and not retake:
                    continue
                if renderer is None:
                    renderer = QwenRenderer(Path(model))
                seed = (
                    recipe["seed_base"] + int(fingerprint(identifier)[:8], 16) + seed_offset
                ) % 2**32
                relative = f"takes/{identifier}-{time.time_ns()}.wav"
                metrics = renderer.render(
                    item["synthesis_text"], recipe["voice"], seed, output / relative
                )
                manifest["selected"][identifier] = {
                    "path": relative,
                    "seed": seed,
                    "settings": recipe["voice"],
                    "synthesis_key": keys[identifier],
                    **metrics,
                }
                atomic_json(output / "manifest.json", manifest)
                print(json.dumps({"chunk": identifier, **metrics}), flush=True)
            manifest["status"] = "rendered" if set(manifest["selected"]) == ids else "incomplete"
            if previous and selection_hash(previous) == selection_hash(manifest):
                delivery = previous.get("delivery")
                try:
                    valid_delivery = (
                        delivery is not None
                        and delivery.get("selection_hash") == selection_hash(manifest)
                        and file_sha256(safe_file(output, delivery["audio_path"]))
                        == delivery["audio_sha256"]
                    )
                except (OSError, ValueError, KeyError):
                    valid_delivery = False
                if valid_delivery:
                    manifest["delivery"] = delivery
                    manifest["verification"] = dict(previous["verification"])
                    reports = previous.get("asr_reports", [])
                    try:
                        valid_reports = bool(reports) and all(
                            file_sha256(safe_file(output, item["path"])) == item["sha256"]
                            and json.loads(safe_file(output, item["path"]).read_text())[
                                "audio_sha256"
                            ]
                            == delivery["audio_sha256"]
                            for item in reports
                        )
                    except (OSError, ValueError, KeyError, TypeError):
                        valid_reports = False
                    if valid_reports:
                        manifest["asr_reports"] = reports
                    else:
                        manifest["verification"]["asr"] = "unverified"
        except BaseException:
            manifest["status"] = "interrupted"
            raise
        finally:
            atomic_json(output / "manifest.json", manifest)
    return manifest


def record_review(output, scope, reviewer, result, evidence):
    """Record an attributed human statement without inferring listening approval."""
    if scope not in {"audition", "episode", "critical-facts"}:
        raise ValueError("Unknown review scope")
    if result not in {"approved", "rejected"}:
        raise ValueError("Review result must be approved or rejected")
    if not reviewer.strip() or not evidence.strip():
        raise ValueError("Review needs reviewer attribution and evidence")
    output = Path(output).resolve()
    with output_lock(output):
        state = _manifest(output)
        if not state or state.get("status") != "rendered":
            raise ValueError("Review requires a fully rendered episode")
        for take in state["selected"].values():
            if file_sha256(safe_file(output, take["path"])) != take["sha256"]:
                raise ValueError("Selected audio changed since rendering")
        review = {
            "scope": scope,
            "reviewer": reviewer,
            "result": result,
            "evidence": evidence,
            "selection_hash": selection_hash(state),
            "recorded_at": time.time(),
        }
        if scope == "episode":
            delivery = state.get("delivery")
            if not delivery or delivery.get("selection_hash") != selection_hash(state):
                raise ValueError("Episode review requires a current encoded delivery")
            path = safe_file(output, delivery["audio_path"])
            if file_sha256(path) != delivery["audio_sha256"]:
                raise ValueError("Encoded audio changed since mastering")
            review["audio_sha256"] = delivery["audio_sha256"]
            state["verification"]["human_listening"] = result
        elif scope == "critical-facts":
            state["verification"]["critical_facts"] = result
        # An audition is explicitly not whole-episode pronunciation approval.
        state.setdefault("reviews", []).append(review)
        atomic_json(output / "manifest.json", state)
    return review


def compare_asr(output, raw_path):
    from podcast_asr import compare_manifest

    output = Path(output).resolve()
    raw = json.loads(Path(raw_path).read_text(encoding="utf-8"))
    with output_lock(output):
        state = _manifest(output)
        delivery = state.get("delivery") if state else None
        if not delivery or delivery.get("selection_hash") != selection_hash(state):
            raise ValueError("ASR comparison requires a current encoded delivery")
        audio = safe_file(output, delivery["audio_path"])
        if file_sha256(audio) != delivery["audio_sha256"]:
            raise ValueError("Encoded audio changed since mastering")
        report = compare_manifest(state, raw)
        target = audio.parent / f"asr-{time.time_ns()}.json"
        atomic_json(target, report)
        state.setdefault("asr_reports", []).append(
            {"path": target.relative_to(output).as_posix(), "sha256": file_sha256(target)}
        )
        state["verification"]["asr"] = "compared-unadjudicated"
        atomic_json(output / "manifest.json", state)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate", help="Check a recipe without loading a model")
    validate.add_argument("recipe", type=Path)
    render = commands.add_parser("render", help="Render an approved recipe with a local model")
    render.add_argument("recipe", type=Path)
    render.add_argument("--model", required=True, type=Path)
    render.add_argument("--revision", required=True)
    render.add_argument("--out", required=True, type=Path)
    render.add_argument("--chunks", help="Comma-separated chunk identifiers")
    render.add_argument("--retake", action="store_true")
    render.add_argument("--seed-offset", type=int, default=0)
    review = commands.add_parser("review", help="Record an actual human review")
    review.add_argument("output", type=Path)
    review.add_argument("--scope", required=True, choices=["audition", "episode", "critical-facts"])
    review.add_argument("--reviewer", required=True)
    review.add_argument("--result", required=True, choices=["approved", "rejected"])
    review.add_argument("--evidence", required=True)
    master = commands.add_parser("master", help="Assemble, encode and check final audio")
    master.add_argument("output", type=Path)
    master.add_argument("--ffmpeg", default="ffmpeg")
    master.add_argument("--ffprobe", default="ffprobe")
    master.add_argument("--paragraph-gap", type=float, default=0.35)
    master.add_argument("--section-gap", type=float, default=1.2)
    master.add_argument("--target-lufs", type=float, default=-19.0)
    master.add_argument("--true-peak", type=float, default=-1.5)
    asr = commands.add_parser("compare-asr", help="Compare a raw transcript with canonical chunks")
    asr.add_argument("output", type=Path)
    asr.add_argument("transcript", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "validate":
            recipe = load_recipe(args.recipe)
            chunks = chunk_recipe(recipe)
            print(json.dumps({"title": recipe["title"], "chunks": chunks}, ensure_ascii=False))
        elif args.command == "render":
            result = render_episode(
                args.recipe,
                args.model,
                args.revision,
                args.out,
                chunks=args.chunks.split(",") if args.chunks else None,
                retake=args.retake,
                seed_offset=args.seed_offset,
            )
            print(json.dumps({"status": result["status"], "output": str(args.out)}))
        elif args.command == "master":
            from podcast_media import master_episode

            print(
                json.dumps(
                    master_episode(
                        args.output,
                        ffmpeg=args.ffmpeg,
                        ffprobe=args.ffprobe,
                        paragraph_gap=args.paragraph_gap,
                        section_gap=args.section_gap,
                        target_lufs=args.target_lufs,
                        true_peak=args.true_peak,
                    )
                )
            )
        elif args.command == "compare-asr":
            print(json.dumps(compare_asr(args.output, args.transcript), ensure_ascii=False))
        else:
            print(
                json.dumps(
                    record_review(
                        args.output, args.scope, args.reviewer, args.result, args.evidence
                    )
                )
            )
    except (ValueError, OSError, RuntimeError, ImportError) as error:
        print(f"podcast: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("podcast: interrupted; completed takes remain on disk", file=sys.stderr)
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
