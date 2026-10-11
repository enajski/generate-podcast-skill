"""Assemble selected PCM takes and audit the final encoded MP3."""

import array
import json
import math
import shutil
import subprocess
import sys
import time
import wave
from pathlib import Path

from podcast_core import atomic_json, chunk_recipe, file_sha256, transcript


def _run(arguments):
    completed = subprocess.run(arguments, capture_output=True, text=True, timeout=300, check=False)
    if completed.returncode:
        raise RuntimeError(f"Media command failed: {completed.stderr[-2000:]}")
    return completed


def _tool(value):
    path = shutil.which(str(value))
    if path is None:
        raise ValueError(f"Required media tool is unavailable: {value}")
    return path


def _loudness(stderr):
    start = stderr.rfind("{")
    if start < 0:
        raise ValueError("FFmpeg returned no loudness measurement")
    data = json.JSONDecoder().raw_decode(stderr[start:])[0]
    for key in ("input_i", "input_tp", "input_lra", "input_thresh", "target_offset"):
        if key not in data or not math.isfinite(float(data[key])):
            raise ValueError("Loudness is not finite; check silence or very short input")
    return data


def read_pcm(path):
    try:
        with wave.open(str(path), "rb") as source:
            if source.getnchannels() != 1 or source.getsampwidth() != 2:
                raise ValueError("Selected takes must be mono PCM16 WAV")
            rate, frames = source.getframerate(), source.getnframes()
            if not 8000 <= rate <= 192000 or frames <= 0:
                raise ValueError("Invalid selected sample rate or empty audio")
            raw = source.readframes(frames)
    except (wave.Error, EOFError) as error:
        raise ValueError("Selected WAV is malformed or incomplete") from error
    if len(raw) != frames * 2:
        raise ValueError("Truncated selected WAV")
    samples = array.array("h", raw)
    if sys.byteorder != "little":
        samples.byteswap()
    return rate, samples


def trim_bounds(samples, rate, threshold_db=-60, margin=0.08):
    """Find outer activity with 20 ms RMS frames and retain an edge margin."""
    window, hop = max(1, round(rate * 0.02)), max(1, round(rate * 0.01))
    threshold = (32768 * 10 ** (threshold_db / 20)) ** 2
    first = last = None
    for start in range(0, max(1, len(samples) - window + 1), hop):
        frame = samples[start : start + window]
        if frame and sum(value * value for value in frame) / len(frame) > threshold:
            first = start if first is None else first
            last = min(len(samples), start + window)
    if first is None:
        raise ValueError("Selected audio is silent at the trim threshold")
    return max(0, first - round(rate * margin)), min(len(samples), last + round(rate * margin))


def assemble(output, manifest, destination, settings):
    from podcast import safe_file

    expected = chunk_recipe(manifest["recipe"])
    if manifest.get("chunks") != expected:
        raise ValueError("Manifest chunks differ from the canonical recipe")
    if set(manifest["selected"]) != {chunk["id"] for chunk in expected}:
        raise ValueError("Every chunk needs a selected take before mastering")
    records, chapters = [], []
    rate = None
    cursor = 0
    with wave.open(str(destination), "wb") as target:
        # Initialize a valid empty header so close() cannot hide a validation error.
        target.setnchannels(1)
        target.setsampwidth(2)
        target.setframerate(24000)
        for index, chunk in enumerate(expected):
            take = manifest["selected"][chunk["id"]]
            path = safe_file(output, take["path"])
            if file_sha256(path) != take["sha256"]:
                raise ValueError(f"Selected audio hash changed: {chunk['id']}")
            current_rate, samples = read_pcm(path)
            if current_rate != take["sample_rate"] or len(samples) != take["samples"]:
                raise ValueError("Selected audio metadata differs from its manifest")
            if rate is None:
                rate = current_rate
                target.setnchannels(1)
                target.setsampwidth(2)
                target.setframerate(rate)
            elif current_rate != rate:
                raise ValueError("Selected takes have different sample rates")
            begin, end = trim_bounds(samples, rate, settings["trim_db"], settings["edge_margin"])
            if not chapters or chapters[-1]["section_id"] != chunk["section_id"]:
                chapters.append(
                    {
                        "section_id": chunk["section_id"],
                        "heading": chunk["heading"],
                        "start_sample": cursor,
                        "start_seconds": cursor / rate,
                    }
                )
            record = {
                "id": chunk["id"],
                "start_sample": cursor,
                "start_seconds": cursor / rate,
                "samples": end - begin,
                "duration_seconds": (end - begin) / rate,
                "trim_front_samples": begin,
                "trim_back_samples": len(samples) - end,
                "source_sha256": take["sha256"],
            }
            trimmed = samples[begin:end]
            if sys.byteorder != "little":
                trimmed.byteswap()
            target.writeframes(trimmed.tobytes())
            cursor += end - begin
            gap = 0
            if index + 1 < len(expected):
                gap = round(
                    rate * settings["section_gap" if chunk["section_end"] else "paragraph_gap"]
                )
                target.writeframes(b"\0\0" * gap)
                cursor += gap
            record["following_gap_samples"] = gap
            records.append(record)
    return {
        "sample_rate": rate,
        "samples": cursor,
        "duration_seconds": cursor / rate,
        "chunks": records,
        "chapters": chapters,
        "settings": settings,
        "time_stretch": False,
        "overlap": False,
    }


def inspect_float_audio(path):
    count = clipped = 0
    peak = 0.0
    with Path(path).open("rb") as stream:
        while raw := stream.read(4 * 65536):
            if len(raw) % 4:
                raise ValueError("Truncated decoded float audio")
            samples = array.array("f", raw)
            if sys.byteorder != "little":
                samples.byteswap()
            if not all(math.isfinite(value) for value in samples):
                raise ValueError("Decoded audio has nonfinite samples")
            count += len(samples)
            clipped += sum(abs(value) >= 1 for value in samples)
            peak = max(peak, max((abs(value) for value in samples), default=0))
    if count == 0:
        raise ValueError("Decoded audio is empty")
    return {"decoded_samples": count, "sample_peak": peak, "clipped_samples": clipped}


def master_episode(
    output,
    *,
    ffmpeg="ffmpeg",
    ffprobe="ffprobe",
    paragraph_gap=0.35,
    section_gap=1.2,
    target_lufs=-19.0,
    true_peak=-1.5,
):
    from podcast import _manifest, output_lock, selection_hash

    settings = {
        "paragraph_gap": paragraph_gap,
        "section_gap": section_gap,
        "target_lufs": target_lufs,
        "true_peak": true_peak,
        "target_lra": 11,
        "trim_db": -60,
        "edge_margin": 0.08,
        "bitrate": "128k",
    }
    for name, low, high in (
        ("paragraph_gap", 0, 10),
        ("section_gap", 0, 10),
        ("target_lufs", -30, -10),
        ("true_peak", -9, -1),
    ):
        value = settings[name]
        if isinstance(value, bool) or not math.isfinite(value) or not low <= value <= high:
            raise ValueError(f"Invalid mastering setting: {name}")
    ffmpeg, ffprobe = _tool(ffmpeg), _tool(ffprobe)
    output = Path(output).resolve()
    with output_lock(output):
        manifest = _manifest(output)
        if not manifest or manifest.get("status") != "rendered":
            raise ValueError("Mastering requires a fully rendered episode")
        history = output / "history"
        if history.is_symlink():
            raise ValueError("History directory must not be a symlink")
        atomic_json(history / f"before-master-{time.time_ns()}.json", manifest)
        parent = output / "deliveries"
        if parent.is_symlink():
            raise ValueError("Delivery directory must not be a symlink")
        parent.mkdir(exist_ok=True)
        destination = parent / f"delivery-{time.time_ns()}"
        destination.mkdir()
        premaster, encoded = destination / "premaster.wav", destination / "episode.mp3"
        assembly = assemble(output, manifest, premaster, settings)
        atomic_json(destination / "assembly.json", assembly)
        rate = assembly["sample_rate"]
        base = f"loudnorm=I={target_lufs}:TP={true_peak}:LRA=11"
        measured = _loudness(
            _run(
                [
                    ffmpeg,
                    "-nostdin",
                    "-hide_banner",
                    "-i",
                    str(premaster),
                    "-af",
                    base + ":print_format=json",
                    "-f",
                    "null",
                    "-",
                ]
            ).stderr
        )
        filtering = (
            base
            + f":measured_I={measured['input_i']}:measured_TP={measured['input_tp']}"
            + f":measured_LRA={measured['input_lra']}:measured_thresh={measured['input_thresh']}"
            + f":offset={measured['target_offset']}:linear=true:print_format=json"
        )
        _run(
            [
                ffmpeg,
                "-nostdin",
                "-n",
                "-hide_banner",
                "-i",
                str(premaster),
                "-af",
                filtering,
                "-ar",
                str(rate),
                "-ac",
                "1",
                "-c:a",
                "libmp3lame",
                "-b:a",
                "128k",
                "-metadata",
                "title=" + manifest["recipe"]["title"],
                "-metadata",
                "artist=Synthetic narration",
                str(encoded),
            ]
        )
        probe = json.loads(
            _run(
                [
                    ffprobe,
                    "-v",
                    "error",
                    "-show_entries",
                    "format=duration,size:stream=codec_name,sample_rate,channels",
                    "-of",
                    "json",
                    str(encoded),
                ]
            ).stdout
        )
        decoded = destination / "decoded.f32"
        try:
            _run(
                [
                    ffmpeg,
                    "-nostdin",
                    "-n",
                    "-v",
                    "error",
                    "-i",
                    str(encoded),
                    "-f",
                    "f32le",
                    "-c:a",
                    "pcm_f32le",
                    str(decoded),
                ]
            )
            integrity = inspect_float_audio(decoded)
        finally:
            decoded.unlink(missing_ok=True)
        final = _loudness(
            _run(
                [
                    ffmpeg,
                    "-nostdin",
                    "-hide_banner",
                    "-i",
                    str(encoded),
                    "-af",
                    base + ":print_format=json",
                    "-f",
                    "null",
                    "-",
                ]
            ).stderr
        )
        streams = probe["streams"]
        checks = {
            "complete_decode": True,
            "sample_count_matches_assembly": integrity["decoded_samples"] == assembly["samples"],
            "mono_native_rate": len(streams) == 1
            and streams[0]["channels"] == 1
            and int(streams[0]["sample_rate"]) == rate,
            "no_clipping": integrity["clipped_samples"] == 0,
            "loudness_within_one_lu": abs(float(final["input_i"]) - target_lufs) <= 1.0,
            "true_peak_within_ceiling": float(final["input_tp"]) <= true_peak + 0.1,
        }
        report = {
            "schema_version": 1,
            "status": "passed" if all(checks.values()) else "failed",
            "checks": checks,
            "encoded_sha256": file_sha256(encoded),
            "tools": {
                "ffmpeg": _run([ffmpeg, "-version"]).stdout.splitlines()[0],
                "ffprobe": _run([ffprobe, "-version"]).stdout.splitlines()[0],
            },
            "probe": probe,
            "integrity": integrity,
            "loudness": final,
            "duration_seconds": integrity["decoded_samples"] / rate,
            "asr": "unverified",
            "human_listening": "unverified",
            "pronunciation": "unverified",
            "critical_facts": "unverified",
        }
        atomic_json(destination / "report.json", report)
        if report["status"] != "passed":
            raise ValueError(f"Encoded audio checks failed. Inspect {destination / 'report.json'}")
        delivery = {
            "path": destination.relative_to(output).as_posix(),
            "audio_path": encoded.relative_to(output).as_posix(),
            "audio_sha256": report["encoded_sha256"],
            "selection_hash": selection_hash(manifest),
            "report": report,
        }
        # Reset current verification before writing either copy of the new delivery.
        manifest["delivery"] = delivery
        manifest["verification"] = {key: "unverified" for key in manifest["verification"]}
        manifest.pop("asr_reports", None)
        atomic_json(destination / "manifest.json", manifest)
        chapters = "\n".join(
            f"- {c['start_seconds']:.2f}s: {c['heading']}" for c in assembly["chapters"]
        )
        notes = transcript(manifest["recipe"]) + "\n## Chapter times\n\n" + chapters
        backend = manifest.get("backend", {})
        notes += (
            "\n\n## Synthesis provenance\n\n"
            + f"Backend: {backend.get('backend', 'unrecorded')}. "
            + f"Model revision: {backend.get('model_revision', 'unrecorded')}. "
            + f"Voice: {manifest['recipe']['voice']['voice']}. "
            + "The manifest records model hashes, package versions, settings and selected takes.\n"
            + "\n## Encoded audio checks at mastering\n\n"
            + f"Duration: {report['duration_seconds']:.2f} seconds. "
            + f"Loudness: {final['input_i']} LUFS. True peak: {final['input_tp']} dBTP. "
            + "Full decoding and sample clipping checks passed.\n\n"
            + "ASR, pronunciation, critical facts and human listening remain unverified. "
            + "Consult report.json, the run manifest and any later review records.\n"
        )
        (destination / "notes.md").write_text(notes, encoding="utf-8")
        atomic_json(output / "manifest.json", manifest)
        return delivery
