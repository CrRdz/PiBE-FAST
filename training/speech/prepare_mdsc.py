#!/usr/bin/env python3
"""Audit an official AISHELL-6B/MDSC tree and create speaker-safe manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import random
import sys
import wave


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--output", default=Path("training/manifests/mdsc.local.csv"), type=Path)
    parser.add_argument("--report", default=Path("training/reports/mdsc_audit.json"), type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    return parser.parse_args()


def _find_class_root(root: Path) -> Path:
    """Find the directory that directly contains Control and Uncontrol.

    AISHELL has distributed MDSC in two layouts: a normalized
    ``train/{Control,Uncontrol}`` tree and separate class archives whose
    extracted shape is ``{Control,Uncontrol}/{train,dev,test}``.
    """

    candidates = (root, root / "train", root / "lrdwws", root / "lrdwws" / "train")
    for candidate in candidates:
        if (candidate / "Control").is_dir() and (candidate / "Uncontrol").is_dir():
            return candidate
    raise ValueError("expected Control and Uncontrol class directories under dataset root")


def _read_transcripts(directory: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for path in sorted(directory.rglob("*.txt")):
        if "__MACOSX" in path.parts or path.name.startswith("._"):
            continue
        for line_number, raw in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
            stripped = raw.strip()
            if not stripped:
                continue
            parts = stripped.split(maxsplit=1)
            if len(parts) != 2:
                raise ValueError(f"invalid transcript line {path}:{line_number}")
            sample_id, text = parts
            if sample_id in values and values[sample_id] != text:
                raise ValueError(f"conflicting transcript for {sample_id}")
            values[sample_id] = text
    if not values:
        raise ValueError(f"no transcript .txt files found under {directory}")
    return values


def _index_wavs(class_root: Path) -> dict[str, Path]:
    values: dict[str, Path] = {}
    for path in sorted(class_root.rglob("*.wav")):
        if "__MACOSX" in path.parts or path.name.startswith("._"):
            continue
        sample_id = path.stem
        if sample_id in values:
            raise ValueError(f"duplicate WAV sample id {sample_id}: {values[sample_id]} and {path}")
        values[sample_id] = path
    if not values:
        raise ValueError(f"no WAV files found under {class_root}")
    return values


def _source_partition(path: Path, class_root: Path) -> str:
    relative = path.relative_to(class_root)
    try:
        wav_index = relative.parts.index("wav")
    except ValueError:
        return "unknown"
    return "/".join(relative.parts[:wav_index]) or "root"


def _speaker_splits(speakers: list[str], seed: int, train_fraction: float, validation_fraction: float) -> dict[str, str]:
    shuffled = sorted(set(speakers))
    random.Random(seed).shuffle(shuffled)
    count = len(shuffled)
    if count < 3:
        raise ValueError("each class needs at least three speakers")
    train_count = max(1, int(round(count * train_fraction)))
    validation_count = max(1, int(round(count * validation_fraction)))
    if train_count + validation_count >= count:
        train_count = count - 2
        validation_count = 1
    return {
        speaker: "train" if index < train_count else "validation" if index < train_count + validation_count else "test"
        for index, speaker in enumerate(shuffled)
    }


def _wav_metadata(path: Path) -> tuple[int, int, int, float]:
    with wave.open(str(path), "rb") as source:
        channels = source.getnchannels()
        rate = source.getframerate()
        frames = source.getnframes()
        width = source.getsampwidth()
    if channels < 1 or width != 2:
        raise ValueError(f"MDSC WAV must contain 16-bit PCM audio: {path}")
    return channels, rate, frames, frames / max(rate, 1)


def build_manifest(root: Path, *, seed: int, train_fraction: float, validation_fraction: float) -> tuple[list[dict[str, object]], dict[str, object]]:
    class_parent = _find_class_root(root.resolve())
    rows: list[dict[str, object]] = []
    missing: list[str] = []
    per_class: dict[int, list[dict[str, object]]] = {0: [], 1: []}
    for folder, label in (("Control", 0), ("Uncontrol", 1)):
        class_root = class_parent / folder
        transcripts = _read_transcripts(class_root)
        wavs = _index_wavs(class_root)
        for sample_id, text in sorted(transcripts.items()):
            speaker_id = sample_id.split("_", 1)[0]
            wav_path = wavs.get(sample_id)
            if wav_path is None:
                missing.append(f"{folder}:{sample_id}")
                continue
            channels, rate, frames, duration = _wav_metadata(wav_path)
            per_class[label].append(
                {
                    "sample_id": sample_id,
                    "speaker_id": speaker_id,
                    "label": label,
                    "class_name": "dysarthria" if label else "control",
                    "text": text,
                    "audio_path": str(wav_path),
                    "sample_rate": rate,
                    "channels": channels,
                    "frames": frames,
                    "duration_seconds": round(duration, 6),
                    "source": "AISHELL-6B/MDSC",
                    "source_partition": _source_partition(wav_path, class_root),
                }
            )
        transcript_ids = set(transcripts)
        extra_wavs = sorted(set(wavs) - transcript_ids)
        if extra_wavs:
            preview = ", ".join(extra_wavs[:10])
            raise ValueError(f"{len(extra_wavs)} WAV files have no transcript under {class_root}: {preview}")
    if missing:
        preview = "\n".join(missing[:10])
        raise ValueError(f"{len(missing)} transcript entries have no WAV, first entries:\n{preview}")
    split_map: dict[str, str] = {}
    for label in (0, 1):
        split_map.update(
            _speaker_splits(
                [str(row["speaker_id"]) for row in per_class[label]],
                seed + label,
                train_fraction,
                validation_fraction,
            )
        )
        rows.extend(per_class[label])
    for row in rows:
        row["split"] = split_map[str(row["speaker_id"])]
    rows.sort(key=lambda row: str(row["sample_id"]))
    speakers_by_split = {
        split: sorted({str(row["speaker_id"]) for row in rows if row["split"] == split})
        for split in ("train", "validation", "test")
    }
    if any(set(speakers_by_split[a]) & set(speakers_by_split[b]) for a, b in (("train", "validation"), ("train", "test"), ("validation", "test"))):
        raise AssertionError("speaker leakage detected")
    report = {
        "dataset": "AISHELL-6B/MDSC",
        "license": "CC BY-NC 4.0",
        "dataset_root": str(root.resolve()),
        "official_structure": "{Control,Uncontrol}/{train,dev,test}/.../{transcript,wav}",
        "samples": len(rows),
        "speakers": len({str(row["speaker_id"]) for row in rows}),
        "class_samples": {"control": len(per_class[0]), "dysarthria": len(per_class[1])},
        "class_speakers": {
            "control": len({str(row["speaker_id"]) for row in per_class[0]}),
            "dysarthria": len({str(row["speaker_id"]) for row in per_class[1]}),
        },
        "audio_formats": {
            f"{channels}ch-{rate}Hz-s16": sum(
                1
                for row in rows
                if int(row["channels"]) == channels and int(row["sample_rate"]) == rate
            )
            for channels, rate in sorted(
                {(int(row["channels"]), int(row["sample_rate"])) for row in rows}
            )
        },
        "split_speakers": {key: len(value) for key, value in speakers_by_split.items()},
        "split_subject_ids_sha256": {
            key: hashlib.sha256("\n".join(value).encode()).hexdigest() for key, value in speakers_by_split.items()
        },
        "seed": seed,
        "warning": "Speaker-grouped research split; MDSC dysarthria is not an acute-stroke label.",
    }
    return rows, report


def main() -> int:
    args = parse_args()
    if args.train_fraction <= 0 or args.validation_fraction <= 0 or args.train_fraction + args.validation_fraction >= 1:
        raise ValueError("train/validation fractions must be positive and sum to less than one")
    rows, report = build_manifest(
        args.dataset_root,
        seed=args.seed,
        train_fraction=args.train_fraction,
        validation_fraction=args.validation_fraction,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": str(args.output), **report}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
