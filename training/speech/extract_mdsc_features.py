#!/usr/bin/env python3
"""Extract the exact NumPy representation used by Raspberry Pi inference."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.speech_representation import SPEECH_REPRESENTATION_FEATURES, extract_speech_representation


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", default=Path("training/processed/mdsc_features.npz"), type=Path)
    parser.add_argument("--limit", type=int, default=0, help="Smoke-test only; zero extracts all rows")
    args = parser.parse_args()
    with args.manifest.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if args.limit > 0:
        rows = rows[: args.limit]
    if not rows:
        raise ValueError("MDSC manifest is empty")
    matrix = []
    for index, row in enumerate(rows, 1):
        matrix.append(extract_speech_representation(row["audio_path"]))
        if index % 250 == 0:
            print(f"extracted {index}/{len(rows)}", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        matrix=np.asarray(matrix, dtype=np.float32),
        labels=np.asarray([int(row["label"]) for row in rows], dtype=np.int8),
        subjects=np.asarray([row["speaker_id"] for row in rows]),
        sample_ids=np.asarray([row["sample_id"] for row in rows]),
        splits=np.asarray([row["split"] for row in rows]),
        texts=np.asarray([row["text"] for row in rows]),
        feature_names=np.asarray(SPEECH_REPRESENTATION_FEATURES),
    )
    print(f"saved {len(rows)} samples x {len(SPEECH_REPRESENTATION_FEATURES)} features to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
