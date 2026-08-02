#!/usr/bin/env python3
"""Convert IntelliRehabDS shoulder-abduction trials to shared 2-D features."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.action_quality_model import ACTION_QUALITY_FEATURES, summarize_action_quality
from app.compensation_model import compensation_frame_features


JOINT_INDEX = {
    "left_shoulder": 4,
    "left_elbow": 5,
    "left_wrist": 6,
    "right_shoulder": 8,
    "right_elbow": 9,
    "right_wrist": 10,
    "left_hip": 12,
    "right_hip": 16,
}
GESTURE_SIDE = {4: "left", 5: "right"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/training/raw/intellirehabds/SkeletonData/Simplified"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("training/processed/intellirehab_action_quality.csv"),
    )
    return parser.parse_args()


def load_trial(path: Path) -> list[dict[str, float]]:
    frames = []
    with path.open(encoding="utf-8-sig") as handle:
        for line in handle:
            values = [float(value) for value in line.strip().split(",") if value]
            if len(values) != 75:
                continue
            points = {
                name: (values[index * 3], values[index * 3 + 1])
                for name, index in JOINT_INDEX.items()
            }
            try:
                frames.append(compensation_frame_features(points))
            except ValueError:
                continue
    return frames


def main() -> int:
    args = parse_args()
    rows = []
    rejected = 0
    for path in sorted(args.input.glob("*.txt")):
        parts = path.stem.split("_")
        if len(parts) < 6:
            continue
        subject, session, gesture_text, repetition, correct_text = parts[:5]
        gesture = int(gesture_text)
        if gesture not in GESTURE_SIDE:
            continue
        frames = load_trial(path)
        if len(frames) < 10:
            rejected += 1
            continue
        features = summarize_action_quality(frames, GESTURE_SIDE[gesture])
        rows.append(
            {
                "subject_id": subject,
                "trial_id": path.stem,
                "gesture": gesture,
                "side": GESTURE_SIDE[gesture],
                # Positive means invalid/incorrect so threshold calibration
                # prioritizes catching movements that should not be accepted.
                "label": int(int(correct_text) != 1),
                "frames": len(frames),
                "source": "IntelliRehabDS-2.0.1",
                **features,
            }
        )
    if not rows:
        raise ValueError(f"no shoulder-abduction trials found under {args.input}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "subject_id", "trial_id", "gesture", "side", "label", "frames", "source",
        *ACTION_QUALITY_FEATURES,
    ]
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} trials to {args.output}; rejected={rejected}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
