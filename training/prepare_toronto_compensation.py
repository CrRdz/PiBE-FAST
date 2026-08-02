#!/usr/bin/env python3
"""Convert Toronto Kinect skeletons into MoveNet-compatible 2-D windows."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.compensation_model import (
    COMPENSATION_MODEL_FEATURES,
    compensation_frame_features,
    summarize_compensation_frames,
)


KINECT_JOINTS = {
    "left_shoulder": 4,
    "left_elbow": 5,
    "left_wrist": 6,
    "right_shoulder": 8,
    "right_elbow": 9,
    "right_wrist": 10,
    "left_hip": 12,
    "right_hip": 16,
}
LABEL_NAMES = {1: "none", 2: "lean_forward", 3: "shoulder_elevation", 4: "trunk_rotation"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/training/raw/toronto-pose/data_new"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("training/processed/toronto_compensation.csv"),
    )
    parser.add_argument("--window-frames", type=int, default=30)
    parser.add_argument("--stride-frames", type=int, default=15)
    parser.add_argument("--minimum-label-purity", type=float, default=0.80)
    return parser.parse_args()


def extract_trial(
    joint_path: Path,
    *,
    window_frames: int,
    stride_frames: int,
    minimum_label_purity: float,
) -> tuple[list[dict[str, object]], Counter[str]]:
    trial_dir = joint_path.parent
    subject_id = trial_dir.parent.name
    coordinates = np.loadtxt(joint_path, delimiter=",").reshape(-1, 25, 3)
    labels = np.loadtxt(trial_dir / "Labels.csv", dtype=int, ndmin=1)
    if len(coordinates) != len(labels):
        raise ValueError(f"frame/label mismatch in {trial_dir}")
    rows: list[dict[str, object]] = []
    stats: Counter[str] = Counter()
    for start in range(0, len(labels) - window_frames + 1, stride_frames):
        stop = start + window_frames
        counts = Counter(int(value) for value in labels[start:stop])
        majority_label, majority_count = counts.most_common(1)[0]
        purity = majority_count / window_frames
        if majority_label not in LABEL_NAMES or purity < minimum_label_purity:
            stats["impure_windows"] += 1
            continue
        frame_rows = []
        for skeleton in coordinates[start:stop]:
            # Kinect x/y are projected as a 2-D view.  The body-centred feature
            # basis removes y-direction and horizontal-mirroring conventions.
            points = {
                name: (float(skeleton[index, 0]), float(skeleton[index, 1]))
                for name, index in KINECT_JOINTS.items()
            }
            try:
                frame_rows.append(compensation_frame_features(points))
            except ValueError:
                stats["invalid_geometry_frames"] += 1
        if len(frame_rows) < max(10, int(window_frames * 0.8)):
            stats["invalid_geometry_windows"] += 1
            continue
        features = summarize_compensation_frames(frame_rows)
        rows.append(
            {
                "subject_id": subject_id,
                "cohort": "healthy_simulated" if subject_id.startswith("H") else "stroke_patient",
                "trial_id": trial_dir.name,
                "window_start": start,
                "source_label": majority_label,
                "source_label_name": LABEL_NAMES[majority_label],
                "label_purity": purity,
                **features,
            }
        )
        stats[f"label_{majority_label}"] += 1
    return rows, stats


def main() -> int:
    args = parse_args()
    if args.window_frames < 10 or args.stride_frames < 1:
        raise ValueError("window must be >=10 frames and stride must be positive")
    joint_files = sorted(args.input.glob("*/*/Joint_Positions.csv"))
    if not joint_files:
        raise FileNotFoundError(f"no Toronto trials found below {args.input}")
    rows: list[dict[str, object]] = []
    totals: Counter[str] = Counter()
    for joint_path in joint_files:
        extracted, stats = extract_trial(
            joint_path,
            window_frames=args.window_frames,
            stride_frames=args.stride_frames,
            minimum_label_purity=args.minimum_label_purity,
        )
        rows.extend(extracted)
        totals.update(stats)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "subject_id",
        "cohort",
        "trial_id",
        "window_start",
        "source_label",
        "source_label_name",
        "label_purity",
        *COMPENSATION_MODEL_FEATURES,
    ]
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(
        f"wrote {len(rows)} windows from {len(joint_files)} trials and "
        f"{len({row['subject_id'] for row in rows})} subjects to {args.output}"
    )
    print(dict(sorted(totals.items())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
