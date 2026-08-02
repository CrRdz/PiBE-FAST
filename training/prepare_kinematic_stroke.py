#!/usr/bin/env python3
"""Convert the public MATLAB kinematic dataset into repetition-level CSV."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import numpy as np
from scipy.io import loadmat

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.kinematic_features import KINEMATIC_FEATURES, summarize_angle_segment


DEFAULT_INPUT = Path("data/training/downloads/kinematic-emg")
DEFAULT_OUTPUT = Path("training/processed/kinematic_stroke.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def _as_items(value: object) -> list[dict[str, object]]:
    if isinstance(value, dict):
        return [value]
    return list(np.atleast_1d(value))


def extract_file(path: Path) -> tuple[list[dict[str, object]], list[str]]:
    """Extract rows and non-fatal skip messages from one subject MAT file."""

    subject_id = path.stem.upper()
    is_stroke = subject_id.startswith("ST")
    is_healthy = subject_id.startswith("HS")
    if not (is_stroke or is_healthy):
        raise ValueError(f"unsupported subject filename: {path.name}")
    source = loadmat(path, simplify_cells=True, variable_names=["s"])["s"]
    arm_field = "DataULpleg" if is_stroke else "DataULdom"
    tasks = _as_items(source[arm_field])
    sample_rate = float(source["KinFreq"])
    capacity = int(source["L_CA"]) if is_stroke else ""
    rows: list[dict[str, object]] = []
    skipped: list[str] = []

    for task_index, task in enumerate(tasks, 1):
        angles = np.asarray(task["Angles"], dtype=float)
        starts = np.atleast_1d(task["Events"]["Start"]).astype(int)
        ends = np.atleast_1d(task["Events"]["End"]).astype(int)
        if len(starts) != len(ends):
            raise ValueError(f"{path.name} task {task_index}: event counts differ")
        for repetition, (matlab_start, matlab_end) in enumerate(
            zip(starts, ends), 1
        ):
            # MATLAB indices are one-based and End is inclusive.  Python's
            # exclusive stop therefore uses the published End value directly.
            start = max(0, int(matlab_start) - 1)
            end = min(angles.shape[1], int(matlab_end))
            try:
                features = summarize_angle_segment(
                    angles[:, start:end],
                    sample_rate=sample_rate,
                    task_index=task_index,
                )
            except ValueError as exc:
                skipped.append(
                    f"{subject_id}/task-{task_index}/rep-{repetition}: {exc}"
                )
                continue
            rows.append(
                {
                    "subject_id": subject_id,
                    "label": int(is_stroke),
                    "group": "stroke_plegic" if is_stroke else "healthy_dominant",
                    "task_index": task_index,
                    "repetition": repetition,
                    "capacity_level": capacity,
                    "source_file": path.name,
                    **features,
                }
            )
    return rows, skipped


def main() -> int:
    args = parse_args()
    files = sorted(args.input.glob("HS*.mat")) + sorted(args.input.glob("ST*.mat"))
    if not files:
        raise FileNotFoundError(f"no HS/ST MAT files found below {args.input}")
    rows: list[dict[str, object]] = []
    skipped: list[str] = []
    for path in files:
        extracted, warnings = extract_file(path)
        rows.extend(extracted)
        skipped.extend(warnings)
    subjects = {str(row["subject_id"]) for row in rows}
    if len(subjects) != 20:
        raise ValueError(f"expected 20 subjects, found {len(subjects)}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "subject_id",
        "label",
        "group",
        "task_index",
        "repetition",
        "capacity_level",
        "source_file",
        *KINEMATIC_FEATURES,
    ]
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(
        f"wrote {len(rows)} repetitions from {len(subjects)} subjects to {args.output}"
    )
    if skipped:
        print(f"skipped {len(skipped)} repetitions:", file=sys.stderr)
        for warning in skipped:
            print(f"- {warning}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
