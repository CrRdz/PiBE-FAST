#!/usr/bin/env python3
"""Convert labeled JSONL keypoint trials into one feature row per trial."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.arm_model import ARM_MODEL_FEATURES, summarize_arm_samples
from app.befast.arms import arm_frame_metrics, compensation_keypoint_features


METADATA_COLUMNS = ("subject_id", "trial_id", "label", "affected_side", "source")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--min-keypoint-score", type=float, default=0.35)
    parser.add_argument("--min-valid-samples", type=int, default=20)
    return parser.parse_args()


def load_trial(
    path: Path,
    *,
    min_score: float,
    start_ts: float | None,
    end_ts: float | None,
) -> tuple[list[tuple[float, float, float]], list[dict[str, float]]]:
    samples = []
    rich_frames = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
            timestamp = float(record.get("ts", 0.0))
            if start_ts is not None and timestamp < start_ts:
                continue
            if end_ts is not None and timestamp > end_ts:
                continue
            metrics = arm_frame_metrics(record.get("keypoints", []), min_score)
            rich = compensation_keypoint_features(
                record.get("keypoints", []), min_score
            )
            if rich is not None:
                rich_frames.append(rich)
            if metrics is not None:
                samples.append(
                    (
                        float(metrics["left_wrist_relative_y"]),
                        float(metrics["right_wrist_relative_y"]),
                        float(metrics["level_difference"]),
                    )
                )
    return samples, rich_frames


def optional_float(row: dict[str, str], name: str) -> float | None:
    value = str(row.get(name, "")).strip()
    return None if not value else float(value)


def main() -> int:
    args = parse_args()
    rows = []
    rejected = []
    with args.manifest.open(newline="", encoding="utf-8-sig") as handle:
        manifest = csv.DictReader(handle)
        required = {"subject_id", "trial_id", "label", "path"}
        missing = required - set(manifest.fieldnames or [])
        if missing:
            raise ValueError(f"manifest is missing columns: {', '.join(sorted(missing))}")
        for row in manifest:
            trial_path = Path(row["path"])
            if not trial_path.is_absolute():
                trial_path = (args.manifest.parent / trial_path).resolve()
            samples, rich_frames = load_trial(
                trial_path,
                min_score=args.min_keypoint_score,
                start_ts=optional_float(row, "start_ts"),
                end_ts=optional_float(row, "end_ts"),
            )
            if (
                len(samples) < args.min_valid_samples
                or len(rich_frames) < args.min_valid_samples
            ):
                rejected.append(
                    (
                        row["trial_id"],
                        min(len(samples), len(rich_frames)),
                        str(trial_path),
                    )
                )
                continue
            features = summarize_arm_samples(samples, rich_frames=rich_frames)
            rows.append(
                {
                    "subject_id": row["subject_id"],
                    "trial_id": row["trial_id"],
                    "label": int(row["label"]),
                    "affected_side": row.get("affected_side", ""),
                    "source": row.get("source", ""),
                    **features,
                }
            )
    if not rows:
        raise ValueError("no trials passed the minimum valid-sample requirement")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[*METADATA_COLUMNS, *ARM_MODEL_FEATURES],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} trials to {args.output}")
    if rejected:
        print(f"rejected {len(rejected)} trials with too few valid samples")
        for trial_id, count, path in rejected[:10]:
            print(f"  {trial_id}: {count} valid samples ({path})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
