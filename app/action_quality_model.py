"""Shared per-arm sequence features for external rehabilitation datasets."""

from __future__ import annotations

from math import hypot
from statistics import median, pstdev
from typing import Mapping, Sequence


ACTION_QUALITY_FEATURES = (
    "wrist_vertical_range",
    "wrist_lateral_reach_range",
    "peak_wrist_above_shoulder",
    "peak_lateral_reach",
    "start_wrist_above_shoulder",
    "end_wrist_above_shoulder",
    "start_end_wrist_distance",
    "elbow_angle_min",
    "elbow_angle_median",
    "elbow_angle_max",
    "elbow_extension_fraction",
    "shoulder_vertical_range",
    "torso_length_median",
    "torso_length_std",
    "body_axis_abs_median",
    "body_axis_abs_max",
    "wrist_mean_step",
    "wrist_p90_step",
    "peak_timing_deviation",
)


def _percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(len(ordered) - 1, lower + 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summarize_action_quality(
    frames: Sequence[Mapping[str, float]], side: str
) -> dict[str, float]:
    """Summarize one active arm using only MoveNet/Kinect-shared 2-D joints."""

    if side not in {"left", "right"}:
        raise ValueError("side must be left or right")
    if len(frames) < 10:
        raise ValueError("at least 10 frames are required")
    wrist_vertical = [float(row[f"{side}_wrist_vertical"]) for row in frames]
    wrist_lateral = [float(row[f"{side}_wrist_lateral"]) for row in frames]
    shoulder_vertical = [float(row[f"{side}_shoulder_vertical"]) for row in frames]
    shoulder_lateral = [float(row[f"{side}_shoulder_lateral"]) for row in frames]
    above_shoulder = [
        wrist - shoulder
        for wrist, shoulder in zip(wrist_vertical, shoulder_vertical)
    ]
    lateral_reach = [
        abs(wrist - shoulder)
        for wrist, shoulder in zip(wrist_lateral, shoulder_lateral)
    ]
    elbow = [float(row[f"{side}_elbow_angle"]) for row in frames]
    steps = [
        hypot(
            wrist_lateral[index] - wrist_lateral[index - 1],
            wrist_vertical[index] - wrist_vertical[index - 1],
        )
        for index in range(1, len(frames))
    ]
    peak_index = max(range(len(above_shoulder)), key=above_shoulder.__getitem__)
    torso = [float(row["torso_length_over_shoulder_width"]) for row in frames]
    axis = [abs(float(row["body_axis_nonorthogonality"])) for row in frames]
    return {
        "wrist_vertical_range": max(wrist_vertical) - min(wrist_vertical),
        "wrist_lateral_reach_range": max(lateral_reach) - min(lateral_reach),
        "peak_wrist_above_shoulder": max(above_shoulder),
        "peak_lateral_reach": max(lateral_reach),
        "start_wrist_above_shoulder": median(above_shoulder[:3]),
        "end_wrist_above_shoulder": median(above_shoulder[-3:]),
        "start_end_wrist_distance": hypot(
            median(wrist_lateral[:3]) - median(wrist_lateral[-3:]),
            median(wrist_vertical[:3]) - median(wrist_vertical[-3:]),
        ),
        "elbow_angle_min": min(elbow),
        "elbow_angle_median": median(elbow),
        "elbow_angle_max": max(elbow),
        "elbow_extension_fraction": sum(value >= 0.70 for value in elbow) / len(elbow),
        "shoulder_vertical_range": max(shoulder_vertical) - min(shoulder_vertical),
        "torso_length_median": median(torso),
        "torso_length_std": pstdev(torso),
        "body_axis_abs_median": median(axis),
        "body_axis_abs_max": max(axis),
        "wrist_mean_step": sum(steps) / len(steps),
        "wrist_p90_step": _percentile(steps, 0.90),
        "peak_timing_deviation": abs(peak_index / (len(frames) - 1) - 0.5),
    }
