"""MoveNet-compatible 2-D features for rehabilitation compensation models.

Only joints shared by Kinect v2 and MoveNet are used.  Coordinates are
expressed in a body-centred basis, making the representation independent of
translation, scale, image y direction, and horizontal mirroring.
"""

from __future__ import annotations

from math import acos, hypot, pi
from pathlib import Path
from statistics import median, pstdev
from typing import Mapping, Sequence

COMPENSATION_TARGETS = ("lean_forward", "shoulder_elevation", "trunk_rotation")
COMMON_JOINTS = (
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
)
FRAME_FEATURES = (
    *(f"{joint}_{axis}" for joint in COMMON_JOINTS for axis in ("lateral", "vertical")),
    "torso_length_over_shoulder_width",
    "hip_width_over_shoulder_width",
    "body_axis_nonorthogonality",
    "left_elbow_angle",
    "right_elbow_angle",
)
COMPENSATION_MODEL_FEATURES = tuple(
    f"{name}_{summary}"
    for name in FRAME_FEATURES
    for summary in ("median", "std", "delta")
)


Point2D = tuple[float, float]


def _subtract(a: Point2D, b: Point2D) -> Point2D:
    return a[0] - b[0], a[1] - b[1]


def _dot(a: Point2D, b: Point2D) -> float:
    return a[0] * b[0] + a[1] * b[1]


def _unit(vector: Point2D) -> tuple[Point2D, float]:
    length = hypot(vector[0], vector[1])
    if length <= 1e-6:
        raise ValueError("degenerate body geometry")
    return (vector[0] / length, vector[1] / length), length


def _midpoint(a: Point2D, b: Point2D) -> Point2D:
    return (a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0


def _joint_angle(proximal: Point2D, joint: Point2D, distal: Point2D) -> float:
    first, _ = _unit(_subtract(proximal, joint))
    second, _ = _unit(_subtract(distal, joint))
    cosine = max(-1.0, min(1.0, _dot(first, second)))
    return acos(cosine) / pi


def compensation_frame_features(points: Mapping[str, Point2D]) -> dict[str, float]:
    """Build one canonical 2-D upper-body observation."""

    missing = [name for name in COMMON_JOINTS if name not in points]
    if missing:
        raise ValueError(f"missing compensation joints: {', '.join(missing)}")
    left_shoulder, right_shoulder = points["left_shoulder"], points["right_shoulder"]
    left_hip, right_hip = points["left_hip"], points["right_hip"]
    shoulder_midpoint = _midpoint(left_shoulder, right_shoulder)
    hip_midpoint = _midpoint(left_hip, right_hip)
    lateral_axis, shoulder_width = _unit(_subtract(right_shoulder, left_shoulder))
    vertical_axis, torso_length = _unit(_subtract(shoulder_midpoint, hip_midpoint))
    _, hip_width = _unit(_subtract(right_hip, left_hip))

    output: dict[str, float] = {}
    for joint in COMMON_JOINTS:
        relative = _subtract(points[joint], hip_midpoint)
        output[f"{joint}_lateral"] = _dot(relative, lateral_axis) / shoulder_width
        output[f"{joint}_vertical"] = _dot(relative, vertical_axis) / shoulder_width
    output.update(
        {
            "torso_length_over_shoulder_width": torso_length / shoulder_width,
            "hip_width_over_shoulder_width": hip_width / shoulder_width,
            "body_axis_nonorthogonality": _dot(lateral_axis, vertical_axis),
            "left_elbow_angle": _joint_angle(
                left_shoulder, points["left_elbow"], points["left_wrist"]
            ),
            "right_elbow_angle": _joint_angle(
                right_shoulder, points["right_elbow"], points["right_wrist"]
            ),
        }
    )
    return output


def summarize_compensation_frames(
    frames: Sequence[Mapping[str, float]],
) -> dict[str, float]:
    """Summarize a short frame window using robust level and change features."""

    if len(frames) < 10:
        raise ValueError("at least 10 compensation frames are required")
    segment = max(1, len(frames) // 3)
    output: dict[str, float] = {}
    for name in FRAME_FEATURES:
        values = [float(frame[name]) for frame in frames]
        first = median(values[:segment])
        last = median(values[-segment:])
        output[f"{name}_median"] = median(values)
        output[f"{name}_std"] = pstdev(values)
        output[f"{name}_delta"] = last - first
    return output


def load_compensation_shadow_models(
    directory: str | Path,
) -> tuple[dict[str, object], list[str]]:
    """Load whichever research shadow models are present without blocking A."""

    root = Path(directory)
    # Local import avoids a cycle: arm_model reuses this module's shared V2
    # geometry features.
    from app.arm_model import LinearBinaryModel

    models: dict[str, object] = {}
    errors: list[str] = []
    for target in COMPENSATION_TARGETS:
        path = root / f"toronto_{target}_v1.json"
        if not path.is_file():
            continue
        component = f"A_COMPENSATION_SHADOW_{target.upper()}"
        try:
            models[target] = LinearBinaryModel.load(path, expected_component=component)
        except (OSError, TypeError, ValueError) as exc:
            errors.append(f"{target}: {exc}")
    return models, errors
