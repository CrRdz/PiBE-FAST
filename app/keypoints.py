"""MoveNet keypoint names and skeleton topology."""

from __future__ import annotations

from typing import Iterable, Mapping


KEYPOINT_NAMES = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]


SKELETON_EDGES = [
    ("nose", "left_eye"),
    ("nose", "right_eye"),
    ("left_eye", "left_ear"),
    ("right_eye", "right_ear"),
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
]


def keypoint_map(keypoints: Iterable[Mapping[str, float]]) -> dict[str, Mapping[str, float]]:
    return {str(kp["name"]): kp for kp in keypoints if "name" in kp}


def visible_keypoints(
    keypoints: Iterable[Mapping[str, float]], min_score: float
) -> list[Mapping[str, float]]:
    return [kp for kp in keypoints if float(kp.get("score", 0.0)) >= min_score]

