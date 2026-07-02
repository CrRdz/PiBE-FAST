"""Draw pose skeleton and runtime state onto frames."""

from __future__ import annotations

from typing import Mapping, Sequence

import cv2

from app.fall_detector import FallDetection
from app.keypoints import SKELETON_EDGES, keypoint_map
from app.pose_classifier import PoseClassification


def draw_overlay(
    frame,
    keypoints: Sequence[Mapping[str, float]],
    pose: PoseClassification,
    fall: FallDetection,
    fps: float,
    min_score: float = 0.25,
):
    output = frame.copy()
    height, width = output.shape[:2]
    by_name = keypoint_map(keypoints)

    for start_name, end_name in SKELETON_EDGES:
        start = by_name.get(start_name)
        end = by_name.get(end_name)
        if not _visible(start, min_score) or not _visible(end, min_score):
            continue
        p1 = _pixel(start, width, height)
        p2 = _pixel(end, width, height)
        cv2.line(output, p1, p2, (0, 220, 255), 2, lineType=cv2.LINE_AA)

    for keypoint in keypoints:
        if not _visible(keypoint, min_score):
            continue
        cv2.circle(output, _pixel(keypoint, width, height), 4, (60, 255, 120), -1)

    state = "fall" if fall.fall else pose.pose
    state_color = (40, 40, 255) if fall.fall else (60, 255, 120)
    lines = [
        f"state: {state}",
        f"pose: {pose.pose} ({pose.confidence:.2f})",
        f"fps: {fps:.1f}",
        f"quality: {pose.quality:.2f}",
    ]
    _draw_status_box(output, lines, state_color)
    return output


def _visible(keypoint: Mapping[str, float] | None, min_score: float) -> bool:
    return bool(keypoint and float(keypoint.get("score", 0.0)) >= min_score)


def _pixel(keypoint: Mapping[str, float], width: int, height: int) -> tuple[int, int]:
    x = int(max(0.0, min(1.0, float(keypoint["x"]))) * (width - 1))
    y = int(max(0.0, min(1.0, float(keypoint["y"]))) * (height - 1))
    return x, y


def _draw_status_box(frame, lines: Sequence[str], accent: tuple[int, int, int]) -> None:
    x, y = 12, 16
    line_height = 24
    box_width = 250
    box_height = line_height * len(lines) + 18
    overlay = frame.copy()
    cv2.rectangle(overlay, (x - 6, y - 14), (x + box_width, y + box_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    cv2.rectangle(frame, (x - 6, y - 14), (x + box_width, y + box_height), accent, 2)
    for index, line in enumerate(lines):
        color = accent if index == 0 else (245, 245, 245)
        cv2.putText(
            frame,
            line,
            (x, y + index * line_height),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            color,
            2,
            lineType=cv2.LINE_AA,
        )
