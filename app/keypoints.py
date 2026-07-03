"""MoveNet keypoint names and skeleton topology."""

from __future__ import annotations

from typing import Iterable, Mapping


# MoveNet SinglePose 固定输出 17 个关键点，顺序必须和模型输出一一对应。
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


# 这些边用于画骨架线，只影响可视化，不参与姿态分类。
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
    # 把 list 转成按名称索引的 dict，后面取 left_shoulder 这类点更方便。
    return {str(kp["name"]): kp for kp in keypoints if "name" in kp}


def visible_keypoints(
    keypoints: Iterable[Mapping[str, float]], min_score: float
) -> list[Mapping[str, float]]:
    # 过滤掉置信度太低的点，避免被遮挡/误检的关键点影响规则判断。
    return [kp for kp in keypoints if float(kp.get("score", 0.0)) >= min_score]
