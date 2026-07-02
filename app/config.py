"""Centralized runtime and heuristic configuration."""

from dataclasses import dataclass


@dataclass(frozen=True)
class PoseClassifierConfig:
    min_keypoint_score: float = 0.25
    min_visible_keypoints: int = 5
    standing_height_width_ratio: float = 1.25
    standing_torso_max_degrees: float = 35.0
    standing_order_tolerance: float = 0.04
    lying_width_height_ratio: float = 1.15
    lying_torso_min_degrees: float = 60.0
    lying_body_y_spread_ratio: float = 0.35
    sitting_hip_knee_y_ratio: float = 0.22
    sitting_knee_bend_max_degrees: float = 155.0
    sitting_torso_max_degrees: float = 55.0
    fallback_pose: str = "unknown"


@dataclass(frozen=True)
class FallDetectorConfig:
    history_seconds: float = 8.0
    transition_seconds: float = 1.6
    center_drop_threshold: float = 0.18
    torso_angle_delta_threshold: float = 40.0
    vertical_torso_max_degrees: float = 40.0
    horizontal_torso_min_degrees: float = 60.0
    lying_hold_seconds: float = 2.0
    max_candidate_seconds: float = 5.0
    recovery_hold_seconds: float = 1.5


@dataclass(frozen=True)
class RuntimeConfig:
    frame_width: int = 640
    frame_height: int = 480
    frame_fps: int = 15
    jpeg_quality: int = 80
    log_every_seconds: float = 1.0
    event_pre_seconds: float = 5.0
    event_post_seconds: float = 5.0

