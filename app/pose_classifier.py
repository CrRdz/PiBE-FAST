"""Heuristic standing/sitting/lying classification from MoveNet keypoints."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import acos, atan2, degrees, sqrt
from typing import Mapping, Sequence

from app.config import PoseClassifierConfig
from app.keypoints import KEYPOINT_NAMES, keypoint_map, visible_keypoints


Point = tuple[float, float]


@dataclass(frozen=True)
class PoseClassification:
    pose: str
    confidence: float
    quality: float
    metrics: dict[str, float] = field(default_factory=dict)
    reason: str = ""


class PoseClassifier:
    def __init__(self, config: PoseClassifierConfig | None = None) -> None:
        self.config = config or PoseClassifierConfig()

    def classify(self, keypoints: Sequence[Mapping[str, float]]) -> PoseClassification:
        cfg = self.config
        by_name = keypoint_map(keypoints)
        visible = visible_keypoints(keypoints, cfg.min_keypoint_score)

        if len(visible) < cfg.min_visible_keypoints:
            return PoseClassification(
                pose=cfg.fallback_pose,
                confidence=0.0,
                quality=_quality(visible, len(keypoints)),
                metrics={},
                reason="too_few_visible_keypoints",
            )

        bbox = _bbox(visible)
        shoulder = _mean_point(by_name, ("left_shoulder", "right_shoulder"), cfg)
        hip = _mean_point(by_name, ("left_hip", "right_hip"), cfg)
        knee = _mean_point(by_name, ("left_knee", "right_knee"), cfg)
        ankle = _mean_point(by_name, ("left_ankle", "right_ankle"), cfg)
        torso_angle = _torso_vertical_angle(shoulder, hip)
        knee_angle = _mean_knee_angle(by_name, cfg)

        width = max(bbox["width"], 1e-6)
        height = max(bbox["height"], 1e-6)
        height_width_ratio = height / width
        width_height_ratio = width / height
        body_y_spread = _body_y_spread([shoulder, hip, knee, ankle])
        metrics = {
            **bbox,
            "height_width_ratio": height_width_ratio,
            "width_height_ratio": width_height_ratio,
            "torso_vertical_degrees": torso_angle,
            "knee_angle_degrees": knee_angle,
            "body_y_spread": body_y_spread,
            "visible_keypoints": float(len(visible)),
        }

        scores = {
            "standing": self._standing_score(
                shoulder, hip, knee, ankle, height_width_ratio, torso_angle
            ),
            "sitting": self._sitting_score(
                shoulder, hip, knee, ankle, knee_angle, torso_angle, height
            ),
            "lying": self._lying_score(
                width_height_ratio, torso_angle, body_y_spread, height
            ),
        }
        pose, score = max(scores.items(), key=lambda item: item[1])

        if score <= 0.0:
            return PoseClassification(
                pose=cfg.fallback_pose,
                confidence=0.0,
                quality=_quality(visible, len(keypoints)),
                metrics=metrics,
                reason="no_rule_matched",
            )

        return PoseClassification(
            pose=pose,
            confidence=min(1.0, score),
            quality=_quality(visible, len(keypoints)),
            metrics=metrics,
            reason=f"{pose}_score",
        )

    def _standing_score(
        self,
        shoulder: Point | None,
        hip: Point | None,
        knee: Point | None,
        ankle: Point | None,
        height_width_ratio: float,
        torso_angle: float,
    ) -> float:
        cfg = self.config
        score = 0.0
        if height_width_ratio >= cfg.standing_height_width_ratio:
            score += 0.35
        if torso_angle <= cfg.standing_torso_max_degrees:
            score += 0.30
        if _vertical_ordered(
            [shoulder, hip, knee, ankle], cfg.standing_order_tolerance
        ):
            score += 0.35
        return score

    def _sitting_score(
        self,
        shoulder: Point | None,
        hip: Point | None,
        knee: Point | None,
        ankle: Point | None,
        knee_angle: float,
        torso_angle: float,
        bbox_height: float,
    ) -> float:
        cfg = self.config
        score = 0.0
        if shoulder and hip and shoulder[1] < hip[1]:
            score += 0.25
        if hip and knee and abs(hip[1] - knee[1]) <= cfg.sitting_hip_knee_y_ratio * bbox_height:
            score += 0.35
        if knee_angle <= cfg.sitting_knee_bend_max_degrees:
            score += 0.25
        if torso_angle <= cfg.sitting_torso_max_degrees:
            score += 0.15
        if ankle and knee and ankle[1] > knee[1]:
            score += 0.05
        return min(score, 1.0)

    def _lying_score(
        self, width_height_ratio: float, torso_angle: float, body_y_spread: float, bbox_height: float
    ) -> float:
        cfg = self.config
        score = 0.0
        if width_height_ratio >= cfg.lying_width_height_ratio:
            score += 0.45
        if torso_angle >= cfg.lying_torso_min_degrees:
            score += 0.40
        if bbox_height > 0 and body_y_spread <= cfg.lying_body_y_spread_ratio * bbox_height:
            score += 0.15
        return score


def _quality(visible: Sequence[Mapping[str, float]], total: int) -> float:
    if not visible:
        return 0.0
    count_score = min(1.0, len(visible) / max(total or len(KEYPOINT_NAMES), 1))
    confidence_score = sum(float(kp.get("score", 0.0)) for kp in visible) / len(visible)
    return round(0.5 * count_score + 0.5 * confidence_score, 4)


def _bbox(keypoints: Sequence[Mapping[str, float]]) -> dict[str, float]:
    xs = [float(kp["x"]) for kp in keypoints]
    ys = [float(kp["y"]) for kp in keypoints]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    return {
        "min_x": min_x,
        "min_y": min_y,
        "max_x": max_x,
        "max_y": max_y,
        "width": max_x - min_x,
        "height": max_y - min_y,
        "center_x": (min_x + max_x) / 2.0,
        "center_y": (min_y + max_y) / 2.0,
    }


def _mean_point(
    by_name: Mapping[str, Mapping[str, float]],
    names: Sequence[str],
    cfg: PoseClassifierConfig,
) -> Point | None:
    points = [
        (float(by_name[name]["x"]), float(by_name[name]["y"]))
        for name in names
        if name in by_name and float(by_name[name].get("score", 0.0)) >= cfg.min_keypoint_score
    ]
    if not points:
        return None
    return (
        sum(point[0] for point in points) / len(points),
        sum(point[1] for point in points) / len(points),
    )


def _torso_vertical_angle(shoulder: Point | None, hip: Point | None) -> float:
    if not shoulder or not hip:
        return 90.0
    dx = shoulder[0] - hip[0]
    dy = shoulder[1] - hip[1]
    return abs(degrees(atan2(abs(dx), abs(dy))))


def _vertical_ordered(points: Sequence[Point | None], tolerance: float) -> bool:
    if any(point is None for point in points):
        return False
    ys = [point[1] for point in points if point is not None]
    return all(upper + tolerance < lower for upper, lower in zip(ys, ys[1:]))


def _mean_knee_angle(
    by_name: Mapping[str, Mapping[str, float]], cfg: PoseClassifierConfig
) -> float:
    angles = []
    for side in ("left", "right"):
        hip = _single_point(by_name, f"{side}_hip", cfg)
        knee = _single_point(by_name, f"{side}_knee", cfg)
        ankle = _single_point(by_name, f"{side}_ankle", cfg)
        if hip and knee and ankle:
            angles.append(_angle(hip, knee, ankle))
    if not angles:
        return 180.0
    return sum(angles) / len(angles)


def _single_point(
    by_name: Mapping[str, Mapping[str, float]],
    name: str,
    cfg: PoseClassifierConfig,
) -> Point | None:
    kp = by_name.get(name)
    if not kp or float(kp.get("score", 0.0)) < cfg.min_keypoint_score:
        return None
    return float(kp["x"]), float(kp["y"])


def _angle(a: Point, b: Point, c: Point) -> float:
    ba = (a[0] - b[0], a[1] - b[1])
    bc = (c[0] - b[0], c[1] - b[1])
    len_ba = sqrt(ba[0] * ba[0] + ba[1] * ba[1])
    len_bc = sqrt(bc[0] * bc[0] + bc[1] * bc[1])
    if len_ba <= 1e-9 or len_bc <= 1e-9:
        return 180.0
    cosine = (ba[0] * bc[0] + ba[1] * bc[1]) / (len_ba * len_bc)
    cosine = max(-1.0, min(1.0, cosine))
    return degrees(acos(cosine))


def _body_y_spread(points: Sequence[Point | None]) -> float:
    ys = [point[1] for point in points if point is not None]
    if len(ys) < 2:
        return 1.0
    return max(ys) - min(ys)
