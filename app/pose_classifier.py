"""Heuristic standing/sitting/lying classification from MoveNet keypoints."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import acos, atan2, degrees, sqrt
from typing import Mapping, Sequence

from app.config import PoseClassifierConfig
from app.keypoints import KEYPOINT_NAMES, keypoint_map, visible_keypoints


Point = tuple[float, float]


# 姿态分类结果，包含最终姿态、置信度、检测质量和中间几何指标。
@dataclass(frozen=True)
class PoseClassification:
    pose: str
    confidence: float
    quality: float
    metrics: dict[str, float] = field(default_factory=dict)
    reason: str = ""


# 基于 MoveNet 关键点的规则分类器，输出 standing / sitting / lying / unknown。
class PoseClassifier:
    def __init__(self, config: PoseClassifierConfig | None = None) -> None:
        # 允许外部传入配置，方便不同摄像头角度使用不同阈值。
        self.config = config or PoseClassifierConfig()

    def classify(self, keypoints: Sequence[Mapping[str, float]]) -> PoseClassification:
        cfg = self.config
        by_name = keypoint_map(keypoints)
        visible = visible_keypoints(keypoints, cfg.min_keypoint_score)

        # 如果可靠关键点太少，说明人可能被遮挡或模型没检测好，此时不要硬判姿态。
        if len(visible) < cfg.min_visible_keypoints:
            return PoseClassification(
                pose=cfg.fallback_pose,
                confidence=0.0,
                quality=_quality(visible, len(keypoints)),
                metrics={},
                reason="too_few_visible_keypoints",
            )

        # bbox 是所有可靠关键点的外接框，用来描述人体整体是“竖着”还是“横着”。
        bbox = _bbox(visible)
        # 左右肩/胯/膝/踝取平均，可以降低单侧关键点抖动对结果的影响。
        shoulder = _mean_point(by_name, ("left_shoulder", "right_shoulder"), cfg)
        hip = _mean_point(by_name, ("left_hip", "right_hip"), cfg)
        knee = _mean_point(by_name, ("left_knee", "right_knee"), cfg)
        ankle = _mean_point(by_name, ("left_ankle", "right_ankle"), cfg)
        # torso_angle 是躯干相对竖直方向的角度：0 度接近竖直，90 度接近水平。
        torso_angle = _torso_vertical_angle(shoulder, hip)
        # knee_angle 是膝盖弯曲程度：接近 180 度表示腿伸直，越小表示弯曲越明显。
        knee_angle = _mean_knee_angle(by_name, cfg)

        width = max(bbox["width"], 1e-6)
        height = max(bbox["height"], 1e-6)
        # 高/宽大通常像站立；宽/高大通常像躺下。
        height_width_ratio = height / width
        width_height_ratio = width / height
        # 肩、胯、膝、踝在 y 方向 spread 小，说明身体主轴更接近水平。
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

        # 三类姿态分别打分，最后选最高分。这样规则可解释，也方便后续调阈值。
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

        # 如果三个规则都没有明显命中，返回 unknown，避免给 fall 状态机喂错误状态。
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
        # 站立的整体外接框应该更高更窄。
        if height_width_ratio >= cfg.standing_height_width_ratio:
            score += 0.35
        # 站立时肩到胯的连线接近竖直。
        if torso_angle <= cfg.standing_torso_max_degrees:
            score += 0.30
        # 站立时关键点纵向顺序应该是肩在上、胯在中、脚踝在下。
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
        # 坐姿中肩一般仍然高于胯，所以躯干没有完全水平。
        if shoulder and hip and shoulder[1] < hip[1]:
            score += 0.25
        # 坐姿最典型的几何特征：胯和膝盖高度接近。
        if hip and knee and abs(hip[1] - knee[1]) <= cfg.sitting_hip_knee_y_ratio * bbox_height:
            score += 0.35
        # 坐下后膝盖通常会弯曲，膝角明显小于伸直的 180 度。
        if knee_angle <= cfg.sitting_knee_bend_max_degrees:
            score += 0.25
        # 坐姿躯干通常还是偏竖直，允许比站姿更倾斜。
        if torso_angle <= cfg.sitting_torso_max_degrees:
            score += 0.15
        # 脚踝在膝盖下方是一个弱特征，只给少量加分。
        if ankle and knee and ankle[1] > knee[1]:
            score += 0.05
        return min(score, 1.0)

    def _lying_score(
        self, width_height_ratio: float, torso_angle: float, body_y_spread: float, bbox_height: float
    ) -> float:
        cfg = self.config
        score = 0.0
        # 躺下时人体外接框通常横向更长。
        if width_height_ratio >= cfg.lying_width_height_ratio:
            score += 0.45
        # 躯干接近水平是 lying 的强特征。
        if torso_angle >= cfg.lying_torso_min_degrees:
            score += 0.40
        # 主要身体关键点 y 值集中，说明肩、胯、膝、踝趋向一条横线。
        if bbox_height > 0 and body_y_spread <= cfg.lying_body_y_spread_ratio * bbox_height:
            score += 0.15
        return score


def _quality(visible: Sequence[Mapping[str, float]], total: int) -> float:
    # quality 综合考虑“可靠点数量”和“可靠点平均置信度”，用于预览和日志。
    if not visible:
        return 0.0
    count_score = min(1.0, len(visible) / max(total or len(KEYPOINT_NAMES), 1))
    confidence_score = sum(float(kp.get("score", 0.0)) for kp in visible) / len(visible)
    return round(0.5 * count_score + 0.5 * confidence_score, 4)


def _bbox(keypoints: Sequence[Mapping[str, float]]) -> dict[str, float]:
    # 用可靠关键点计算归一化外接框，center_y 后面也会给 fall 状态机使用。
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
    # 对左右同名身体部位求平均。例如肩膀中心 = left_shoulder/right_shoulder 平均。
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
    # 用肩中心到胯中心的向量计算躯干角度。缺点时返回 90，让它更偏向“非竖直”。
    if not shoulder or not hip:
        return 90.0
    dx = shoulder[0] - hip[0]
    dy = shoulder[1] - hip[1]
    return abs(degrees(atan2(abs(dx), abs(dy))))


def _vertical_ordered(points: Sequence[Point | None], tolerance: float) -> bool:
    # 检查一组点的 y 坐标是否从上到下递增；图像坐标中 y 越大越靠下。
    if any(point is None for point in points):
        return False
    ys = [point[1] for point in points if point is not None]
    return all(upper + tolerance < lower for upper, lower in zip(ys, ys[1:]))


def _mean_knee_angle(
    by_name: Mapping[str, Mapping[str, float]], cfg: PoseClassifierConfig
) -> float:
    # 分别计算左右膝盖角度后取平均；缺一侧时仍可用另一侧。
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
    # 读取单个关键点，并统一处理“点不存在或置信度太低”的情况。
    kp = by_name.get(name)
    if not kp or float(kp.get("score", 0.0)) < cfg.min_keypoint_score:
        return None
    return float(kp["x"]), float(kp["y"])


def _angle(a: Point, b: Point, c: Point) -> float:
    # 计算以 b 为顶点的夹角，用于估算膝盖是否弯曲。
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
    # 计算身体主关键点在纵向上的跨度；躺下时这个值通常更小。
    ys = [point[1] for point in points if point is not None]
    if len(ys) < 2:
        return 1.0
    return max(ys) - min(ys)
