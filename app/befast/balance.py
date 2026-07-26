"""B（Balance）：检测站立中心偏移/摆动，并与人工平衡观察合并。"""

from __future__ import annotations

from math import hypot
from statistics import median
from typing import Any, Mapping, Sequence

from app.keypoints import keypoint_map

from .config import BefastConfig
from .pose_geometry import percentile, visible_point
from .result import MotionResult, motion_report_item, report_item


def balance_frame_metrics(
    keypoints: Sequence[Mapping[str, float]], min_score: float
) -> dict[str, float] | None:
    """从单帧关键点计算身体中心相对双脚支撑中心的横向偏移。"""

    points = keypoint_map(keypoints)
    # 肩、胯用于估计身体中心，脚踝用于估计站立支撑中心。
    required_names = (
        "left_shoulder",
        "right_shoulder",
        "left_hip",
        "right_hip",
        "left_ankle",
        "right_ankle",
    )
    required = [visible_point(points, name, min_score) for name in required_names]
    if any(point is None for point in required):
        return None
    left_shoulder, right_shoulder, left_hip, right_hip, left_ankle, right_ankle = required
    assert left_shoulder and right_shoulder and left_hip and right_hip
    assert left_ankle and right_ankle
    shoulder_width = hypot(
        right_shoulder[0] - left_shoulder[0],
        right_shoulder[1] - left_shoulder[1],
    )
    if shoulder_width <= 1e-4:
        return None
    # 所有横向距离都除以肩宽，使结果不依赖画面缩放。
    shoulder_center_x = (left_shoulder[0] + right_shoulder[0]) / 2.0
    hip_center_x = (left_hip[0] + right_hip[0]) / 2.0
    support_center_x = (left_ankle[0] + right_ankle[0]) / 2.0
    body_center_x = (shoulder_center_x + hip_center_x) / 2.0
    return {
        "body_support_offset": (body_center_x - support_center_x) / shoulder_width,
        "torso_lateral_offset": (shoulder_center_x - hip_center_x) / shoulder_width,
        "shoulder_width": shoulder_width,
    }


def balance_report_item(
    manual_problem: bool | None, result: MotionResult
) -> dict[str, Any]:
    """合并人工平衡回答与姿态结果，生成最终 B 报告项。"""

    # 人工明确报告平衡问题时直接保留阳性，不被姿态结果覆盖。
    if manual_problem is True:
        return report_item("positive", "manual", "reported_balance_problem")
    if result.status in {"positive", "skipped"}:
        return motion_report_item(result, "pose")
    # B 同时包含主观症状和粗粒度姿态代理；两者都阴性才输出阴性。
    if manual_problem is False and result.status == "negative":
        return motion_report_item(result, "manual+pose")
    if manual_problem is False and result.status == "insufficient":
        return motion_report_item(result, "pose")
    return report_item("pending", "manual+pose", "not_fully_checked")


class BalanceScreen:
    """累计站立姿态样本，检测持续侧偏和较大的横向摆动。"""

    def __init__(self, config: BefastConfig) -> None:
        self.config = config
        self.reset()

    @property
    def duration_seconds(self) -> float:
        """返回准备期与正式采样期的总时长。"""

        return self.config.balance_warmup_seconds + self.config.balance_capture_seconds

    def reset(self) -> None:
        """清空本轮采样时间、质量计数和中心偏移序列。"""

        self.start_ts: float | None = None
        self.capture_frames = 0
        self.valid_frames = 0
        self.offsets: list[float] = []
        self.live_metrics: dict[str, float] = {}

    def start(self, ts: float) -> None:
        """从指定时间开始一轮新的平衡检查。"""

        self.reset()
        self.start_ts = float(ts)

    def update(
        self,
        ts: float,
        keypoints: Sequence[Mapping[str, float]],
        pose: str,
    ) -> bool:
        """处理一帧站立姿态；返回值表示本轮采样时间是否结束。"""

        if self.start_ts is None:
            self.start(ts)
        elapsed = max(0.0, float(ts) - float(self.start_ts))
        # 准备期用于站稳和确认身旁有支撑，不纳入摆动统计。
        if elapsed < self.config.balance_warmup_seconds:
            return elapsed >= self.duration_seconds

        self.capture_frames += 1
        frame_metrics = balance_frame_metrics(
            keypoints, self.config.min_keypoint_score
        )
        # standing 只作为安全/质量门槛，本身不是卒中诊断信号。
        if frame_metrics is not None and pose == "standing":
            self.valid_frames += 1
            self.live_metrics = frame_metrics
            self.offsets.append(float(frame_metrics["body_support_offset"]))
        return elapsed >= self.duration_seconds

    def finish(self) -> MotionResult:
        """汇总中心偏移序列，输出持续侧偏、过度摆动或阴性结果。"""

        valid_fraction = self.valid_frames / max(self.capture_frames, 1)
        # 关键点可见且保持站立的帧数不足时，不输出“正常”。
        if (
            len(self.offsets) < self.config.balance_min_valid_samples
            or valid_fraction < self.config.balance_min_valid_fraction
        ):
            return MotionResult(
                status="insufficient",
                reason="stable_standing_pose_not_visible_long_enough",
                quality=valid_fraction,
                metrics={
                    "valid_samples": float(len(self.offsets)),
                    "valid_fraction": valid_fraction,
                },
            )

        center_offset = median(self.offsets)
        # 采用 5%～95% 范围衡量摆动，减少偶发关键点跳变的影响。
        sway_range = percentile(self.offsets, 0.95) - percentile(self.offsets, 0.05)
        metrics = {
            "median_body_support_offset": center_offset,
            "sway_range": sway_range,
            "valid_samples": float(len(self.offsets)),
            "valid_fraction": valid_fraction,
        }
        if abs(center_offset) >= self.config.balance_offset_threshold:
            # 正偏移表示身体中心位于支撑中心右侧，负值则位于左侧。
            return MotionResult(
                status="positive",
                reason="persistent_lateral_body_offset",
                affected_side="right" if center_offset > 0 else "left",
                quality=valid_fraction,
                metrics=metrics,
            )
        if sway_range >= self.config.balance_sway_range_threshold:
            return MotionResult(
                status="positive",
                reason="large_standing_sway",
                quality=valid_fraction,
                metrics=metrics,
            )
        return MotionResult(
            status="negative",
            reason="no_clear_pose_based_balance_asymmetry",
            quality=valid_fraction,
            metrics=metrics,
        )
