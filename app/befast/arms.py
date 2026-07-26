"""A（Arm weakness）：引导用户抬臂并检测高度不对称或单侧下垂。"""

from __future__ import annotations

from math import hypot
from statistics import median
from typing import Mapping, Sequence

from app.keypoints import keypoint_map

from .config import BefastConfig
from .pose_geometry import visible_point
from .result import MotionResult


def arm_frame_metrics(
    keypoints: Sequence[Mapping[str, float]], min_score: float
) -> dict[str, float] | None:
    """从单帧 MoveNet 关键点计算左右手腕相对肩膀的归一化高度。"""

    points = keypoint_map(keypoints)
    # A 检查只依赖双肩和双腕；任何一点不可见都不能形成有效样本。
    left_shoulder = visible_point(points, "left_shoulder", min_score)
    right_shoulder = visible_point(points, "right_shoulder", min_score)
    left_wrist = visible_point(points, "left_wrist", min_score)
    right_wrist = visible_point(points, "right_wrist", min_score)
    if not all((left_shoulder, right_shoulder, left_wrist, right_wrist)):
        return None
    assert left_shoulder and right_shoulder and left_wrist and right_wrist
    shoulder_width = hypot(
        right_shoulder[0] - left_shoulder[0],
        right_shoulder[1] - left_shoulder[1],
    )
    if shoulder_width <= 1e-4:
        return None
    # 图像坐标 y 向下增大，因此正值表示手腕位于同侧肩膀下方。
    left_relative_y = (left_wrist[1] - left_shoulder[1]) / shoulder_width
    right_relative_y = (right_wrist[1] - right_shoulder[1]) / shoulder_width
    return {
        "left_wrist_relative_y": left_relative_y,
        "right_wrist_relative_y": right_relative_y,
        "level_difference": left_relative_y - right_relative_y,
        "shoulder_width": shoulder_width,
    }


class ArmDriftScreen:
    """在引导式抬臂过程中累计样本并判断左右手臂不对称。"""

    def __init__(self, config: BefastConfig) -> None:
        self.config = config
        self.reset()

    @property
    def duration_seconds(self) -> float:
        """返回准备期与正式采样期的总时长。"""

        return self.config.arm_warmup_seconds + self.config.arm_capture_seconds

    def reset(self) -> None:
        """清空本轮时间、质量计数和高度样本。"""

        self.start_ts: float | None = None
        self.capture_frames = 0
        self.valid_frames = 0
        self.samples: list[tuple[float, float, float]] = []
        self.live_metrics: dict[str, float] = {}

    def start(self, ts: float) -> None:
        """从指定时间开始一轮新的抬臂检查。"""

        self.reset()
        self.start_ts = float(ts)

    def update(
        self, ts: float, keypoints: Sequence[Mapping[str, float]]
    ) -> bool:
        """处理一帧关键点；返回值表示本轮采样时间是否结束。"""

        if self.start_ts is None:
            self.start(ts)
        elapsed = max(0.0, float(ts) - float(self.start_ts))
        # 准备期允许用户抬起手臂，不把动作过程计入漂移样本。
        if elapsed < self.config.arm_warmup_seconds:
            return elapsed >= self.duration_seconds

        self.capture_frames += 1
        frame_metrics = arm_frame_metrics(keypoints, self.config.min_keypoint_score)
        if frame_metrics is not None:
            self.valid_frames += 1
            self.live_metrics = frame_metrics
            self.samples.append(
                (
                    float(frame_metrics["left_wrist_relative_y"]),
                    float(frame_metrics["right_wrist_relative_y"]),
                    float(frame_metrics["level_difference"]),
                )
            )
        return elapsed >= self.duration_seconds

    def finish(self) -> MotionResult:
        """汇总有效样本，输出阳性、阴性或质量不足。"""

        valid_fraction = self.valid_frames / max(self.capture_frames, 1)
        # 数量和比例双重门槛可避免低帧率或短暂遮挡被误判为正常。
        if (
            len(self.samples) < self.config.arm_min_valid_samples
            or valid_fraction < self.config.arm_min_valid_fraction
        ):
            return MotionResult(
                status="insufficient",
                reason="arms_not_visible_long_enough",
                quality=valid_fraction,
                metrics={
                    "valid_samples": float(len(self.samples)),
                    "valid_fraction": valid_fraction,
                },
            )

        segment = max(1, len(self.samples) // 3)
        first = self.samples[:segment]
        last = self.samples[-segment:]
        # 用前后各三分之一的中位数估算下垂量，比首尾单帧更抗抖动。
        initial_left = median(sample[0] for sample in first)
        initial_right = median(sample[1] for sample in first)
        final_left = median(sample[0] for sample in last)
        final_right = median(sample[1] for sample in last)
        level_difference = median(sample[2] for sample in self.samples)
        left_drop = final_left - initial_left
        right_drop = final_right - initial_right
        drift_difference = left_drop - right_drop
        metrics = {
            "level_difference": level_difference,
            "left_drop": left_drop,
            "right_drop": right_drop,
            "drift_difference": drift_difference,
            "initial_left_wrist_relative_y": initial_left,
            "initial_right_wrist_relative_y": initial_right,
            "valid_samples": float(len(self.samples)),
            "valid_fraction": valid_fraction,
        }

        level_abnormal = (
            abs(level_difference) >= self.config.arm_level_difference_threshold
        )
        drift_abnormal = (
            abs(drift_difference) >= self.config.arm_drift_difference_threshold
        )
        if level_abnormal or drift_abnormal:
            # 持续高度差优先于动态漂移差，便于给出更直观的结果原因。
            side_signal = level_difference if level_abnormal else drift_difference
            return MotionResult(
                status="positive",
                reason=(
                    "persistent_arm_height_asymmetry"
                    if level_abnormal
                    else "asymmetric_arm_drift"
                ),
                # 图像 y 向下增大，所以正信号代表受试者左臂更低。
                affected_side="left" if side_signal > 0 else "right",
                quality=valid_fraction,
                metrics=metrics,
            )

        if (
            initial_left > self.config.arm_max_initial_wrist_below_shoulder
            or initial_right > self.config.arm_max_initial_wrist_below_shoulder
        ):
            # 两臂没有达到基本抬起高度时，不能把“小差异”解释为阴性。
            return MotionResult(
                status="insufficient",
                reason="both_arms_were_not_held_up",
                quality=valid_fraction,
                metrics=metrics,
            )
        return MotionResult(
            status="negative",
            reason="no_clear_arm_asymmetry",
            quality=valid_fraction,
            metrics=metrics,
        )
