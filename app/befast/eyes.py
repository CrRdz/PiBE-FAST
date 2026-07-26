"""E（Eyes）：引导注视左/中/右目标并检测水平眼球运动异常。"""

from __future__ import annotations

from statistics import median

from app.face_landmarker import FaceObservation

from .config import BefastConfig
from .face_geometry import aligned_face_points
from .result import MotionResult


def eye_frame_metrics(
    observation: FaceObservation,
    min_interocular_width: float,
) -> dict[str, float] | None:
    """校正头部滚转后，返回左右虹膜在各自眼裂中的水平位置。"""

    # 468/473 是受试者右/左虹膜中心，其余索引为眼角和鼻尖。
    # 这些编号来自 MediaPipe Face Landmarker 的 478 点拓扑。
    result = aligned_face_points(
        observation,
        (1, 33, 133, 263, 362, 468, 473),
        min_interocular_width,
    )
    if result is None:
        return None
    points, scale = result

    def iris_ratio(iris_index: int, corner_a: int, corner_b: int) -> float | None:
        """计算虹膜在两眼角之间的相对水平位置。"""

        low = min(points[corner_a][0], points[corner_b][0])
        high = max(points[corner_a][0], points[corner_b][0])
        width = high - low
        # 眼裂宽度过小会让虹膜比值极不稳定，该帧直接按无效处理。
        if width <= scale * 0.04:
            return None
        return (points[iris_index][0] - low) / width

    right_gaze = iris_ratio(468, 33, 133)
    left_gaze = iris_ratio(473, 362, 263)
    if right_gaze is None or left_gaze is None:
        return None
    return {
        "left_gaze_x": left_gaze,
        "right_gaze_x": right_gaze,
        "head_x": points[1][0] / scale,
        "interocular_width": scale,
        "inference_ms": observation.inference_ms,
    }


class EyeMovementScreen:
    """按中、左、右三个阶段收集注视样本并评估眼球活动范围。"""

    # 固定顺序同时决定前端目标移动顺序和 samples 的分组键。
    TARGETS = ("center", "left", "right")

    def __init__(self, config: BefastConfig) -> None:
        self.config = config
        self.reset()

    @property
    def duration_seconds(self) -> float:
        """返回三个注视目标所需的总时长。"""

        return self.config.eye_target_seconds * len(self.TARGETS)

    def reset(self) -> None:
        """清空本轮时间、质量计数和各目标样本。"""

        self.start_ts: float | None = None
        self.capture_frames = 0
        self.valid_frames = 0
        self.samples: dict[str, list[dict[str, float]]] = {
            target: [] for target in self.TARGETS
        }
        self.live_metrics: dict[str, float] = {}

    def start(self, ts: float) -> None:
        """从指定时间开始一轮新的眼球运动检查。"""

        self.reset()
        self.start_ts = float(ts)

    def target(self, ts: float) -> str:
        """根据已用时间返回当前应注视的目标。"""

        if self.start_ts is None:
            return self.TARGETS[0]
        elapsed = max(0.0, float(ts) - self.start_ts)
        index = min(
            len(self.TARGETS) - 1,
            int(elapsed / self.config.eye_target_seconds),
        )
        # min 保证超过总时长后仍停留在最后一个目标，不会数组越界。
        return self.TARGETS[index]

    def update(self, ts: float, observation: FaceObservation | None) -> bool:
        """处理一帧面部观察；返回值表示三个目标是否全部采集完毕。"""

        if self.start_ts is None:
            self.start(ts)
        self.capture_frames += 1
        metrics = (
            eye_frame_metrics(observation, self.config.face_min_interocular_width)
            if observation is not None
            else None
        )
        if metrics is not None:
            self.valid_frames += 1
            self.live_metrics = metrics
            self.samples[self.target(ts)].append(metrics)
        elapsed = max(0.0, float(ts) - float(self.start_ts))
        return elapsed >= self.duration_seconds

    def finish(self) -> MotionResult:
        """比较三个目标阶段，输出活动减少、不对称、质量不足或阴性。"""

        valid_fraction = self.valid_frames / max(self.capture_frames, 1)
        counts = {target: len(values) for target, values in self.samples.items()}
        if (
            any(
                count < self.config.eye_min_samples_per_target
                for count in counts.values()
            )
            or valid_fraction < self.config.eye_min_valid_fraction
        ):
            # 每个方向都必须有足够样本，避免只看到某一段就作出判断。
            return MotionResult(
                status="insufficient",
                reason="face_or_irises_not_visible_long_enough",
                quality=valid_fraction,
                metrics={
                    "center_samples": float(counts["center"]),
                    "left_samples": float(counts["left"]),
                    "right_samples": float(counts["right"]),
                    "valid_fraction": valid_fraction,
                },
            )

        medians = {
            target: {
                name: median(sample[name] for sample in values)
                for name in ("left_gaze_x", "right_gaze_x", "head_x")
            }
            for target, values in self.samples.items()
        }
        # 左目标到右目标的虹膜位置差，代表每只眼的水平活动范围。
        left_range = abs(
            medians["right"]["left_gaze_x"] - medians["left"]["left_gaze_x"]
        )
        right_range = abs(
            medians["right"]["right_gaze_x"] - medians["left"]["right_gaze_x"]
        )
        conjugacy_errors: list[float] = []
        # 共轭误差比较双眼相对中央注视的位移是否一致。
        for target in ("left", "right"):
            left_delta = (
                medians[target]["left_gaze_x"] - medians["center"]["left_gaze_x"]
            )
            right_delta = (
                medians[target]["right_gaze_x"]
                - medians["center"]["right_gaze_x"]
            )
            conjugacy_errors.append(abs(left_delta - right_delta))
        conjugacy_error = max(conjugacy_errors)
        head_positions = [values["head_x"] for values in medians.values()]
        # 鼻尖相对双眼中心移动过多，说明用户可能用转头代替转眼。
        head_motion = max(head_positions) - min(head_positions)
        range_asymmetry = abs(left_range - right_range)
        metrics = {
            "left_gaze_range": left_range,
            "right_gaze_range": right_range,
            "range_asymmetry": range_asymmetry,
            "max_conjugacy_error": conjugacy_error,
            "head_motion": head_motion,
            "valid_fraction": valid_fraction,
        }

        if head_motion >= self.config.eye_head_motion_threshold:
            return MotionResult(
                status="insufficient",
                reason="head_moved_during_eye_test",
                quality=valid_fraction,
                metrics=metrics,
            )
        if (
            left_range < self.config.eye_gaze_range_threshold
            and right_range < self.config.eye_gaze_range_threshold
        ):
            # 两只眼的活动范围都小，表示对左右视觉目标的可见反应减少。
            return MotionResult(
                status="positive",
                reason="reduced_visual_target_response",
                quality=valid_fraction,
                metrics=metrics,
            )
        if range_asymmetry >= self.config.eye_range_asymmetry_threshold:
            # 活动范围较小的一侧作为可能受影响侧，仅供筛查提示。
            return MotionResult(
                status="positive",
                reason="asymmetric_eye_excursion",
                affected_side="left" if left_range < right_range else "right",
                quality=valid_fraction,
                metrics=metrics,
            )
        if conjugacy_error >= self.config.eye_conjugacy_threshold:
            return MotionResult(
                status="positive",
                reason="asymmetric_conjugate_gaze",
                quality=valid_fraction,
                metrics=metrics,
            )
        return MotionResult(
            status="negative",
            reason="no_visible_eye_movement_abnormality",
            quality=valid_fraction,
            metrics=metrics,
        )
