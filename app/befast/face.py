"""F（Face）：比较中性表情和微笑阶段，检测下半脸运动不对称。"""

from __future__ import annotations

from statistics import median

from app.face_landmarker import FaceObservation

from .config import BefastConfig
from .face_geometry import aligned_face_points
from .result import MotionResult


def face_frame_metrics(
    observation: FaceObservation,
    min_interocular_width: float,
) -> dict[str, float] | None:
    """返回滚转校正后的嘴角高度差和左右微笑 blendshape 强度。"""

    # 291 是受试者左嘴角，61 是右嘴角；33/263 用于眼距归一化。
    result = aligned_face_points(
        observation,
        (33, 61, 263, 291),
        min_interocular_width,
    )
    if result is None:
        return None
    points, scale = result
    return {
        "mouth_corner_y_difference": (points[291][1] - points[61][1]) / scale,
        "left_smile": float(observation.blendshapes.get("mouthSmileLeft", 0.0)),
        "right_smile": float(observation.blendshapes.get("mouthSmileRight", 0.0)),
        "interocular_width": scale,
        "inference_ms": observation.inference_ms,
    }


class FaceSmileScreen:
    """以中性表情为个人基线，评估微笑时左右下半脸的运动差异。"""

    def __init__(self, config: BefastConfig) -> None:
        self.config = config
        self.reset()

    @property
    def duration_seconds(self) -> float:
        """返回中性阶段与微笑阶段的总时长。"""

        return self.config.face_neutral_seconds + self.config.face_smile_seconds

    def reset(self) -> None:
        """清空本轮时间、质量计数和两个表情阶段的样本。"""

        self.start_ts: float | None = None
        self.capture_frames = 0
        self.valid_frames = 0
        self.neutral_samples: list[dict[str, float]] = []
        self.smile_samples: list[dict[str, float]] = []
        self.live_metrics: dict[str, float] = {}

    def start(self, ts: float) -> None:
        """从指定时间开始一轮新的面部检查。"""

        self.reset()
        self.start_ts = float(ts)

    def phase(self, ts: float) -> str:
        """根据已用时间返回当前应保持中性还是微笑。"""

        if self.start_ts is None:
            return "neutral"
        elapsed = max(0.0, float(ts) - self.start_ts)
        return "neutral" if elapsed < self.config.face_neutral_seconds else "smile"

    def update(self, ts: float, observation: FaceObservation | None) -> bool:
        """处理一帧面部观察，并放入当前表情阶段的样本组。"""

        if self.start_ts is None:
            self.start(ts)
        self.capture_frames += 1
        metrics = (
            face_frame_metrics(observation, self.config.face_min_interocular_width)
            if observation is not None
            else None
        )
        if metrics is not None:
            self.valid_frames += 1
            self.live_metrics = metrics
            if self.phase(ts) == "neutral":
                self.neutral_samples.append(metrics)
            else:
                self.smile_samples.append(metrics)
        elapsed = max(0.0, float(ts) - float(self.start_ts))
        return elapsed >= self.duration_seconds

    def finish(self) -> MotionResult:
        """比较中性和微笑阶段，输出面部不对称、质量不足或阴性。"""

        valid_fraction = self.valid_frames / max(self.capture_frames, 1)
        if (
            len(self.neutral_samples) < self.config.face_min_samples_per_phase
            or len(self.smile_samples) < self.config.face_min_samples_per_phase
            or valid_fraction < self.config.face_min_valid_fraction
        ):
            # 两个阶段都必须达到样本门槛，才能计算“相对基线”的变化。
            return MotionResult(
                status="insufficient",
                reason="face_not_visible_during_neutral_and_smile",
                quality=valid_fraction,
                metrics={
                    "neutral_samples": float(len(self.neutral_samples)),
                    "smile_samples": float(len(self.smile_samples)),
                    "valid_fraction": valid_fraction,
                },
            )

        neutral_corner = median(
            sample["mouth_corner_y_difference"] for sample in self.neutral_samples
        )
        smile_corner = median(
            sample["mouth_corner_y_difference"] for sample in self.smile_samples
        )
        left_smile = median(sample["left_smile"] for sample in self.smile_samples)
        right_smile = median(sample["right_smile"] for sample in self.smile_samples)
        smile_strength = max(left_smile, right_smile)
        # 减去中性基线，可降低静态脸型或相机角度造成的固定嘴角差。
        corner_delta = smile_corner - neutral_corner
        # 正值表示受试者左侧激活弱于右侧。
        smile_weakness = right_smile - left_smile
        metrics = {
            "neutral_mouth_corner_difference": neutral_corner,
            "smile_mouth_corner_difference": smile_corner,
            "mouth_corner_delta": corner_delta,
            "left_smile_score": left_smile,
            "right_smile_score": right_smile,
            "smile_score_difference": smile_weakness,
            "smile_strength": smile_strength,
            "valid_fraction": valid_fraction,
        }

        if smile_strength < self.config.face_min_smile_score:
            # 没检测到明确微笑时不能把左右差异当作可靠的运动结果。
            return MotionResult(
                status="insufficient",
                reason="smile_expression_not_detected",
                quality=valid_fraction,
                metrics=metrics,
            )
        corner_abnormal = (
            abs(corner_delta) >= self.config.face_corner_delta_threshold
        )
        score_abnormal = (
            abs(smile_weakness)
            >= self.config.face_smile_score_difference_threshold
        )
        if corner_abnormal or score_abnormal:
            # 优先使用相对基线的嘴角变化，其次使用模型的微笑激活差。
            side_signal = corner_delta if corner_abnormal else smile_weakness
            return MotionResult(
                status="positive",
                reason=(
                    "asymmetric_mouth_corner_movement"
                    if corner_abnormal
                    else "asymmetric_smile_activation"
                ),
                affected_side="left" if side_signal > 0 else "right",
                quality=valid_fraction,
                metrics=metrics,
            )
        return MotionResult(
            status="negative",
            reason="no_clear_smile_asymmetry",
            quality=valid_fraction,
            metrics=metrics,
        )
