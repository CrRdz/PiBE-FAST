"""A（Arms）：验证完整双臂平举动作，再评估保持阶段的左右不对称。"""

from __future__ import annotations

from math import acos, degrees, hypot
from statistics import median
from typing import Mapping, Sequence

from app.action_quality_model import summarize_action_quality
from app.arm_model import LinearBinaryModel
from app.compensation_model import (
    COMMON_JOINTS,
    compensation_frame_features,
    load_compensation_shadow_models,
    summarize_compensation_frames,
)
from app.keypoints import keypoint_map

from .config import BefastConfig
from .pose_geometry import visible_point
from .result import MotionResult


def _angle(a: tuple[float, float], b: tuple[float, float], c: tuple[float, float]) -> float:
    first = (a[0] - b[0], a[1] - b[1])
    second = (c[0] - b[0], c[1] - b[1])
    first_length = hypot(*first)
    second_length = hypot(*second)
    if first_length <= 1e-6 or second_length <= 1e-6:
        return 0.0
    cosine = (first[0] * second[0] + first[1] * second[1]) / (
        first_length * second_length
    )
    return degrees(acos(max(-1.0, min(1.0, cosine))))


def arm_frame_metrics(
    keypoints: Sequence[Mapping[str, float]], min_score: float
) -> dict[str, float] | None:
    """计算与距离无关的手腕高度、肘角和横向展开程度。"""

    points = keypoint_map(keypoints)
    names = (
        "left_shoulder",
        "right_shoulder",
        "left_elbow",
        "right_elbow",
        "left_wrist",
        "right_wrist",
    )
    visible = {name: visible_point(points, name, min_score) for name in names}
    if not all(visible.values()):
        return None
    left_shoulder = visible["left_shoulder"]
    right_shoulder = visible["right_shoulder"]
    left_elbow = visible["left_elbow"]
    right_elbow = visible["right_elbow"]
    left_wrist = visible["left_wrist"]
    right_wrist = visible["right_wrist"]
    assert all(
        point is not None
        for point in (
            left_shoulder,
            right_shoulder,
            left_elbow,
            right_elbow,
            left_wrist,
            right_wrist,
        )
    )
    shoulder_width = hypot(
        right_shoulder[0] - left_shoulder[0],
        right_shoulder[1] - left_shoulder[1],
    )
    if shoulder_width <= 1e-4:
        return None
    left_relative_y = (left_wrist[1] - left_shoulder[1]) / shoulder_width
    right_relative_y = (right_wrist[1] - right_shoulder[1]) / shoulder_width
    return {
        "left_wrist_relative_y": left_relative_y,
        "right_wrist_relative_y": right_relative_y,
        "level_difference": left_relative_y - right_relative_y,
        "left_elbow_angle_degrees": _angle(left_shoulder, left_elbow, left_wrist),
        "right_elbow_angle_degrees": _angle(right_shoulder, right_elbow, right_wrist),
        "left_lateral_reach": abs(left_wrist[0] - left_shoulder[0]) / shoulder_width,
        "right_lateral_reach": abs(right_wrist[0] - right_shoulder[0]) / shoulder_width,
        "shoulder_width": shoulder_width,
    }


def compensation_keypoint_features(
    keypoints: Sequence[Mapping[str, float]], min_score: float
) -> dict[str, float] | None:
    """提取 Toronto 研究模型需要的八关节二维特征。"""

    mapped = keypoint_map(keypoints)
    points = {name: visible_point(mapped, name, min_score) for name in COMMON_JOINTS}
    if not all(points.values()):
        return None
    try:
        return compensation_frame_features(
            {name: point for name, point in points.items() if point is not None}
        )
    except ValueError:
        return None


class ArmDriftScreen:
    """双臂侧平举状态机；旧类名保留以兼容会话接口。"""

    PHASES = ("raise", "hold", "lower")
    PHASE_STATES = ("preview", "countdown", "recording", "retry", "complete")

    def __init__(self, config: BefastConfig) -> None:
        self.config = config
        self.compensation_models = {}
        self.compensation_model_errors: list[str] = []
        if config.arm_compensation_shadow_enabled:
            self.compensation_models, self.compensation_model_errors = (
                load_compensation_shadow_models(
                    config.arm_compensation_shadow_model_dir
                )
            )
        self.action_quality_model = None
        self.action_quality_model_error: str | None = None
        if config.arm_action_quality_shadow_enabled:
            try:
                self.action_quality_model = LinearBinaryModel.load(
                    config.arm_action_quality_shadow_model_path,
                    expected_component="A_ACTION_QUALITY_SHADOW",
                )
            except FileNotFoundError:
                pass
            except (OSError, TypeError, ValueError) as exc:
                self.action_quality_model_error = str(exc)
        self.reset()

    @property
    def duration_seconds(self) -> float:
        return (
            self.config.arm_phase_countdown_seconds
            + self.config.arm_raise_timeout_seconds
            + self.config.arm_hold_seconds
            + self.config.arm_lower_timeout_seconds
        )

    def reset(self) -> None:
        self.start_ts: float | None = None
        self.phase_state = "preview"
        self.phase_index = 0
        self.phase_state_started_at: float | None = None
        self.pose_reached_since: float | None = None
        self.auto_ready_since: float | None = None
        self.phase_failure_reason: str | None = None
        self.start_pose_observed = False
        self.capture_frames = 0
        self.valid_frames = 0
        self.hold_capture_frames = 0
        self.hold_valid_frames = 0
        self.samples: list[tuple[float, float, float]] = []
        self.compensation_frames: list[dict[str, float]] = []
        self.live_metrics: dict[str, float] = {}
        self.phase_completion_metrics: dict[str, dict[str, float]] = {}

    def start(self, ts: float) -> None:
        self.reset()
        self.start_ts = float(ts)

    @property
    def current_phase(self) -> str:
        return self.PHASES[min(self.phase_index, len(self.PHASES) - 1)]

    def ready_phase(self, ts: float) -> None:
        """用户确认看懂完整动作后，清空旧尝试并开始三秒倒数。"""

        if self.phase_state not in {"preview", "retry"}:
            raise ValueError("the arm action is not waiting for readiness")
        self.reset()
        self.start_ts = float(ts)
        self.phase_state = "countdown"
        self.phase_state_started_at = float(ts)

    def _phase_duration(self) -> float:
        return {
            "raise": self.config.arm_raise_timeout_seconds,
            "hold": self.config.arm_hold_seconds,
            "lower": self.config.arm_lower_timeout_seconds,
        }[self.current_phase]

    def _sync_countdown(self, ts: float) -> None:
        if self.phase_state != "countdown" or self.phase_state_started_at is None:
            return
        if float(ts) - self.phase_state_started_at >= self.config.arm_phase_countdown_seconds:
            self.phase_state_started_at += self.config.arm_phase_countdown_seconds
            self.phase_state = "recording"

    def auto_ready_status(self, ts: float) -> tuple[bool, float]:
        """Return whether the hands-free start pose is active and its remaining time."""

        if self.auto_ready_since is None:
            return False, float(self.config.arm_auto_ready_seconds)
        remaining = self.config.arm_auto_ready_seconds - (
            float(ts) - self.auto_ready_since
        )
        return True, max(0.0, remaining)

    def _update_auto_ready(
        self, ts: float, metrics: Mapping[str, float] | None
    ) -> bool:
        """Start countdown hands-free after a stable, arms-down preview pose."""

        if not self.config.arm_auto_ready_enabled:
            self.auto_ready_since = None
            return False
        preview_elapsed = (
            float(ts) - self.start_ts if self.start_ts is not None else 0.0
        )
        preview_seen = (
            self.phase_state == "retry"
            or preview_elapsed >= self.config.arm_preview_min_seconds
        )
        if metrics is None or not preview_seen or not self._arms_down(metrics):
            self.auto_ready_since = None
            return False
        if self.auto_ready_since is None:
            self.auto_ready_since = float(ts)
            return False
        if float(ts) - self.auto_ready_since < self.config.arm_auto_ready_seconds:
            return False
        self.ready_phase(ts)
        return True

    def phase(self, ts: float) -> str:
        self._sync_countdown(ts)
        return self.current_phase

    def phase_status(self, ts: float) -> tuple[int, float]:
        self._sync_countdown(ts)
        if self.phase_state_started_at is None:
            return self.phase_index, self._phase_duration()
        if self.phase_state == "countdown":
            remaining = self.config.arm_phase_countdown_seconds - (
                float(ts) - self.phase_state_started_at
            )
        elif self.phase_state == "recording":
            remaining = self._phase_duration() - (
                float(ts) - self.phase_state_started_at
            )
        else:
            remaining = 0.0
        return self.phase_index, max(0.0, remaining)

    def progress(self, ts: float) -> float:
        self._sync_countdown(ts)
        if self.phase_state == "complete":
            return 1.0
        weights = (0.30, 0.50, 0.20)
        completed = sum(weights[: self.phase_index])
        if self.phase_state == "recording" and self.phase_state_started_at is not None:
            fraction = min(
                1.0,
                max(0.0, float(ts) - self.phase_state_started_at)
                / max(self._phase_duration(), 0.01),
            )
            completed += weights[self.phase_index] * fraction
        return completed

    def _arms_down(self, metrics: Mapping[str, float]) -> bool:
        minimum = self.config.arm_start_wrist_below_shoulder
        return (
            metrics["left_wrist_relative_y"] >= minimum
            and metrics["right_wrist_relative_y"] >= minimum
        )

    def _arms_raised(self, metrics: Mapping[str, float]) -> bool:
        height = self.config.arm_raise_wrist_height_tolerance
        elbow = self.config.arm_min_elbow_angle_degrees
        reach = self.config.arm_min_lateral_reach
        return (
            metrics["left_wrist_relative_y"] <= height
            and metrics["right_wrist_relative_y"] <= height
            and metrics["left_elbow_angle_degrees"] >= elbow
            and metrics["right_elbow_angle_degrees"] >= elbow
            and metrics["left_lateral_reach"] >= reach
            and metrics["right_lateral_reach"] >= reach
        )

    def _sustained(self, ts: float, condition: bool) -> bool:
        if not condition:
            self.pose_reached_since = None
            return False
        if self.pose_reached_since is None:
            self.pose_reached_since = float(ts)
            return False
        return float(ts) - self.pose_reached_since >= self.config.arm_pose_sustain_seconds

    def _advance_phase(self, ts: float, metrics: Mapping[str, float]) -> None:
        phase = self.current_phase
        self.phase_completion_metrics[phase] = {
            "completed": 1.0,
            "completed_at": float(ts),
            **{key: float(value) for key, value in metrics.items()},
        }
        self.phase_index += 1
        self.pose_reached_since = None
        if self.phase_index >= len(self.PHASES):
            self.phase_index = len(self.PHASES) - 1
            self.phase_state = "complete"
            self.phase_state_started_at = None
        else:
            self.phase_state_started_at = float(ts)

    def _fail(self, reason: str) -> None:
        self.phase_state = "retry"
        self.phase_state_started_at = None
        self.pose_reached_since = None
        self.phase_failure_reason = reason

    def update(
        self, ts: float, keypoints: Sequence[Mapping[str, float]]
    ) -> bool:
        """处理一帧；仅在完整完成抬起、保持和放下后返回 True。"""

        if self.start_ts is None:
            self.start(ts)
        self._sync_countdown(ts)
        if self.phase_state in {"preview", "retry"}:
            preview_metrics = arm_frame_metrics(
                keypoints, self.config.min_keypoint_score
            )
            if preview_metrics is not None:
                self.live_metrics = dict(preview_metrics)
            self._update_auto_ready(ts, preview_metrics)
            return False
        if self.phase_state != "recording":
            return self.phase_state == "complete"

        self.capture_frames += 1
        metrics = arm_frame_metrics(keypoints, self.config.min_keypoint_score)
        shadow = compensation_keypoint_features(keypoints, self.config.min_keypoint_score)
        if shadow is not None:
            self.compensation_frames.append(shadow)
        if metrics is not None:
            self.valid_frames += 1
            self.live_metrics = dict(metrics)

        assert self.phase_state_started_at is not None
        elapsed = float(ts) - self.phase_state_started_at
        phase = self.current_phase
        if phase == "raise":
            if metrics is not None and self._arms_down(metrics):
                self.start_pose_observed = True
            reached = (
                metrics is not None
                and self.start_pose_observed
                and self._arms_raised(metrics)
            )
            if self._sustained(ts, reached):
                self._advance_phase(ts, metrics or {})
            elif elapsed >= self.config.arm_raise_timeout_seconds:
                reason = (
                    "arm_start_pose_not_observed"
                    if not self.start_pose_observed
                    else "both_arms_were_not_raised"
                )
                self._fail(reason)
        elif phase == "hold":
            self.hold_capture_frames += 1
            if metrics is not None and self._arms_raised(metrics):
                self.hold_valid_frames += 1
                self.samples.append(
                    (
                        float(metrics["left_wrist_relative_y"]),
                        float(metrics["right_wrist_relative_y"]),
                        float(metrics["level_difference"]),
                    )
                )
            if elapsed >= self.config.arm_hold_seconds:
                valid_fraction = self.hold_valid_frames / max(self.hold_capture_frames, 1)
                self.live_metrics["hold_valid_fraction"] = valid_fraction
                if (
                    len(self.samples) < self.config.arm_min_valid_samples
                    or valid_fraction < self.config.arm_min_valid_fraction
                ):
                    self._fail("both_arms_were_not_held_up")
                else:
                    self._advance_phase(ts, self.live_metrics)
        else:
            lowered = metrics is not None and self._arms_down(metrics)
            if self._sustained(ts, lowered):
                self._advance_phase(ts, metrics or {})
            elif elapsed >= self.config.arm_lower_timeout_seconds:
                self._fail("both_arms_were_not_lowered")
        return self.phase_state == "complete"

    def _compensation_shadow_output(self) -> tuple[dict[str, float], dict[str, str]]:
        """输出研究概率；这些值永远不参与正式 A 判定。"""

        size = max(10, int(self.config.arm_compensation_shadow_window_frames))
        if not self.compensation_models or len(self.compensation_frames) < 10:
            return {}, {"compensation_shadow_mode": "research_only_no_decision"}
        windows = [
            self.compensation_frames[start : start + size]
            for start in range(0, len(self.compensation_frames), size)
            if len(self.compensation_frames[start : start + size]) >= 10
        ]
        if not windows:
            return {}, {"compensation_shadow_mode": "research_only_no_decision"}
        summaries = [summarize_compensation_frames(window) for window in windows]
        output: dict[str, float] = {
            "shadow_compensation_window_count": float(len(summaries))
        }
        versions = []
        for target, model in self.compensation_models.items():
            probabilities = [model.predict_probability(row) for row in summaries]
            output[f"shadow_{target}_probability_median"] = median(probabilities)
            output[f"shadow_{target}_probability_max"] = max(probabilities)
            output[f"shadow_{target}_decision_threshold"] = model.decision_threshold
            versions.append(model.version)
        return output, {
            "compensation_shadow_mode": "research_only_no_decision",
            "compensation_shadow_models": ",".join(versions),
        }

    def _action_quality_shadow_output(self) -> tuple[dict[str, float], dict[str, str]]:
        """Evaluate each arm with IntelliRehabDS; never gate the Web result."""

        details = {"action_quality_shadow_mode": "research_only_no_decision"}
        if self.action_quality_model is None or len(self.compensation_frames) < 10:
            return {}, details
        probabilities = {}
        for side in ("left", "right"):
            features = summarize_action_quality(self.compensation_frames, side)
            probabilities[side] = self.action_quality_model.predict_probability(features)
        details["action_quality_shadow_model"] = self.action_quality_model.version
        return {
            "shadow_action_invalid_probability_left": probabilities["left"],
            "shadow_action_invalid_probability_right": probabilities["right"],
            "shadow_action_invalid_probability_max": max(probabilities.values()),
            "shadow_action_invalid_decision_threshold": (
                self.action_quality_model.decision_threshold
            ),
        }, details

    def finish(self) -> MotionResult:
        """完成门控通过后，汇总保持阶段的高度差和单侧下落。"""

        if self.phase_state != "complete":
            return MotionResult(
                status="insufficient",
                reason=self.phase_failure_reason or "arm_action_not_completed",
                quality=0.0,
            )
        valid_fraction = self.hold_valid_frames / max(self.hold_capture_frames, 1)
        if len(self.samples) < self.config.arm_min_valid_samples:
            return MotionResult(
                status="insufficient",
                reason="arms_not_visible_long_enough",
                quality=valid_fraction,
            )

        segment = max(1, len(self.samples) // 3)
        first = self.samples[:segment]
        last = self.samples[-segment:]
        initial_left = median(value[0] for value in first)
        initial_right = median(value[1] for value in first)
        final_left = median(value[0] for value in last)
        final_right = median(value[1] for value in last)
        level_difference = median(value[2] for value in self.samples)
        left_drop = final_left - initial_left
        right_drop = final_right - initial_right
        drift_difference = left_drop - right_drop
        metrics = {
            "action_completed": 1.0,
            "action_completion_score": valid_fraction,
            "valid_samples": float(len(self.samples)),
            "valid_fraction": valid_fraction,
            "hold_duration_seconds": float(self.config.arm_hold_seconds),
            "level_difference": level_difference,
            "left_drop": left_drop,
            "right_drop": right_drop,
            "drift_difference": drift_difference,
            "initial_left_wrist_relative_y": initial_left,
            "initial_right_wrist_relative_y": initial_right,
        }
        shadow_metrics, shadow_details = self._compensation_shadow_output()
        metrics.update(shadow_metrics)
        quality_metrics, quality_details = self._action_quality_shadow_output()
        metrics.update(quality_metrics)
        shadow_details.update(quality_details)

        height_positive = abs(level_difference) >= self.config.arm_level_difference_threshold
        drift_positive = abs(drift_difference) >= self.config.arm_drift_difference_threshold
        if not (height_positive or drift_positive):
            return MotionResult(
                status="negative",
                reason="no_clear_arm_asymmetry",
                quality=valid_fraction,
                metrics=metrics,
                details=shadow_details,
            )
        height_score = level_difference / max(self.config.arm_level_difference_threshold, 1e-6)
        drift_score = drift_difference / max(self.config.arm_drift_difference_threshold, 1e-6)
        dominant = height_score if abs(height_score) >= abs(drift_score) else drift_score
        affected_side = "left" if dominant > 0 else "right"
        reason = "persistent_arm_height_asymmetry" if height_positive else "asymmetric_arm_drift"
        return MotionResult(
            status="positive",
            reason=reason,
            affected_side=affected_side,
            quality=valid_fraction,
            metrics=metrics,
            details=shadow_details,
        )
