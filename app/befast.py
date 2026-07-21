"""Explainable, dataset-free BE-FAST screening helpers.

This module is deliberately conservative. MoveNet quantifies arm drift (A) and
coarse standing balance (B). MediaPipe Face Landmarker supports a guided visible
gaze-response proxy (E) and neutral-to-smile facial symmetry check (F). Speech
(S) still requires an explicit observation. The result is a screening prompt,
never a stroke diagnosis or a substitute for asking about vision symptoms.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import atan2, cos, hypot, sin
from statistics import median
import threading
import time
from typing import Any, Mapping, Sequence

from app.keypoints import keypoint_map
from app.face_landmarker import FaceObservation


Point = tuple[float, float]


@dataclass(frozen=True)
class BefastConfig:
    """Engineering defaults that must be validated before clinical use."""

    min_keypoint_score: float = 0.35

    arm_warmup_seconds: float = 1.5
    arm_capture_seconds: float = 6.0
    arm_min_valid_samples: int = 20
    arm_min_valid_fraction: float = 0.55
    # All arm distances are normalized by shoulder width.
    arm_level_difference_threshold: float = 0.30
    arm_drift_difference_threshold: float = 0.22
    arm_max_initial_wrist_below_shoulder: float = 0.85

    # MediaPipe Face Landmarker is sampled at a lower rate than the camera.
    face_inference_fps: float = 5.0
    face_min_interocular_width: float = 0.075

    # Three seconds gives the user time to notice, refocus, and hold each target.
    eye_target_seconds: float = 3.0
    eye_min_samples_per_target: int = 4
    eye_min_valid_fraction: float = 0.50
    eye_gaze_range_threshold: float = 0.12
    eye_range_asymmetry_threshold: float = 0.10
    eye_conjugacy_threshold: float = 0.14
    eye_head_motion_threshold: float = 0.35

    face_neutral_seconds: float = 2.0
    face_smile_seconds: float = 3.0
    face_min_samples_per_phase: int = 5
    face_min_valid_fraction: float = 0.50
    face_min_smile_score: float = 0.22
    face_corner_delta_threshold: float = 0.075
    face_smile_score_difference_threshold: float = 0.24

    balance_warmup_seconds: float = 1.5
    balance_capture_seconds: float = 6.0
    balance_min_valid_samples: int = 20
    balance_min_valid_fraction: float = 0.55
    # Body-center offset and sway are normalized by shoulder width.
    balance_offset_threshold: float = 0.40
    balance_sway_range_threshold: float = 0.50


@dataclass(frozen=True)
class MotionResult:
    status: str = "not_run"
    reason: str = "not_run"
    affected_side: str | None = None
    quality: float = 0.0
    metrics: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "reason": self.reason,
            "affected_side": self.affected_side,
            "quality": round(self.quality, 4),
            "metrics": {
                key: round(float(value), 5) for key, value in self.metrics.items()
            },
        }


class ArmDriftScreen:
    """Measure left/right arm-height asymmetry during a guided arm hold."""

    def __init__(self, config: BefastConfig) -> None:
        self.config = config
        self.reset()

    @property
    def duration_seconds(self) -> float:
        return self.config.arm_warmup_seconds + self.config.arm_capture_seconds

    def reset(self) -> None:
        self.start_ts: float | None = None
        self.capture_frames = 0
        self.valid_frames = 0
        self.samples: list[tuple[float, float, float]] = []
        self.live_metrics: dict[str, float] = {}

    def start(self, ts: float) -> None:
        self.reset()
        self.start_ts = float(ts)

    def update(
        self, ts: float, keypoints: Sequence[Mapping[str, float]]
    ) -> bool:
        if self.start_ts is None:
            self.start(ts)
        elapsed = max(0.0, float(ts) - float(self.start_ts))
        if elapsed < self.config.arm_warmup_seconds:
            return elapsed >= self.duration_seconds

        self.capture_frames += 1
        frame_metrics = _arm_frame_metrics(keypoints, self.config.min_keypoint_score)
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
        valid_fraction = self.valid_frames / max(self.capture_frames, 1)
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
            side_signal = level_difference if level_abnormal else drift_difference
            # Image y grows downwards: a positive value means the left arm is lower.
            affected_side = "left" if side_signal > 0 else "right"
            reason = (
                "persistent_arm_height_asymmetry"
                if level_abnormal
                else "asymmetric_arm_drift"
            )
            return MotionResult(
                status="positive",
                reason=reason,
                affected_side=affected_side,
                quality=valid_fraction,
                metrics=metrics,
            )

        if (
            initial_left > self.config.arm_max_initial_wrist_below_shoulder
            or initial_right > self.config.arm_max_initial_wrist_below_shoulder
        ):
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


class BalanceScreen:
    """Measure coarse standing center offset and sway from MoveNet landmarks."""

    def __init__(self, config: BefastConfig) -> None:
        self.config = config
        self.reset()

    @property
    def duration_seconds(self) -> float:
        return self.config.balance_warmup_seconds + self.config.balance_capture_seconds

    def reset(self) -> None:
        self.start_ts: float | None = None
        self.capture_frames = 0
        self.valid_frames = 0
        self.offsets: list[float] = []
        self.live_metrics: dict[str, float] = {}

    def start(self, ts: float) -> None:
        self.reset()
        self.start_ts = float(ts)

    def update(
        self,
        ts: float,
        keypoints: Sequence[Mapping[str, float]],
        pose: str,
    ) -> bool:
        if self.start_ts is None:
            self.start(ts)
        elapsed = max(0.0, float(ts) - float(self.start_ts))
        if elapsed < self.config.balance_warmup_seconds:
            return elapsed >= self.duration_seconds

        self.capture_frames += 1
        frame_metrics = _balance_frame_metrics(
            keypoints, self.config.min_keypoint_score
        )
        # The standing rule is only a safety/quality gate, not a diagnosis.
        if frame_metrics is not None and pose == "standing":
            self.valid_frames += 1
            self.live_metrics = frame_metrics
            self.offsets.append(float(frame_metrics["body_support_offset"]))
        return elapsed >= self.duration_seconds

    def finish(self) -> MotionResult:
        valid_fraction = self.valid_frames / max(self.capture_frames, 1)
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
        low = _percentile(self.offsets, 0.05)
        high = _percentile(self.offsets, 0.95)
        sway_range = high - low
        metrics = {
            "median_body_support_offset": center_offset,
            "sway_range": sway_range,
            "valid_samples": float(len(self.offsets)),
            "valid_fraction": valid_fraction,
        }

        if abs(center_offset) >= self.config.balance_offset_threshold:
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


class EyeMovementScreen:
    """Guided center/left/right target task using MediaPipe iris landmarks."""

    TARGETS = ("center", "left", "right")

    def __init__(self, config: BefastConfig) -> None:
        self.config = config
        self.reset()

    @property
    def duration_seconds(self) -> float:
        return self.config.eye_target_seconds * len(self.TARGETS)

    def reset(self) -> None:
        self.start_ts: float | None = None
        self.capture_frames = 0
        self.valid_frames = 0
        self.samples: dict[str, list[dict[str, float]]] = {
            target: [] for target in self.TARGETS
        }
        self.live_metrics: dict[str, float] = {}

    def start(self, ts: float) -> None:
        self.reset()
        self.start_ts = float(ts)

    def target(self, ts: float) -> str:
        if self.start_ts is None:
            return self.TARGETS[0]
        elapsed = max(0.0, float(ts) - self.start_ts)
        index = min(
            len(self.TARGETS) - 1,
            int(elapsed / self.config.eye_target_seconds),
        )
        return self.TARGETS[index]

    def update(self, ts: float, observation: FaceObservation | None) -> bool:
        if self.start_ts is None:
            self.start(ts)
        self.capture_frames += 1
        metrics = (
            _eye_frame_metrics(
                observation, self.config.face_min_interocular_width
            )
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
        valid_fraction = self.valid_frames / max(self.capture_frames, 1)
        counts = {target: len(values) for target, values in self.samples.items()}
        if (
            any(
                count < self.config.eye_min_samples_per_target
                for count in counts.values()
            )
            or valid_fraction < self.config.eye_min_valid_fraction
        ):
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
        left_range = abs(
            medians["right"]["left_gaze_x"]
            - medians["left"]["left_gaze_x"]
        )
        right_range = abs(
            medians["right"]["right_gaze_x"]
            - medians["left"]["right_gaze_x"]
        )
        conjugacy_errors: list[float] = []
        for target in ("left", "right"):
            left_delta = (
                medians[target]["left_gaze_x"]
                - medians["center"]["left_gaze_x"]
            )
            right_delta = (
                medians[target]["right_gaze_x"]
                - medians["center"]["right_gaze_x"]
            )
            conjugacy_errors.append(abs(left_delta - right_delta))
        conjugacy_error = max(conjugacy_errors)
        head_positions = [values["head_x"] for values in medians.values()]
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
            return MotionResult(
                status="positive",
                reason="reduced_visual_target_response",
                quality=valid_fraction,
                metrics=metrics,
            )
        if range_asymmetry >= self.config.eye_range_asymmetry_threshold:
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


class FaceSmileScreen:
    """Compare neutral and smile phases for lower-face movement asymmetry."""

    def __init__(self, config: BefastConfig) -> None:
        self.config = config
        self.reset()

    @property
    def duration_seconds(self) -> float:
        return self.config.face_neutral_seconds + self.config.face_smile_seconds

    def reset(self) -> None:
        self.start_ts: float | None = None
        self.capture_frames = 0
        self.valid_frames = 0
        self.neutral_samples: list[dict[str, float]] = []
        self.smile_samples: list[dict[str, float]] = []
        self.live_metrics: dict[str, float] = {}

    def start(self, ts: float) -> None:
        self.reset()
        self.start_ts = float(ts)

    def phase(self, ts: float) -> str:
        if self.start_ts is None:
            return "neutral"
        elapsed = max(0.0, float(ts) - self.start_ts)
        return "neutral" if elapsed < self.config.face_neutral_seconds else "smile"

    def update(self, ts: float, observation: FaceObservation | None) -> bool:
        if self.start_ts is None:
            self.start(ts)
        self.capture_frames += 1
        metrics = (
            _face_frame_metrics(
                observation, self.config.face_min_interocular_width
            )
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
        valid_fraction = self.valid_frames / max(self.capture_frames, 1)
        if (
            len(self.neutral_samples) < self.config.face_min_samples_per_phase
            or len(self.smile_samples) < self.config.face_min_samples_per_phase
            or valid_fraction < self.config.face_min_valid_fraction
        ):
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
        right_smile = median(
            sample["right_smile"] for sample in self.smile_samples
        )
        smile_strength = max(left_smile, right_smile)
        corner_delta = smile_corner - neutral_corner
        # Positive means the subject's left side moved less than the right side.
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


class BefastSession:
    """Thread-safe standby/active BE-FAST session shared by worker and Web threads."""

    MANUAL_KEYS = (
        "balance_problem",
        "speech_problem",
    )
    SKIP_STAGE_ALIASES = {
        "idle": "eyes",
        "eyes": "eyes",
        "retry_eyes": "eyes",
        "ready_face": "face",
        "face": "face",
        "retry_face": "face",
        "ready_arms": "arms",
        "arms": "arms",
        "retry_arms": "arms",
        "ready_balance": "balance",
        "balance": "balance",
        "retry_balance": "balance",
    }
    SKIP_FLOW = {
        "eyes": ("eye_result", "eye_screen", "ready_face"),
        "face": ("face_result", "face_screen", "ready_arms"),
        "arms": ("arm_result", "arm_screen", "ready_balance"),
        "balance": ("balance_result", "balance_screen", "review"),
    }

    def __init__(self, config: BefastConfig | None = None) -> None:
        self.config = config or BefastConfig()
        self.lock = threading.RLock()
        self.eye_screen = EyeMovementScreen(self.config)
        self.face_screen = FaceSmileScreen(self.config)
        self.arm_screen = ArmDriftScreen(self.config)
        self.balance_screen = BalanceScreen(self.config)
        self.reset()

    def reset(self) -> None:
        """Return the always-on device to low-load standby and clear one screen."""

        with self.lock:
            self.mode = "standby"
            self.screening_started_at: float | None = None
            self.trigger: dict[str, Any] | None = None
            self._clear_screen_locked()

    def _clear_screen_locked(self) -> None:
        """Clear active test state while preserving the caller-selected mode."""

        self.stage = "idle"
        self.stage_started_at: float | None = None
        self.eye_result = MotionResult()
        self.face_result = MotionResult()
        self.arm_result = MotionResult()
        self.balance_result = MotionResult()
        self.manual_complete = False
        self.manual: dict[str, bool | None] = {
            key: None for key in self.MANUAL_KEYS
        }
        self.new_or_sudden: bool | None = None
        self.onset_time: str | None = None
        self.guidance: dict[str, Any] = {
            "ready": False,
            "status": "waiting",
            "reason": "waiting_for_screening_trigger",
            "stage": "idle",
            "updated_at": None,
            "metrics": {},
        }
        self.retry_counts = {stage: 0 for stage in ("eyes", "face", "arms", "balance")}
        self.eye_screen.reset()
        self.face_screen.reset()
        self.arm_screen.reset()
        self.balance_screen.reset()

    def start_screening(
        self,
        source: str = "user",
        reason: str = "manual_request",
        now: float | None = None,
    ) -> None:
        """Open a new guided screen after a user, schedule, or passive trigger."""

        ts = time.time() if now is None else float(now)
        with self.lock:
            self._clear_screen_locked()
            self.mode = "screening"
            self.screening_started_at = ts
            self.trigger = {
                "source": str(source),
                "reason": str(reason),
                "triggered_at": round(ts, 4),
            }

    def start_stage(self, stage: str, now: float | None = None) -> None:
        if stage not in {"eyes", "face", "arms", "balance"}:
            raise ValueError("stage must be 'eyes', 'face', 'arms', or 'balance'")
        ts = time.time() if now is None else float(now)
        with self.lock:
            if self.mode != "screening":
                self._clear_screen_locked()
                self.mode = "screening"
                self.screening_started_at = ts
                self.trigger = {
                    "source": "stage_api",
                    "reason": "direct_stage_start",
                    "triggered_at": round(ts, 4),
                }
            self.stage = stage
            self.stage_started_at = ts
            self._set_guidance_locked(False, "collecting_action", ts)
            if stage == "eyes":
                self.eye_result = MotionResult(status="checking", reason="checking")
                self.eye_screen.start(ts)
            elif stage == "face":
                self.face_result = MotionResult(status="checking", reason="checking")
                self.face_screen.start(ts)
            elif stage == "arms":
                self.arm_result = MotionResult(status="checking", reason="checking")
                self.arm_screen.start(ts)
            else:
                self.balance_result = MotionResult(
                    status="checking", reason="checking"
                )
                self.balance_screen.start(ts)

    def skip_current_stage(self, now: float | None = None) -> str:
        """Skip the current automated check without treating it as normal."""

        ts = time.time() if now is None else float(now)
        with self.lock:
            if self.mode != "screening":
                raise ValueError("a screening must be active before a check can be skipped")
            check = self.SKIP_STAGE_ALIASES.get(self.stage)
            if check is None:
                raise ValueError("the current stage cannot be skipped")
            result_attr, screen_attr, next_stage = self.SKIP_FLOW[check]
            setattr(
                self,
                result_attr,
                MotionResult(status="skipped", reason="user_skipped"),
            )
            getattr(self, screen_attr).reset()
            self.stage = next_stage
            self.stage_started_at = None
            self._set_guidance_locked(False, f"prepare_{next_stage}", ts)
            return check

    def submit_manual(
        self,
        observations: Mapping[str, bool],
        new_or_sudden: bool,
        onset_time: str | None = None,
    ) -> None:
        missing = [key for key in self.MANUAL_KEYS if key not in observations]
        if missing:
            raise ValueError(f"missing manual observations: {', '.join(missing)}")
        with self.lock:
            if self.mode != "screening":
                self._clear_screen_locked()
                self.mode = "screening"
                self.screening_started_at = time.time()
                self.trigger = {
                    "source": "manual_form",
                    "reason": "manual_observation_submitted",
                    "triggered_at": round(self.screening_started_at, 4),
                }
            for key in self.MANUAL_KEYS:
                value = observations[key]
                if not isinstance(value, bool):
                    raise ValueError(f"{key} must be a boolean")
                self.manual[key] = value
            self.new_or_sudden = bool(new_or_sudden)
            self.onset_time = str(onset_time).strip() if onset_time else None
            self.manual_complete = True
            if self.stage in {"idle", "ready_balance", "review"}:
                self.stage = "review"

    def observe_face(
        self, ts: float, observation: FaceObservation | None
    ) -> None:
        """Update placement guidance while waiting to start E or F."""

        with self.lock:
            self._update_face_guidance_locked(ts, observation)

    def update_face(
        self, ts: float, observation: FaceObservation | None
    ) -> None:
        """Advance only the active face-based stage at its inference cadence."""

        with self.lock:
            self._update_face_guidance_locked(ts, observation)
            if self.stage == "eyes":
                if self.eye_screen.update(ts, observation):
                    result = self.eye_screen.finish()
                    self.eye_result = result
                    self._finish_or_retry_locked(
                        "eyes", result, "ready_face", self.eye_screen, ts
                    )
            elif self.stage == "face":
                if self.face_screen.update(ts, observation):
                    result = self.face_screen.finish()
                    self.face_result = result
                    self._finish_or_retry_locked(
                        "face", result, "ready_arms", self.face_screen, ts
                    )

    def observe_pose(
        self,
        ts: float,
        keypoints: Sequence[Mapping[str, float]],
        pose: str,
    ) -> None:
        """Update placement guidance while waiting to start A or B."""

        with self.lock:
            self._update_pose_guidance_locked(ts, keypoints, pose)

    def update(
        self,
        ts: float,
        keypoints: Sequence[Mapping[str, float]],
        pose: str,
    ) -> None:
        with self.lock:
            self._update_pose_guidance_locked(ts, keypoints, pose)
            if self.stage == "arms":
                if self.arm_screen.update(ts, keypoints):
                    result = self.arm_screen.finish()
                    self.arm_result = result
                    self._finish_or_retry_locked(
                        "arms", result, "ready_balance", self.arm_screen, ts
                    )
            elif self.stage == "balance":
                if self.balance_screen.update(ts, keypoints, pose):
                    result = self.balance_screen.finish()
                    self.balance_result = result
                    self._finish_or_retry_locked(
                        "balance", result, "review", self.balance_screen, ts
                    )

    def snapshot(self, now: float | None = None) -> dict[str, Any]:
        ts = time.time() if now is None else float(now)
        with self.lock:
            items = self._items_locked()
            if self.mode == "standby":
                decision, reasons = "standby", ["waiting_for_screening_trigger"]
            else:
                decision, reasons = self._decision_locked(items)
            duration = self._active_duration_locked()
            elapsed = (
                max(0.0, ts - self.stage_started_at)
                if self.stage_started_at is not None
                else 0.0
            )
            progress = min(1.0, elapsed / duration) if duration > 0 else 0.0
            snapshot = {
                "mode": self.mode,
                "stage": self.stage,
                "prompt": self._prompt_locked(),
                "progress": round(progress, 4),
                "decision": decision,
                "emergency": decision == "emergency",
                "reasons": reasons,
                "items": items,
                "new_or_sudden": self.new_or_sudden,
                "onset_time": self.onset_time,
                "manual_complete": self.manual_complete,
                "screening_started_at": self.screening_started_at,
                "trigger": dict(self.trigger) if self.trigger is not None else None,
                "guidance": {
                    **self.guidance,
                    "metrics": dict(self.guidance.get("metrics", {})),
                },
                "retry_counts": dict(self.retry_counts),
                "disclaimer": (
                    "Screening prototype only; it cannot diagnose or exclude stroke. "
                    "Any sudden BE-FAST sign requires emergency medical help."
                ),
            }
            if self.stage == "eyes":
                snapshot["eye_target"] = self.eye_screen.target(ts)
                target_elapsed = elapsed % self.config.eye_target_seconds
                snapshot["eye_target_remaining"] = round(
                    max(0.0, self.config.eye_target_seconds - target_elapsed),
                    1,
                )
            if self.stage == "face":
                snapshot["face_phase"] = self.face_screen.phase(ts)
            return snapshot

    def _finish_or_retry_locked(
        self,
        stage: str,
        result: MotionResult,
        next_stage: str,
        screen: Any,
        ts: float,
    ) -> None:
        self.stage_started_at = None
        if result.status == "insufficient":
            self.retry_counts[stage] += 1
            self.stage = f"retry_{stage}"
            screen.reset()
            self._set_guidance_locked(False, result.reason, ts)
            return
        self.stage = next_stage
        self._set_guidance_locked(False, f"prepare_{next_stage}", ts)

    def _set_guidance_locked(
        self,
        ready: bool,
        reason: str,
        ts: float,
        metrics: Mapping[str, float] | None = None,
    ) -> None:
        self.guidance = {
            "ready": bool(ready),
            "status": "correct" if ready else "adjust",
            "reason": str(reason),
            "stage": self.stage,
            "updated_at": round(float(ts), 4),
            "metrics": {
                str(key): round(float(value), 5)
                for key, value in (metrics or {}).items()
                if isinstance(value, (int, float))
            },
        }

    def _update_face_guidance_locked(
        self, ts: float, observation: FaceObservation | None
    ) -> None:
        if self.stage in {"idle", "retry_eyes", "eyes"}:
            metrics = (
                _eye_frame_metrics(observation, self.config.face_min_interocular_width)
                if observation is not None
                else None
            )
            self._set_guidance_locked(
                metrics is not None,
                "face_and_eyes_ready" if metrics is not None else "show_face_and_eyes",
                ts,
                metrics,
            )
            return
        if self.stage not in {"ready_face", "retry_face", "face"}:
            return
        metrics = (
            _face_frame_metrics(observation, self.config.face_min_interocular_width)
            if observation is not None
            else None
        )
        if metrics is None:
            self._set_guidance_locked(False, "show_full_face", ts)
            return
        smile_strength = max(metrics["left_smile"], metrics["right_smile"])
        phase = self.face_screen.phase(ts) if self.stage == "face" else "neutral"
        if phase == "smile":
            ready = smile_strength >= self.config.face_min_smile_score
            reason = "smile_detected" if ready else "smile_now"
        else:
            ready = smile_strength < self.config.face_min_smile_score
            reason = "neutral_face_ready" if ready else "relax_face_first"
        self._set_guidance_locked(ready, reason, ts, metrics)

    def _update_pose_guidance_locked(
        self,
        ts: float,
        keypoints: Sequence[Mapping[str, float]],
        pose: str,
    ) -> None:
        if self.stage in {"ready_arms", "retry_arms", "arms"}:
            metrics = _arm_frame_metrics(keypoints, self.config.min_keypoint_score)
            if metrics is None:
                self._set_guidance_locked(False, "show_shoulders_and_wrists", ts)
                return
            # One arm may genuinely be weak. Requiring both arms to be level here
            # would hide the very sign that A is intended to detect.
            one_arm_raised = min(
                metrics["left_wrist_relative_y"],
                metrics["right_wrist_relative_y"],
            ) <= self.config.arm_max_initial_wrist_below_shoulder
            self._set_guidance_locked(
                one_arm_raised,
                "arms_detected_hold_still" if one_arm_raised else "raise_both_arms",
                ts,
                metrics,
            )
            return
        if self.stage not in {"ready_balance", "retry_balance", "balance"}:
            return
        metrics = _balance_frame_metrics(keypoints, self.config.min_keypoint_score)
        ready = metrics is not None and pose == "standing"
        self._set_guidance_locked(
            ready,
            "standing_pose_ready" if ready else "show_full_body_and_stand_safely",
            ts,
            metrics,
        )

    def _items_locked(self) -> dict[str, dict[str, Any]]:
        balance_manual = self.manual["balance_problem"]
        if balance_manual is True:
            balance = _item("positive", "manual", "reported_balance_problem")
        elif self.balance_result.status == "positive":
            balance = _motion_item(self.balance_result, "pose")
        elif self.balance_result.status == "skipped":
            balance = _motion_item(self.balance_result, "pose")
        elif balance_manual is False and self.balance_result.status == "negative":
            balance = _motion_item(self.balance_result, "manual+pose")
        elif balance_manual is False and self.balance_result.status == "insufficient":
            balance = _motion_item(self.balance_result, "pose")
        else:
            balance = _item("pending", "manual+pose", "not_fully_checked")

        return {
            "B": balance,
            "E": _motion_item(self.eye_result, "mediapipe_face"),
            "F": _motion_item(self.face_result, "mediapipe_face"),
            "A": _motion_item(self.arm_result, "pose"),
            "S": self._manual_item_locked("speech_problem", "reported_speech_problem"),
        }

    def _manual_item_locked(self, key: str, positive_reason: str) -> dict[str, Any]:
        if not self.manual_complete:
            return _item("pending", "manual", "not_checked")
        return _item(
            "positive" if self.manual[key] else "negative",
            "manual",
            positive_reason if self.manual[key] else "not_reported",
        )

    def _decision_locked(
        self, items: Mapping[str, Mapping[str, Any]]
    ) -> tuple[str, list[str]]:
        positives = [code for code, item in items.items() if item["status"] == "positive"]
        if positives:
            reasons = [f"{code}:{items[code]['reason']}" for code in positives]
            if self.new_or_sudden is True:
                return "emergency", reasons
            return "warning", reasons

        statuses = [str(item["status"]) for item in items.values()]
        if statuses and all(status == "negative" for status in statuses):
            return "clear", ["no_obvious_befast_sign_detected"]
        if "insufficient" in statuses:
            return "incomplete", ["motion_check_quality_insufficient"]
        if "skipped" in statuses:
            return "incomplete", ["one_or_more_checks_skipped"]
        return "incomplete", ["complete_all_befast_checks"]

    def _active_duration_locked(self) -> float:
        if self.stage == "eyes":
            return self.eye_screen.duration_seconds
        if self.stage == "face":
            return self.face_screen.duration_seconds
        if self.stage == "arms":
            return self.arm_screen.duration_seconds
        if self.stage == "balance":
            return self.balance_screen.duration_seconds
        return 0.0

    def _prompt_locked(self) -> str:
        prompts = {
            "idle": (
                "The Raspberry Pi is in low-load standby. Start a screen when needed."
                if self.mode == "standby"
                else "Center your face, then start the guided eye check."
            ),
            "eyes": "Keep your head still and follow the moving target using only your eyes.",
            "retry_eyes": "Center your face and eyes so the eye check can restart.",
            "ready_face": "Eye movement check finished. Prepare for the smile check.",
            "face": "Keep a neutral face, then smile when prompted.",
            "retry_face": "Show your full face and relax before restarting the smile check.",
            "ready_arms": "Face check finished. Sit down and prepare to raise both arms.",
            "arms": "Keep both arms raised forward and level.",
            "retry_arms": "Show both shoulders and wrists, then raise both arms again.",
            "ready_balance": "Arm check finished. Only start balance check if safe.",
            "balance": "Stand still with support nearby; stop if unsafe.",
            "retry_balance": "Show your full body and stand only with support nearby.",
            "review": "Review speech, sudden onset, and all automated results.",
        }
        return prompts.get(self.stage, "Review the screening result.")


def _item(status: str, source: str, reason: str) -> dict[str, Any]:
    return {
        "status": status,
        "source": source,
        "reason": reason,
        "affected_side": None,
        "quality": None,
        "metrics": {},
    }


def _motion_item(result: MotionResult, source: str) -> dict[str, Any]:
    if result.status in {"not_run", "checking"}:
        status = "pending" if result.status == "not_run" else "checking"
    else:
        status = result.status
    item = result.as_dict()
    item["status"] = status
    item["source"] = source
    return item


def _aligned_face_points(
    observation: FaceObservation,
    indices: Sequence[int],
    min_interocular_width: float,
) -> tuple[dict[int, Point], float] | None:
    """Roll-align requested landmarks and return interocular normalization scale."""

    outer_right = observation.point(33)
    outer_left = observation.point(263)
    if outer_right is None or outer_left is None:
        return None
    dx = outer_left[0] - outer_right[0]
    dy = outer_left[1] - outer_right[1]
    scale = hypot(dx, dy)
    if scale < min_interocular_width:
        return None

    center_x = (outer_left[0] + outer_right[0]) / 2.0
    center_y = (outer_left[1] + outer_right[1]) / 2.0
    angle = atan2(dy, dx)
    cosine = cos(angle)
    sine = sin(angle)
    aligned: dict[int, Point] = {}
    for index in indices:
        point = observation.point(index)
        if point is None:
            return None
        relative_x = point[0] - center_x
        relative_y = point[1] - center_y
        aligned[index] = (
            cosine * relative_x + sine * relative_y,
            -sine * relative_x + cosine * relative_y,
        )
    return aligned, scale


def _eye_frame_metrics(
    observation: FaceObservation,
    min_interocular_width: float,
) -> dict[str, float] | None:
    """Return iris positions within each eye after in-plane roll correction."""

    # 468/473 are the subject's right/left iris centers. The remaining indices
    # are eye corners and the nose tip from the 478-landmark MediaPipe topology.
    result = _aligned_face_points(
        observation,
        (1, 33, 133, 263, 362, 468, 473),
        min_interocular_width,
    )
    if result is None:
        return None
    points, scale = result

    def iris_ratio(iris_index: int, corner_a: int, corner_b: int) -> float | None:
        low = min(points[corner_a][0], points[corner_b][0])
        high = max(points[corner_a][0], points[corner_b][0])
        width = high - low
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


def _face_frame_metrics(
    observation: FaceObservation,
    min_interocular_width: float,
) -> dict[str, float] | None:
    """Return roll-corrected mouth-corner symmetry and smile activations."""

    # 291 is the subject's left mouth corner; 61 is the right mouth corner.
    result = _aligned_face_points(
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


def _arm_frame_metrics(
    keypoints: Sequence[Mapping[str, float]], min_score: float
) -> dict[str, float] | None:
    points = keypoint_map(keypoints)
    left_shoulder = _point(points, "left_shoulder", min_score)
    right_shoulder = _point(points, "right_shoulder", min_score)
    left_wrist = _point(points, "left_wrist", min_score)
    right_wrist = _point(points, "right_wrist", min_score)
    if not all((left_shoulder, right_shoulder, left_wrist, right_wrist)):
        return None
    assert left_shoulder and right_shoulder and left_wrist and right_wrist
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
        "shoulder_width": shoulder_width,
    }


def _balance_frame_metrics(
    keypoints: Sequence[Mapping[str, float]], min_score: float
) -> dict[str, float] | None:
    points = keypoint_map(keypoints)
    required_names = (
        "left_shoulder",
        "right_shoulder",
        "left_hip",
        "right_hip",
        "left_ankle",
        "right_ankle",
    )
    required = [_point(points, name, min_score) for name in required_names]
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
    shoulder_center_x = (left_shoulder[0] + right_shoulder[0]) / 2.0
    hip_center_x = (left_hip[0] + right_hip[0]) / 2.0
    support_center_x = (left_ankle[0] + right_ankle[0]) / 2.0
    body_center_x = (shoulder_center_x + hip_center_x) / 2.0
    return {
        "body_support_offset": (body_center_x - support_center_x) / shoulder_width,
        "torso_lateral_offset": (shoulder_center_x - hip_center_x) / shoulder_width,
        "shoulder_width": shoulder_width,
    }


def _point(
    points: Mapping[str, Mapping[str, float]], name: str, min_score: float
) -> Point | None:
    point = points.get(name)
    if point is None or float(point.get("score", 0.0)) < min_score:
        return None
    return float(point["x"]), float(point["y"])


def _percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    index = int(round((len(ordered) - 1) * max(0.0, min(1.0, fraction))))
    return ordered[index]
