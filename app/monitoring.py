"""Low-rate passive pose monitoring used only to trigger an active screen.

Passive observations are intentionally kept separate from BE-FAST results. A
detected fall or a sustained balance change relative to a personal baseline can
request attention and open the guided workflow, but neither is treated as
evidence that a stroke is present or absent.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import inf
from typing import Any, Mapping, Sequence

from app.befast.balance import (
    BALANCE_EVIDENCE_VERSION,
    BalanceScreen,
    PersonalBalanceBaseline,
    balance_frame_metrics,
)
from app.befast.config import BefastConfig
from app.befast.result import MotionResult
from app.fall_detector import FallDetection, FallDetector
from app.pose_classifier import PoseClassification


@dataclass(frozen=True)
class PassiveMonitoringConfig:
    """Raspberry Pi-oriented defaults for the always-on standby layer."""

    enabled: bool = True
    inference_fps: float = 2.0
    trigger_cooldown_seconds: float = 60.0


class PassiveMonitor:
    """低频运行 MoveNet，并监测摔倒及个人站立平衡的持续变化。"""

    def __init__(
        self,
        config: PassiveMonitoringConfig | None = None,
        *,
        balance_config: BefastConfig | None = None,
        balance_baseline: PersonalBalanceBaseline | None = None,
    ) -> None:
        self.config = config or PassiveMonitoringConfig()
        self.fall_detector = FallDetector()
        self.balance_config = balance_config or BefastConfig()
        self.balance_screen = BalanceScreen(
            self.balance_config,
            balance_baseline,
        )
        self.last_inference_ts = -inf
        self.last_trigger_ts = -inf
        self.last_trigger_reason: str | None = None
        self.last_detection = FallDetection(
            fall=False,
            state="disabled" if not self.config.enabled else "waiting",
            confidence=0.0,
            reason="disabled" if not self.config.enabled else "awaiting_pose_sample",
        )
        self.last_balance_result = MotionResult(
            status="not_run",
            reason="awaiting_continuous_standing_window",
            details={"evidence_version": BALANCE_EVIDENCE_VERSION},
        )
        self.inference_count = 0

    @property
    def interval_seconds(self) -> float:
        return 1.0 / max(float(self.config.inference_fps), 0.1)

    def should_infer(self, ts: float) -> bool:
        return bool(
            self.config.enabled
            and float(ts) - self.last_inference_ts >= self.interval_seconds
        )

    def update(
        self,
        ts: float,
        pose: PoseClassification,
        keypoints: Sequence[Mapping[str, float]] | None = None,
    ) -> bool:
        """Consume one throttled pose sample and report a new trigger edge."""

        timestamp = float(ts)
        self.last_inference_ts = timestamp
        self.inference_count += 1
        self.last_detection = self.fall_detector.update(
            timestamp,
            pose.pose,
            pose.metrics,
            pose.quality,
        )
        balance_changed = self._update_balance(timestamp, pose, keypoints)
        trigger_reason = None
        if self.last_detection.fall:
            trigger_reason = "suspected_fall_trigger"
        elif balance_changed:
            trigger_reason = "sustained_personal_balance_change"
        if trigger_reason is None:
            return False
        if timestamp - self.last_trigger_ts < self.config.trigger_cooldown_seconds:
            return False
        self.last_trigger_ts = timestamp
        self.last_trigger_reason = trigger_reason
        return True

    def reset_alarm(self) -> None:
        """Clear temporal fall state after the user returns to standby."""

        self.fall_detector.reset()
        self.last_detection = FallDetection(
            fall=False,
            state="waiting" if self.config.enabled else "disabled",
            confidence=0.0,
            reason="alarm_reset" if self.config.enabled else "disabled",
        )
        # 离开主动筛查后重新开始一个完整站立窗口，但保留长期个人基线。
        self.balance_screen.reset()

    def snapshot(self, mode: str, inference_mode: str) -> dict[str, Any]:
        baseline = self.balance_screen.baseline.snapshot()
        return {
            "enabled": self.config.enabled,
            "mode": str(mode),
            "inference_mode": str(inference_mode),
            "standby_pose_fps": round(float(self.config.inference_fps), 2),
            "state": self.last_detection.state,
            "reason": self.last_detection.reason,
            "fall_detected": self.last_detection.fall,
            "confidence": round(float(self.last_detection.confidence), 4),
            "last_inference_at": (
                None if self.last_inference_ts == -inf else round(self.last_inference_ts, 4)
            ),
            "last_trigger_at": (
                None if self.last_trigger_ts == -inf else round(self.last_trigger_ts, 4)
            ),
            "last_trigger_reason": self.last_trigger_reason,
            "inference_count": self.inference_count,
            "medical_role": "trigger_only_not_stroke_diagnosis",
            "balance_change": {
                "window_active": self.balance_screen.start_ts is not None,
                "window_seconds": self.balance_config.balance_capture_seconds,
                "valid_samples": self.balance_screen.valid_frames,
                "capture_samples": self.balance_screen.capture_frames,
                "baseline_ready": baseline["ready"],
                "baseline_windows": baseline["windows"],
                "baseline_target": baseline["target_windows"],
                "comparison": baseline["comparison"],
                "latest_result": self.last_balance_result.as_dict(),
                "evidence_version": BALANCE_EVIDENCE_VERSION,
                "measurement": "camera_trunk_kinematic_proxy_not_cop",
            },
        }

    def _update_balance(
        self,
        ts: float,
        pose: PoseClassification,
        keypoints: Sequence[Mapping[str, float]] | None,
    ) -> bool:
        """Opportunistically collect a complete quiet-standing window."""

        if keypoints is None:
            return False
        frame_metrics = balance_frame_metrics(
            keypoints,
            self.balance_config.min_keypoint_score,
        )
        if self.balance_screen.start_ts is None:
            if pose.pose != "standing" or frame_metrics is None:
                return False
            self.balance_screen.start(ts)

        if not self.balance_screen.update(ts, keypoints, pose.pose):
            return False
        self.last_balance_result = self.balance_screen.finish()
        self.balance_screen.reset()
        return self.last_balance_result.status == "positive"
