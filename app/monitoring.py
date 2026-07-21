"""Low-rate passive monitoring used only to trigger an active screen.

Passive observations are intentionally kept separate from BE-FAST results. A
detected fall can request attention and open the guided workflow, but it is not
treated as evidence that a stroke is present or absent.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import inf
from typing import Any

from app.fall_detector import FallDetection, FallDetector
from app.pose_classifier import PoseClassification


@dataclass(frozen=True)
class PassiveMonitoringConfig:
    """Raspberry Pi-oriented defaults for the always-on standby layer."""

    enabled: bool = True
    inference_fps: float = 2.0
    trigger_cooldown_seconds: float = 60.0


class PassiveMonitor:
    """Throttle MoveNet and turn a confirmed fall into a screening trigger."""

    def __init__(self, config: PassiveMonitoringConfig | None = None) -> None:
        self.config = config or PassiveMonitoringConfig()
        self.fall_detector = FallDetector()
        self.last_inference_ts = -inf
        self.last_trigger_ts = -inf
        self.last_detection = FallDetection(
            fall=False,
            state="disabled" if not self.config.enabled else "waiting",
            confidence=0.0,
            reason="disabled" if not self.config.enabled else "awaiting_pose_sample",
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

    def update(self, ts: float, pose: PoseClassification) -> bool:
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
        if not self.last_detection.fall:
            return False
        if timestamp - self.last_trigger_ts < self.config.trigger_cooldown_seconds:
            return False
        self.last_trigger_ts = timestamp
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

    def snapshot(self, mode: str, inference_mode: str) -> dict[str, Any]:
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
            "inference_count": self.inference_count,
            "medical_role": "trigger_only_not_stroke_diagnosis",
        }
