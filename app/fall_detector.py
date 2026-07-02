"""Temporal fall detector built on top of pose classifications."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Mapping

from app.config import FallDetectorConfig


@dataclass(frozen=True)
class PoseFrame:
    ts: float
    pose: str
    center_y: float | None
    torso_angle: float | None
    quality: float


@dataclass(frozen=True)
class FallDetection:
    fall: bool
    state: str
    confidence: float
    reason: str


class FallDetector:
    """Detects fall sequences rather than treating lying as a fall."""

    def __init__(self, config: FallDetectorConfig | None = None) -> None:
        self.config = config or FallDetectorConfig()
        self.history: Deque[PoseFrame] = deque()
        self.candidate_since: float | None = None
        self.fall_active = False

    def update(
        self,
        ts: float,
        pose: str,
        metrics: Mapping[str, float] | None = None,
        quality: float = 1.0,
    ) -> FallDetection:
        metrics = metrics or {}
        frame = PoseFrame(
            ts=ts,
            pose=pose,
            center_y=_optional_float(metrics.get("center_y")),
            torso_angle=_optional_float(metrics.get("torso_vertical_degrees")),
            quality=quality,
        )
        self.history.append(frame)
        self._trim_history(ts)

        if self.fall_active:
            if self._contiguous_duration(ts, {"standing", "sitting"}) >= self.config.recovery_hold_seconds:
                self.fall_active = False
                self.candidate_since = None
                return FallDetection(False, "recovered", 0.0, "stable_upright")
            return FallDetection(True, "fall", 1.0, "fall_active")

        if self.candidate_since is not None:
            lying_duration = self._contiguous_duration(ts, {"lying"})
            if pose == "lying" and lying_duration >= self.config.lying_hold_seconds:
                self.fall_active = True
                return FallDetection(True, "fall", 1.0, "lying_after_fast_transition")
            if ts - self.candidate_since > self.config.max_candidate_seconds and pose != "lying":
                self.candidate_since = None
                return FallDetection(False, "normal", 0.0, "candidate_expired")
            return FallDetection(False, "falling", 0.6, "fast_transition_candidate")

        if self._is_fast_fall_transition(frame):
            self.candidate_since = ts
            return FallDetection(False, "falling", 0.6, "fast_drop_and_torso_rotation")

        return FallDetection(False, "normal", 0.0, "no_fall_sequence")

    def reset(self) -> None:
        self.history.clear()
        self.candidate_since = None
        self.fall_active = False

    def _trim_history(self, now: float) -> None:
        cutoff = now - self.config.history_seconds
        while self.history and self.history[0].ts < cutoff:
            self.history.popleft()

    def _is_fast_fall_transition(self, current: PoseFrame) -> bool:
        cfg = self.config
        if current.center_y is None or current.torso_angle is None:
            return False
        if current.torso_angle < cfg.horizontal_torso_min_degrees:
            return False

        cutoff = current.ts - cfg.transition_seconds
        for previous in reversed(self.history):
            if previous.ts < cutoff:
                break
            if previous is current:
                continue
            if previous.pose not in {"standing", "sitting"}:
                continue
            if previous.center_y is None or previous.torso_angle is None:
                continue
            center_drop = current.center_y - previous.center_y
            torso_delta = current.torso_angle - previous.torso_angle
            if (
                previous.torso_angle <= cfg.vertical_torso_max_degrees
                and center_drop >= cfg.center_drop_threshold
                and torso_delta >= cfg.torso_angle_delta_threshold
            ):
                return True
        return False

    def _contiguous_duration(self, now: float, poses: set[str]) -> float:
        start = None
        for frame in reversed(self.history):
            if frame.pose not in poses:
                break
            start = frame.ts
        if start is None:
            return 0.0
        return max(0.0, now - start)


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None

