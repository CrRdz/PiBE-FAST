"""Temporal fall detector built on top of pose classifications."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Mapping

from app.config import FallDetectorConfig


# 摔倒状态机保存的单帧摘要，只保留判断 fall 所需的时间、姿态和几何指标。
@dataclass(frozen=True)
class PoseFrame:
    ts: float
    pose: str
    center_y: float | None
    torso_angle: float | None
    quality: float


# 摔倒检测输出结果，描述当前是否 fall、状态阶段、置信度和触发原因。
@dataclass(frozen=True)
class FallDetection:
    fall: bool
    state: str
    confidence: float
    reason: str


# 连续帧摔倒状态机，用“快速下落 + 躯干旋转 + 持续躺下”判断疑似摔倒。
class FallDetector:
    """Detects fall sequences rather than treating lying as a fall."""

    def __init__(self, config: FallDetectorConfig | None = None) -> None:
        self.config = config or FallDetectorConfig()
        # history 只保存轻量摘要，不保存图像，避免常驻内存越来越大。
        self.history: Deque[PoseFrame] = deque()
        # candidate_since 不为 None 表示已经看到“快速下落/旋转”，正在等待持续 lying 确认。
        self.candidate_since: float | None = None
        # fall_active 表示已经确认 fall，会保持到人重新稳定站起/坐起。
        self.fall_active = False

    def update(
        self,
        ts: float,
        pose: str,
        metrics: Mapping[str, float] | None = None,
        quality: float = 1.0,
    ) -> FallDetection:
        metrics = metrics or {}
        # 每帧只从分类器指标里取 fall 需要的两个连续量：人体中心 y 和躯干角度。
        frame = PoseFrame(
            ts=ts,
            pose=pose,
            center_y=_optional_float(metrics.get("center_y")),
            torso_angle=_optional_float(metrics.get("torso_vertical_degrees")),
            quality=quality,
        )
        self.history.append(frame)
        # 保留最近几秒即可，旧帧对“当前是否摔倒”没有价值。
        self._trim_history(ts)

        # 已经确认 fall 后，不会因为某一帧误判而立刻取消；需要持续站/坐一段时间。
        if self.fall_active:
            if self._contiguous_duration(ts, {"standing", "sitting"}) >= self.config.recovery_hold_seconds:
                self.fall_active = False
                self.candidate_since = None
                return FallDetection(False, "recovered", 0.0, "stable_upright")
            return FallDetection(True, "fall", 1.0, "fall_active")

        # falling 候选阶段：已经看到快速变化，但还要等 lying 持续足够久才确认。
        if self.candidate_since is not None:
            lying_duration = self._contiguous_duration(ts, {"lying"})
            if pose == "lying" and lying_duration >= self.config.lying_hold_seconds:
                self.fall_active = True
                return FallDetection(True, "fall", 1.0, "lying_after_fast_transition")
            # 如果候选阶段过久且没有进入 lying，说明可能只是弯腰、坐下或检测抖动。
            if ts - self.candidate_since > self.config.max_candidate_seconds and pose != "lying":
                self.candidate_since = None
                return FallDetection(False, "normal", 0.0, "candidate_expired")
            return FallDetection(False, "falling", 0.6, "fast_transition_candidate")

        # 正常阶段：检查当前帧是否像“从站/坐快速倒向水平”的转折点。
        if self._is_fast_fall_transition(frame):
            self.candidate_since = ts
            return FallDetection(False, "falling", 0.6, "fast_drop_and_torso_rotation")

        return FallDetection(False, "normal", 0.0, "no_fall_sequence")

    def reset(self) -> None:
        # 测试或重新开始检测时使用，清空所有历史状态。
        self.history.clear()
        self.candidate_since = None
        self.fall_active = False

    def _trim_history(self, now: float) -> None:
        # 删除超出 history_seconds 的旧帧，保证判断只基于最近一小段时间。
        cutoff = now - self.config.history_seconds
        while self.history and self.history[0].ts < cutoff:
            self.history.popleft()

    def _is_fast_fall_transition(self, current: PoseFrame) -> bool:
        cfg = self.config
        # 没有人体中心或躯干角度时，无法判断“快速下降 + 旋转”。
        if current.center_y is None or current.torso_angle is None:
            return False
        # 当前帧必须已经接近水平，否则还不能说像摔倒落地。
        if current.torso_angle < cfg.horizontal_torso_min_degrees:
            return False

        cutoff = current.ts - cfg.transition_seconds
        # 从近到远找 transition_seconds 内的站/坐姿态，检查是否发生了快速变化。
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
            # 三个条件同时满足才认为是“疑似摔倒转折”：
            # 1. 之前躯干接近竖直；2. 中心点明显向下；3. 躯干快速转向水平。
            if (
                previous.torso_angle <= cfg.vertical_torso_max_degrees
                and center_drop >= cfg.center_drop_threshold
                and torso_delta >= cfg.torso_angle_delta_threshold
            ):
                return True
        return False

    def _contiguous_duration(self, now: float, poses: set[str]) -> float:
        # 计算“从当前往回看，某些姿态连续持续了多久”。
        # 例如连续 lying 超过 2 秒，才把 falling 确认为 fall。
        start = None
        for frame in reversed(self.history):
            if frame.pose not in poses:
                break
            start = frame.ts
        if start is None:
            return 0.0
        return max(0.0, now - start)


def _optional_float(value: object) -> float | None:
    # metrics 来自分类器，做一次安全转换，避免 None/字符串导致状态机崩溃。
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
