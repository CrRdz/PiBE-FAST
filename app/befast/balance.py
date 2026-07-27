"""B（Balance）：个人基线上的持续侧倾与横向摆动变化监测。

本模块有意使用 ``trunk kinematic proxy``（躯干运动学代理）命名。单目
MoveNet 关键点既不是力台压力中心（COP），也不是真实全身质心（TBCM）。
特征选择来自卒中 lateropulsion 与静态姿势描记研究；自动判定只表示相对
个人基线的持续变化，不能诊断或排除卒中。
"""

from __future__ import annotations

import json
from math import atan2, degrees, hypot, sqrt
from pathlib import Path
from statistics import median
import threading
from typing import Any, Mapping, Sequence

from app.keypoints import keypoint_map

from .config import BefastConfig
from .pose_geometry import percentile, visible_point
from .result import MotionResult, motion_report_item, report_item


BALANCE_EVIDENCE_VERSION = "stroke-balance-evidence-v1"
BALANCE_BASELINE_FEATURES = (
    "median_trunk_roll_degrees",
    "ml_sway_mean_velocity",
)


def balance_frame_metrics(
    keypoints: Sequence[Mapping[str, float]], min_score: float
) -> dict[str, float] | None:
    """计算单帧躯干侧倾及躯干相对双踝中点的横向位置。"""

    points = keypoint_map(keypoints)
    # 肩、胯描述躯干轴，脚踝中点仅作为画面内的支撑参照；它不是 COP。
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
    if shoulder_width <= 0.0:
        return None

    shoulder_center_x = (left_shoulder[0] + right_shoulder[0]) / 2.0
    shoulder_center_y = (left_shoulder[1] + right_shoulder[1]) / 2.0
    hip_center_x = (left_hip[0] + right_hip[0]) / 2.0
    hip_center_y = (left_hip[1] + right_hip[1]) / 2.0
    trunk_height = hip_center_y - shoulder_center_y
    if trunk_height <= 0.0:
        return None
    ankle_midpoint_x = (left_ankle[0] + right_ankle[0]) / 2.0
    trunk_center_x = (shoulder_center_x + hip_center_x) / 2.0
    # 正值表示肩中心相对髋中心向画面右侧倾斜，负值表示向左。
    trunk_roll_degrees = degrees(
        atan2(
            shoulder_center_x - hip_center_x,
            trunk_height,
        )
    )
    return {
        "trunk_support_offset": (
            trunk_center_x - ankle_midpoint_x
        )
        / shoulder_width,
        "trunk_roll_degrees": trunk_roll_degrees,
        "torso_lateral_offset": (shoulder_center_x - hip_center_x) / shoulder_width,
        "shoulder_width": shoulder_width,
    }


def summarize_balance_window(
    timestamps: Sequence[float],
    frame_metrics: Sequence[Mapping[str, float]],
) -> dict[str, float]:
    """把一段安静站立转换成文献常用的幅度、路径和速度类运动学代理。"""

    if len(timestamps) != len(frame_metrics) or not frame_metrics:
        raise ValueError("timestamps and frame_metrics must be non-empty and aligned")

    offsets = [float(value["trunk_support_offset"]) for value in frame_metrics]
    rolls = [float(value["trunk_roll_degrees"]) for value in frame_metrics]
    offset_center = median(offsets)
    roll_center = median(rolls)
    offset_residuals = [value - offset_center for value in offsets]
    roll_residuals = [value - roll_center for value in rolls]
    path_length = sum(
        abs(current - previous)
        for previous, current in zip(offsets, offsets[1:])
    )
    duration = max(float(timestamps[-1]) - float(timestamps[0]), 1e-9)

    return {
        "median_trunk_support_offset": offset_center,
        "median_trunk_roll_degrees": roll_center,
        "ml_sway_rms": sqrt(
            sum(value * value for value in offset_residuals)
            / len(offset_residuals)
        ),
        "ml_sway_p95_range": (
            percentile(offsets, 0.95) - percentile(offsets, 0.05)
        ),
        "ml_sway_path_length": path_length,
        "ml_sway_mean_velocity": path_length / duration,
        "trunk_roll_rms_degrees": sqrt(
            sum(value * value for value in roll_residuals)
            / len(roll_residuals)
        ),
        "window_duration_seconds": duration,
        "valid_samples": float(len(frame_metrics)),
    }


class PersonalBalanceBaseline:
    """保存个人重复站立窗口，并用 median/MAD 监测后续持续变化。"""

    def __init__(
        self,
        config: BefastConfig | None = None,
        path: str | Path | None = None,
    ) -> None:
        self.config = config or BefastConfig()
        self.path = Path(path) if path is not None else None
        self.lock = threading.RLock()
        self.samples: list[dict[str, float]] = []
        self._load()

    @property
    def ready(self) -> bool:
        with self.lock:
            return len(self.samples) >= max(
                1, int(self.config.balance_baseline_windows)
            )

    def reset(self) -> None:
        with self.lock:
            self.samples.clear()
            if self.path is not None:
                self.path.unlink(missing_ok=True)

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            return {
                "ready": len(self.samples)
                >= max(1, int(self.config.balance_baseline_windows)),
                "windows": len(self.samples),
                "target_windows": max(
                    1, int(self.config.balance_baseline_windows)
                ),
                "profile": self._profile_locked(),
                "comparison": "personal_median_mad",
                "evidence_version": BALANCE_EVIDENCE_VERSION,
            }

    def assess(
        self,
        features: Mapping[str, float],
        *,
        learn_if_needed: bool = True,
    ) -> dict[str, Any]:
        """校准个人基线，或返回当前窗口相对基线的稳健变化分数。"""

        sample = {
            name: float(features[name])
            for name in BALANCE_BASELINE_FEATURES
            if name in features
        }
        if len(sample) != len(BALANCE_BASELINE_FEATURES):
            raise ValueError("balance summary is missing baseline features")

        with self.lock:
            target = max(1, int(self.config.balance_baseline_windows))
            if len(self.samples) < target:
                if learn_if_needed:
                    self.samples.append(sample)
                    self._save_locked()
                return {
                    "status": "calibrating",
                    "changed_domains": [],
                    "unscorable_domains": [],
                    "scores": {},
                    "baseline_windows": len(self.samples),
                    "baseline_target": target,
                }

            profile = self._profile_locked()
            roll_values = profile["median_trunk_roll_degrees"]
            velocity_values = profile["ml_sway_mean_velocity"]
            roll_delta = abs(
                sample["median_trunk_roll_degrees"] - roll_values["median"]
            )
            # Stroke balance literature associates instability with increased sway
            # speed; a decrease is therefore not treated as an abnormal direction.
            velocity_delta = max(
                0.0,
                sample["ml_sway_mean_velocity"] - velocity_values["median"],
            )
            roll_score = (
                0.0
                if roll_delta == 0.0
                else (
                    roll_delta / roll_values["scale"]
                    if roll_values["scale"] > 0.0
                    else None
                )
            )
            velocity_score = (
                0.0
                if velocity_delta == 0.0
                else (
                    velocity_delta / velocity_values["scale"]
                    if velocity_values["scale"] > 0.0
                    else None
                )
            )
            threshold = float(self.config.balance_robust_z_threshold)
            changed_domains = []
            unscorable_domains = []
            if roll_score is None:
                unscorable_domains.append("trunk_orientation")
            elif roll_score >= threshold:
                changed_domains.append("trunk_orientation")
            if velocity_score is None:
                unscorable_domains.append("mediolateral_sway")
            elif velocity_score >= threshold:
                changed_domains.append("mediolateral_sway")
            status = "changed" if changed_domains else "stable"
            if unscorable_domains and not changed_domains:
                status = "unscorable"
            return {
                "status": status,
                "changed_domains": changed_domains,
                "unscorable_domains": unscorable_domains,
                "scores": {
                    "trunk_orientation": roll_score,
                    "mediolateral_sway": velocity_score,
                },
                "baseline_windows": len(self.samples),
                "baseline_target": target,
            }

    def _profile_locked(self) -> dict[str, dict[str, float]]:
        if not self.samples:
            return {}
        profile: dict[str, dict[str, float]] = {}
        for name in BALANCE_BASELINE_FEATURES:
            values = [
                float(sample[name])
                for sample in self.samples
                if name in sample
            ]
            if not values:
                continue
            center = median(values)
            mad_scale = 1.4826 * median(
                [abs(value - center) for value in values]
            )
            iqr_scale = (
                percentile(values, 0.75) - percentile(values, 0.25)
            ) / 1.349
            profile[name] = {
                "median": center,
                # 不人为加入传感器噪声下限。若两种离散度都为零而后续值
                # 改变，modified Z-score 无定义，调用方应返回数据不足。
                "scale": max(mad_scale, iqr_scale),
                "sample_count": float(len(values)),
            }
        return profile

    def _load(self) -> None:
        if self.path is None or not self.path.exists():
            return
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            loaded = []
            for raw in payload.get("samples", []):
                sample = {
                    name: float(raw[name])
                    for name in BALANCE_BASELINE_FEATURES
                    if name in raw
                }
                if len(sample) == len(BALANCE_BASELINE_FEATURES):
                    loaded.append(sample)
            limit = max(1, int(self.config.balance_baseline_windows))
            self.samples = loaded[-limit:]
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            self.samples = []

    def _save_locked(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "evidence_version": BALANCE_EVIDENCE_VERSION,
                    "samples": self.samples,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        temporary.replace(self.path)


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
    """累计安静站立窗口，并与同一人的多窗口基线比较。"""

    def __init__(
        self,
        config: BefastConfig,
        baseline: PersonalBalanceBaseline | None = None,
    ) -> None:
        self.config = config
        self.baseline = baseline or PersonalBalanceBaseline(config)
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
        self.timestamps: list[float] = []
        self.samples: list[dict[str, float]] = []
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
            self.timestamps.append(float(ts))
            self.samples.append(frame_metrics)
        return elapsed >= self.duration_seconds

    def finish(self) -> MotionResult:
        """汇总站立窗口，并输出相对个人基线的持续变化结果。"""

        valid_fraction = self.valid_frames / max(self.capture_frames, 1)
        # 关键点可见且保持站立的帧数不足时，不输出“正常”。
        if (
            len(self.samples) < self.config.balance_min_valid_samples
            or valid_fraction < self.config.balance_min_valid_fraction
        ):
            return MotionResult(
                status="insufficient",
                reason="stable_standing_pose_not_visible_long_enough",
                quality=valid_fraction,
                metrics={
                    "valid_samples": float(len(self.samples)),
                    "valid_fraction": valid_fraction,
                },
                details={
                    "measurement": "camera_trunk_kinematic_proxy_not_cop",
                    "evidence_version": BALANCE_EVIDENCE_VERSION,
                },
            )

        metrics = summarize_balance_window(self.timestamps, self.samples)
        metrics = {
            **metrics,
            "valid_fraction": valid_fraction,
        }
        assessment = self.baseline.assess(metrics)
        metrics.update(
            {
                "baseline_windows": float(assessment["baseline_windows"]),
                "baseline_target": float(assessment["baseline_target"]),
            }
        )
        for domain, metric_name in (
            ("trunk_orientation", "trunk_orientation_change_score"),
            ("mediolateral_sway", "mediolateral_sway_change_score"),
        ):
            score = assessment["scores"].get(domain)
            if score is not None:
                metrics[metric_name] = float(score)
        details = {
            "measurement": "camera_trunk_kinematic_proxy_not_cop",
            "comparison": "personal_median_mad",
            "statistical_change_threshold": (
                self.config.balance_robust_z_threshold
            ),
            "threshold_role": "monitoring_only_not_clinical_cutoff",
            "evidence_version": BALANCE_EVIDENCE_VERSION,
        }
        if assessment["status"] == "calibrating":
            return MotionResult(
                status="insufficient",
                reason="personal_balance_baseline_calibrating",
                quality=valid_fraction,
                metrics=metrics,
                details=details,
            )
        if assessment["status"] == "unscorable":
            return MotionResult(
                status="insufficient",
                reason="personal_balance_baseline_variability_not_estimable",
                quality=valid_fraction,
                metrics=metrics,
                details={
                    **details,
                    "unscorable_domains": assessment["unscorable_domains"],
                },
            )
        if "trunk_orientation" in assessment["changed_domains"]:
            roll = metrics["median_trunk_roll_degrees"]
            return MotionResult(
                status="positive",
                reason="sustained_trunk_orientation_change",
                quality=valid_fraction,
                metrics=metrics,
                details={
                    **details,
                    # 只报告画面方向，不把身体倾斜方向误写成卒中患侧。
                    "screen_direction": "right" if roll > 0 else "left",
                },
            )
        if "mediolateral_sway" in assessment["changed_domains"]:
            return MotionResult(
                status="positive",
                reason="increased_mediolateral_sway_velocity",
                quality=valid_fraction,
                metrics=metrics,
                details=details,
            )
        return MotionResult(
            status="negative",
            reason="no_sustained_change_from_personal_balance_baseline",
            quality=valid_fraction,
            metrics=metrics,
            details=details,
        )
