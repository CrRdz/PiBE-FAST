"""E（Eyes）：质量门控的重复水平注视终点检查。"""

from __future__ import annotations

from math import asin, atan2, degrees, hypot, sqrt
from statistics import median

from app.face_landmarker import FaceObservation

from .config import BefastConfig
from .face_geometry import aligned_face_points
from .result import MotionResult


EYE_EVIDENCE_VERSION = "stroke-eye-phenotype-v3"
_EPSILON = 1e-9


def _median_absolute_deviation(values: list[float]) -> float:
    """返回未缩放 MAD；空输入只在内部防御性路径中出现。"""

    if not values:
        return 0.0
    center = median(values)
    return median(abs(value - center) for value in values)


def _head_pose_degrees(
    transform: tuple[tuple[float, ...], ...] | None,
) -> tuple[float, float, float] | None:
    """从 MediaPipe canonical-face 变换矩阵提取 pitch/yaw/roll。

    这里只使用同一轮内的角度变化来排除转头代偿，不把角度当作诊断特征。
    """

    if transform is None or len(transform) < 3 or any(
        len(row) < 3 for row in transform[:3]
    ):
        return None
    matrix = [[float(transform[row][column]) for column in range(3)] for row in range(3)]
    # canonical-face 矩阵可能带统一尺度；用三个列向量平均长度去掉尺度。
    column_norms = [
        sqrt(sum(matrix[row][column] ** 2 for row in range(3)))
        for column in range(3)
    ]
    scale = sum(column_norms) / 3.0
    if scale <= _EPSILON:
        return None
    rotation = [[value / scale for value in row] for row in matrix]

    # Z-Y-X Euler 分解。具体正负方向不参与判定，只比较一轮内的范围。
    pitch = atan2(rotation[2][1], rotation[2][2])
    yaw = asin(max(-1.0, min(1.0, -rotation[2][0])))
    roll = atan2(rotation[1][0], rotation[0][0])
    return degrees(pitch), degrees(yaw), degrees(roll)


def eye_frame_metrics(
    observation: FaceObservation,
    min_interocular_width: float,
    min_eye_width_pixels: float = 0.0,
    enforce_relative_eye_width: bool = True,
) -> dict[str, float] | None:
    """返回虹膜终点、眼部像素尺度和独立三维头姿质量指标。"""

    # 468/473 是受试者右/左虹膜中心，其余索引为眼角和鼻尖。
    result = aligned_face_points(
        observation,
        (1, 33, 133, 263, 362, 468, 473),
        min_interocular_width,
    )
    if result is None:
        return None
    points, scale = result

    right_eye_width = abs(points[133][0] - points[33][0])
    left_eye_width = abs(points[263][0] - points[362][0])
    if right_eye_width <= 0.0 or left_eye_width <= 0.0:
        return None
    if enforce_relative_eye_width and (
        right_eye_width <= scale * 0.04 or left_eye_width <= scale * 0.04
    ):
        return None

    right_eye_width_pixels = (
        right_eye_width * observation.frame_width if observation.frame_width else 0.0
    )
    left_eye_width_pixels = (
        left_eye_width * observation.frame_width if observation.frame_width else 0.0
    )
    if (
        min_eye_width_pixels > 0.0
        and observation.frame_width > 0
        and min(right_eye_width_pixels, left_eye_width_pixels) < min_eye_width_pixels
    ):
        return None

    def iris_ratio(iris_index: int, corner_a: int, corner_b: int) -> float:
        low = min(points[corner_a][0], points[corner_b][0])
        high = max(points[corner_a][0], points[corner_b][0])
        return (points[iris_index][0] - low) / (high - low)

    pose = _head_pose_degrees(observation.facial_transform)
    metrics = {
        "left_gaze_x": iris_ratio(473, 362, 263),
        "right_gaze_x": iris_ratio(468, 33, 133),
        "left_eye_width_pixels": left_eye_width_pixels,
        "right_eye_width_pixels": right_eye_width_pixels,
        "interocular_width": scale,
        "inference_ms": observation.inference_ms,
        "head_pose_available": 1.0 if pose is not None else 0.0,
    }
    if pose is not None:
        metrics.update(
            {
                "head_pitch_degrees": pose[0],
                "head_yaw_degrees": pose[1],
                "head_roll_degrees": pose[2],
            }
        )
    return metrics


class EyeMovementScreen:
    """重复采集中心—侧方配对试次，只分析低帧率稳定注视终点。"""

    # 左右各重复三次，并交替反转方向顺序，降低练习和疲劳的顺序偏差。
    TRIALS = (
        "rest",
        "center",
        "left",
        "center",
        "right",
        "center",
        "right",
        "center",
        "left",
        "center",
        "left",
        "center",
        "right",
    )
    LATERAL_PAIRS = (
        (1, 2, "left"),
        (3, 4, "right"),
        (5, 6, "right"),
        (7, 8, "left"),
        (9, 10, "left"),
        (11, 12, "right"),
    )

    def __init__(self, config: BefastConfig) -> None:
        self.config = config
        self.setup: dict[str, float] = {}
        self.configure_setup()
        self.reset()

    @property
    def duration_seconds(self) -> float:
        return self.config.eye_target_seconds * len(self.TRIALS)

    def configure_setup(
        self,
        viewing_distance_cm: float | None = None,
        screen_width_cm: float | None = None,
        achieved_target_visual_angle_degrees: float | None = None,
    ) -> None:
        """记录前端用于构造固定视角刺激的物理设置。"""

        distance = (
            self.config.eye_default_viewing_distance_cm
            if viewing_distance_cm is None
            else float(viewing_distance_cm)
        )
        width = (
            self.config.eye_default_screen_width_cm
            if screen_width_cm is None
            else float(screen_width_cm)
        )
        if not 20.0 <= distance <= 150.0:
            raise ValueError("viewing_distance_cm must be between 20 and 150")
        if not 5.0 <= width <= 100.0:
            raise ValueError("screen_width_cm must be between 5 and 100")
        achieved_angle = (
            self.config.eye_target_visual_angle_degrees
            if achieved_target_visual_angle_degrees is None
            else float(achieved_target_visual_angle_degrees)
        )
        if not 1.0 <= achieved_angle <= 30.0:
            raise ValueError(
                "achieved_target_visual_angle_degrees must be between 1 and 30"
            )
        self.setup = {
            "target_visual_angle_degrees": float(
                self.config.eye_target_visual_angle_degrees
            ),
            "viewing_distance_cm": distance,
            "screen_width_cm": width,
            "achieved_target_visual_angle_degrees": achieved_angle,
        }

    def reset(self) -> None:
        self.start_ts: float | None = None
        self.capture_frames_by_trial = [0 for _ in self.TRIALS]
        self.valid_frames_by_trial = [0 for _ in self.TRIALS]
        self.samples: list[list[dict[str, float]]] = [[] for _ in self.TRIALS]
        self.live_metrics: dict[str, float] = {}

    def start(self, ts: float) -> None:
        self.reset()
        self.start_ts = float(ts)

    def trial_index(self, ts: float) -> int:
        if self.start_ts is None:
            return 0
        elapsed = max(0.0, float(ts) - self.start_ts)
        return min(
            len(self.TRIALS) - 1,
            int(elapsed / self.config.eye_target_seconds),
        )

    def target(self, ts: float) -> str:
        return self.TRIALS[self.trial_index(ts)]

    def is_settling(self, ts: float) -> bool:
        if self.start_ts is None:
            return True
        elapsed = max(0.0, float(ts) - self.start_ts)
        within_trial = elapsed % self.config.eye_target_seconds
        return within_trial < self.config.eye_settle_seconds

    def update(self, ts: float, observation: FaceObservation | None) -> bool:
        """忽略目标切换后的稳定段，只把随后帧放入当前重复试次。"""

        if self.start_ts is None:
            self.start(ts)
        elapsed = max(0.0, float(ts) - float(self.start_ts))
        if elapsed >= self.duration_seconds:
            return True
        if self.is_settling(ts):
            return False

        trial = self.trial_index(ts)
        self.capture_frames_by_trial[trial] += 1
        metrics = (
            eye_frame_metrics(
                observation,
                (
                    self.config.face_min_interocular_width
                    if self.config.eye_enable_unvalidated_quality_gates
                    else 0.0
                ),
                (
                    self.config.eye_min_eye_width_pixels
                    if self.config.eye_enable_unvalidated_quality_gates
                    else 0.0
                ),
                self.config.eye_enable_unvalidated_quality_gates,
            )
            if observation is not None
            else None
        )
        if metrics is not None:
            self.valid_frames_by_trial[trial] += 1
            self.live_metrics = metrics
            self.samples[trial].append(metrics)
        return False

    def finish(self) -> MotionResult:
        """应用逐试次质量门控，再评估可见响应、重复性和相对对称性。"""

        counts = [len(values) for values in self.samples]
        valid_fractions = [
            valid / max(captured, 1)
            for valid, captured in zip(
                self.valid_frames_by_trial, self.capture_frames_by_trial
            )
        ]
        quality = min(valid_fractions, default=0.0)
        quality_metrics = {
            "minimum_trial_valid_fraction": quality,
            "total_valid_samples": float(sum(counts)),
            **{
                f"trial_{index + 1}_samples": float(count)
                for index, count in enumerate(counts)
            },
        }
        minimum_samples = (
            self.config.eye_min_samples_per_trial
            if self.config.eye_enable_unvalidated_quality_gates
            else 1
        )
        if any(count < minimum_samples for count in counts) or (
            self.config.eye_enable_unvalidated_quality_gates
            and any(
                fraction < self.config.eye_min_valid_fraction_per_trial
                for fraction in valid_fractions
            )
        ):
            return MotionResult(
                status="insufficient",
                reason="eye_trials_not_visible_long_enough",
                quality=quality,
                metrics=quality_metrics,
            )

        summaries: list[dict[str, float]] = []
        for trial_index, values in enumerate(self.samples):
            left_values = [sample["left_gaze_x"] for sample in values]
            right_values = [sample["right_gaze_x"] for sample in values]
            pose_values = [
                sample for sample in values if sample["head_pose_available"] >= 0.5
            ]
            summary = {
                "left_gaze_x": median(left_values),
                "right_gaze_x": median(right_values),
                "left_gaze_mad": _median_absolute_deviation(left_values),
                "right_gaze_mad": _median_absolute_deviation(right_values),
                "left_eye_width_pixels": median(
                    sample["left_eye_width_pixels"] for sample in values
                ),
                "right_eye_width_pixels": median(
                    sample["right_eye_width_pixels"] for sample in values
                ),
                "head_pose_fraction": len(pose_values) / len(values),
            }
            if pose_values:
                for name in (
                    "head_pitch_degrees",
                    "head_yaw_degrees",
                    "head_roll_degrees",
                ):
                    summary[name] = median(sample[name] for sample in pose_values)
            summaries.append(summary)
            trial_number = trial_index + 1
            for eye in ("left", "right"):
                quality_metrics[
                    f"trial_{trial_number}_{eye}_iris_position"
                ] = summary[f"{eye}_gaze_x"]
                quality_metrics[
                    f"trial_{trial_number}_{eye}_iris_mad"
                ] = summary[f"{eye}_gaze_mad"]

        max_gaze_mad = max(
            max(summary["left_gaze_mad"], summary["right_gaze_mad"])
            for summary in summaries
        )
        quality_metrics["max_trial_gaze_mad"] = max_gaze_mad
        if (
            self.config.eye_enable_unvalidated_quality_gates
            and max_gaze_mad > self.config.eye_max_gaze_mad
        ):
            return MotionResult(
                status="insufficient",
                reason="unstable_eye_landmarks_during_test",
                quality=quality,
                metrics=quality_metrics,
            )

        minimum_pose_fraction = min(
            summary["head_pose_fraction"] for summary in summaries
        )
        quality_metrics["minimum_head_pose_fraction"] = minimum_pose_fraction
        if (
            self.config.eye_enable_unvalidated_quality_gates
            and minimum_pose_fraction < self.config.eye_min_head_pose_fraction
        ):
            return MotionResult(
                status="insufficient",
                reason="head_pose_not_available_during_eye_test",
                quality=quality,
                metrics=quality_metrics,
            )

        pose_names = (
            "head_pitch_degrees",
            "head_yaw_degrees",
            "head_roll_degrees",
        )
        complete_head_pose = all(
            all(name in summary for name in pose_names) for summary in summaries
        )
        quality_metrics["complete_head_pose"] = 1.0 if complete_head_pose else 0.0
        max_head_rotation: float | None = None
        if complete_head_pose:
            head_ranges: dict[str, float] = {}
            for name in pose_names:
                values = [summary[name] for summary in summaries]
                head_ranges[name] = max(values) - min(values)
            max_head_rotation = max(head_ranges.values())
            quality_metrics.update(
                {
                    "head_pitch_range_degrees": head_ranges["head_pitch_degrees"],
                    "head_yaw_range_degrees": head_ranges["head_yaw_degrees"],
                    "head_roll_range_degrees": head_ranges["head_roll_degrees"],
                    "max_head_rotation_degrees": max_head_rotation,
                }
            )
        if (
            self.config.eye_enable_unvalidated_quality_gates
            and max_head_rotation is not None
            and max_head_rotation > self.config.eye_max_head_rotation_degrees
        ):
            return MotionResult(
                status="insufficient",
                reason="head_moved_during_eye_test",
                quality=quality,
                metrics=quality_metrics,
            )

        raw_responses: dict[str, dict[str, list[float]]] = {
            eye: {"left": [], "right": []} for eye in ("left", "right")
        }
        response_noises: dict[str, dict[str, list[float]]] = {
            eye: {"left": [], "right": []} for eye in ("left", "right")
        }
        for center_index, target_index, direction in self.LATERAL_PAIRS:
            for eye in ("left", "right"):
                key = f"{eye}_gaze_x"
                raw_response = (
                    summaries[target_index][key] - summaries[center_index][key]
                )
                raw_responses[eye][direction].append(raw_response)
                if self.config.eye_enable_unvalidated_warning_thresholds:
                    center_sigma = max(
                        1.4826 * summaries[center_index][f"{eye}_gaze_mad"],
                        self.config.eye_landmark_noise_floor_pixels
                        / max(
                            summaries[center_index][f"{eye}_eye_width_pixels"],
                            _EPSILON,
                        ),
                    )
                    target_sigma = max(
                        1.4826 * summaries[target_index][f"{eye}_gaze_mad"],
                        self.config.eye_landmark_noise_floor_pixels
                        / max(
                            summaries[target_index][f"{eye}_eye_width_pixels"],
                            _EPSILON,
                        ),
                    )
                    response_noises[eye][direction].append(
                        hypot(center_sigma, target_sigma)
                    )

        # 镜像约定来自采集配置，不能从受试者响应反推；否则始终看反方向的
        # 任务错误也可能被解释成“镜像摄像头”并错误通过。
        coordinate_orientation = -1.0 if self.config.eye_camera_mirrored else 1.0
        responses: dict[str, dict[str, list[float]]] = {
            eye: {"left": [], "right": []} for eye in ("left", "right")
        }
        response_snrs: dict[str, dict[str, list[float]]] = {
            eye: {"left": [], "right": []} for eye in ("left", "right")
        }
        for eye in ("left", "right"):
            for direction in ("left", "right"):
                expected_sign = (
                    -coordinate_orientation
                    if direction == "left"
                    else coordinate_orientation
                )
                for index, raw_response in enumerate(raw_responses[eye][direction]):
                    response = expected_sign * raw_response
                    responses[eye][direction].append(response)
                    if self.config.eye_enable_unvalidated_warning_thresholds:
                        noise = response_noises[eye][direction][index]
                        response_snrs[eye][direction].append(response / noise)

        repeat_errors: dict[str, float] = {}
        for eye in ("left", "right"):
            for direction in ("left", "right"):
                ordered = sorted(responses[eye][direction])
                center = median(ordered)
                scale = max(abs(center), _EPSILON)
                # 三次中允许一个离群值，只要求中位数与距离最近的另一轮一致。
                repeat_errors[f"{eye}_{direction}"] = min(
                    abs(center - ordered[0]),
                    abs(ordered[-1] - center),
                ) / scale
                repeat_spread = max(
                    abs(response - center) for response in ordered
                ) / scale
                for repetition, response in enumerate(
                    responses[eye][direction], start=1
                ):
                    prefix = f"{eye}_{direction}_repeat_{repetition}"
                    quality_metrics[f"{prefix}_response"] = response
                    if self.config.eye_enable_unvalidated_warning_thresholds:
                        quality_metrics[f"{prefix}_noise"] = response_noises[eye][
                            direction
                        ][repetition - 1]
                        quality_metrics[f"{prefix}_snr"] = response_snrs[eye][
                            direction
                        ][repetition - 1]
                quality_metrics[
                    f"{eye}_{direction}_repeat_relative_error"
                ] = repeat_errors[f"{eye}_{direction}"]
                quality_metrics[
                    f"{eye}_{direction}_repeat_relative_spread"
                ] = repeat_spread
        max_repeat_error = max(repeat_errors.values())
        quality_metrics["max_repeat_relative_error"] = max_repeat_error
        quality_metrics["coordinate_orientation"] = coordinate_orientation
        if (
            self.config.eye_enable_unvalidated_quality_gates
            and max_repeat_error > self.config.eye_max_repeat_relative_error
        ):
            return MotionResult(
                status="insufficient",
                reason="eye_response_not_repeatable",
                quality=quality,
                metrics=quality_metrics,
            )

        persistent_failures: list[tuple[str, str]] = []
        if self.config.eye_enable_unvalidated_warning_thresholds:
            for eye in ("left", "right"):
                for direction in ("left", "right"):
                    passed = [
                        response > 0.0
                        and snr >= self.config.eye_response_snr_threshold
                        for response, snr in zip(
                            responses[eye][direction],
                            response_snrs[eye][direction],
                        )
                    ]
                    passed_count = sum(passed)
                    if passed_count == 0:
                        persistent_failures.append((eye, direction))
                    elif passed_count < 2:
                        return MotionResult(
                            status="insufficient",
                            reason="eye_response_not_repeatable",
                            quality=quality,
                            metrics={
                                **quality_metrics,
                                "minimum_response_snr": min(
                                    min(values)
                                    for directions in response_snrs.values()
                                    for values in directions.values()
                                ),
                            },
                        )

        minimum_response_snr = (
            min(
                min(values)
                for directions in response_snrs.values()
                for values in directions.values()
            )
            if self.config.eye_enable_unvalidated_warning_thresholds
            else None
        )
        response_medians = {
            eye: {
                direction: median(values)
                for direction, values in directions.items()
            }
            for eye, directions in responses.items()
        }
        response_amplitudes = {
            eye: {
                direction: max(0.0, value)
                for direction, value in directions.items()
            }
            for eye, directions in response_medians.items()
        }
        left_range = sum(response_amplitudes["left"].values())
        right_range = sum(response_amplitudes["right"].values())
        inter_eye_asymmetry = abs(left_range - right_range) / max(
            left_range, right_range, _EPSILON
        )
        directional_asymmetries = {
            eye: abs(values["left"] - values["right"])
            / max(values["left"], values["right"], _EPSILON)
            for eye, values in response_amplitudes.items()
        }
        binocular_directional_responses = {
            direction: (
                response_amplitudes["left"][direction]
                + response_amplitudes["right"][direction]
            )
            / 2.0
            for direction in ("left", "right")
        }
        binocular_directional_asymmetry = abs(
            binocular_directional_responses["left"]
            - binocular_directional_responses["right"]
        ) / max(max(binocular_directional_responses.values()), _EPSILON)
        # 按方向直接比较两眼的归一化虹膜响应。旧实现先把每只眼的两个方向
        # 除以该眼总响应，使左右方向误差在数学上必然相等，无法定位方向。
        conjugacy_errors = {
            direction: abs(
                response_amplitudes["left"][direction]
                - response_amplitudes["right"][direction]
            )
            / max(
                response_amplitudes["left"][direction],
                response_amplitudes["right"][direction],
                _EPSILON,
            )
            for direction in ("left", "right")
        }
        max_conjugacy_error = max(conjugacy_errors.values())
        lateral_endpoints = {
            eye: {
                direction: median(
                    summaries[target_index][f"{eye}_gaze_x"]
                    for _, target_index, pair_direction in self.LATERAL_PAIRS
                    if pair_direction == direction
                )
                for direction in ("left", "right")
            }
            for eye in ("left", "right")
        }
        rest_bias_degrees: dict[str, float] = {}
        achieved_angle = self.setup["achieved_target_visual_angle_degrees"]
        for eye in ("left", "right"):
            center_endpoint = median(
                summaries[center_index][f"{eye}_gaze_x"]
                for center_index, _, _ in self.LATERAL_PAIRS
            )
            excursion = abs(
                lateral_endpoints[eye]["right"]
                - lateral_endpoints[eye]["left"]
            )
            rest_bias_degrees[eye] = (
                coordinate_orientation
                * (summaries[0][f"{eye}_gaze_x"] - center_endpoint)
                / excursion
                * 2.0
                * achieved_angle
                if excursion > _EPSILON
                else 0.0
            )
        same_rest_direction = (
            rest_bias_degrees["left"] * rest_bias_degrees["right"] > 0.0
        )
        conjugate_rest_deviation = (
            min(
                abs(rest_bias_degrees["left"]),
                abs(rest_bias_degrees["right"]),
            )
            if same_rest_direction
            else 0.0
        )
        metrics = {
            **quality_metrics,
            **self.setup,
            "left_gaze_range": left_range,
            "right_gaze_range": right_range,
            "range_asymmetry": inter_eye_asymmetry,
            "inter_eye_range_asymmetry": inter_eye_asymmetry,
            "left_eye_directional_asymmetry": directional_asymmetries["left"],
            "right_eye_directional_asymmetry": directional_asymmetries["right"],
            "binocular_directional_asymmetry": binocular_directional_asymmetry,
            "max_directional_asymmetry": binocular_directional_asymmetry,
            "left_conjugacy_relative_error": conjugacy_errors["left"],
            "right_conjugacy_relative_error": conjugacy_errors["right"],
            "max_conjugacy_error": max_conjugacy_error,
            "left_left_response": response_amplitudes["left"]["left"],
            "left_right_response": response_amplitudes["left"]["right"],
            "right_left_response": response_amplitudes["right"]["left"],
            "right_right_response": response_amplitudes["right"]["right"],
            "binocular_left_response": binocular_directional_responses["left"],
            "binocular_right_response": binocular_directional_responses["right"],
            "coordinate_orientation": coordinate_orientation,
            "left_rest_gaze_bias_degrees": rest_bias_degrees["left"],
            "right_rest_gaze_bias_degrees": rest_bias_degrees["right"],
            "conjugate_rest_gaze_deviation_degrees": conjugate_rest_deviation,
        }
        if minimum_response_snr is not None:
            metrics["minimum_response_snr"] = minimum_response_snr

        # 这些连续指标已具备可重复记录路径，但候选界值尚未由目标摄像头的
        # 健康受试者数据估计。默认仅返回技术验证记录，避免内部工程值形成
        # 自动阳性或阴性结论。显式启用只供预先声明的离线消融研究。
        if not self.config.eye_enable_unvalidated_warning_thresholds:
            return MotionResult(
                status="insufficient",
                reason="eye_metrics_recorded_for_validation",
                quality=quality,
                metrics=metrics,
            )

        if persistent_failures:
            failed_eyes = {eye for eye, _ in persistent_failures}
            failed_directions = {
                direction for _, direction in persistent_failures
            }
            if len(persistent_failures) == 4:
                return MotionResult(
                    status="insufficient",
                    reason="visual_target_following_not_demonstrated",
                    quality=quality,
                    metrics=metrics,
                )
            bilateral_directions = [
                direction
                for direction in failed_directions
                if {
                    eye
                    for eye, failed_direction in persistent_failures
                    if failed_direction == direction
                }
                == {"left", "right"}
            ]
            if bilateral_directions:
                return MotionResult(
                    status="positive",
                    reason="bilateral_directional_gaze_restriction",
                    quality=quality,
                    metrics=metrics,
                    details={
                        "restricted_gaze_direction": bilateral_directions[0]
                    },
                )
            return MotionResult(
                status="positive",
                reason="possible_disconjugate_gaze_restriction",
                affected_side=next(iter(failed_eyes)) if len(failed_eyes) == 1 else None,
                quality=quality,
                metrics=metrics,
            )
        if (
            conjugate_rest_deviation
            >= self.config.eye_rest_gaze_deviation_degrees_threshold
        ):
            return MotionResult(
                status="positive",
                reason="conjugate_rest_gaze_deviation",
                quality=quality,
                metrics=metrics,
                details={
                    "gaze_direction": (
                        "right"
                        if rest_bias_degrees["left"] > 0.0
                        else "left"
                    )
                },
            )
        if (
            binocular_directional_asymmetry
            >= self.config.eye_directional_asymmetry_threshold
        ):
            return MotionResult(
                status="positive",
                reason="binocular_directional_gaze_hypometria",
                quality=quality,
                metrics=metrics,
            )
        if (
            max_conjugacy_error
            >= self.config.eye_conjugacy_relative_error_threshold
        ):
            direction = max(conjugacy_errors, key=conjugacy_errors.get)
            affected_eye = (
                "left"
                if response_medians["left"][direction]
                < response_medians["right"][direction]
                else "right"
            )
            return MotionResult(
                status="positive",
                reason="possible_binocular_endpoint_dysconjugacy",
                affected_side=affected_eye,
                quality=quality,
                metrics=metrics,
            )
        return MotionResult(
            status="negative",
            reason="no_repeatable_visible_eye_endpoint_abnormality",
            quality=quality,
            metrics=metrics,
        )
