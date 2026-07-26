"""集中保存引导式 BE-FAST 筛查使用的时间窗口和经验阈值。"""

from dataclasses import dataclass


@dataclass(frozen=True)
class BefastConfig:
    """工程默认值；正式用于临床前必须用合规数据重新验证和校准。"""

    # MoveNet 关键点低于该置信度时按不可见处理。
    min_keypoint_score: float = 0.35

    # A：先预留抬臂准备时间，再进入正式采样窗口。
    arm_warmup_seconds: float = 1.5
    arm_capture_seconds: float = 6.0
    # 样本数和有效帧比例同时达标，才允许输出阴性或阳性结果。
    arm_min_valid_samples: int = 20
    arm_min_valid_fraction: float = 0.55
    # 手臂距离统一除以肩宽，以降低人与相机距离变化造成的影响。
    arm_level_difference_threshold: float = 0.30
    arm_drift_difference_threshold: float = 0.22
    # 手腕相对肩膀低得过多，说明受试者没有完成抬臂动作。
    arm_max_initial_wrist_below_shoulder: float = 0.85

    # MediaPipe Face Landmarker 以低于摄像头帧率的频率运行，减少树莓派负载。
    face_inference_fps: float = 5.0
    # 双眼外眼角距离过小时，面部像素不足，E/F 检查按质量不足处理。
    face_min_interocular_width: float = 0.075

    # E：每个目标停留 3 秒，让用户有时间看见、重新聚焦并保持注视。
    eye_target_seconds: float = 3.0
    eye_min_samples_per_target: int = 4
    eye_min_valid_fraction: float = 0.50
    # 依次限制总眼球活动范围、双眼范围差、共轭运动误差和头部代偿。
    eye_gaze_range_threshold: float = 0.12
    eye_range_asymmetry_threshold: float = 0.10
    eye_conjugacy_threshold: float = 0.14
    eye_head_motion_threshold: float = 0.35

    # F：先采集中性表情作为个人基线，再采集微笑阶段。
    face_neutral_seconds: float = 2.0
    face_smile_seconds: float = 3.0
    face_min_samples_per_phase: int = 5
    face_min_valid_fraction: float = 0.50
    # 微笑强度不足时不判断对称性，避免把“没有笑”误报为面瘫表现。
    face_min_smile_score: float = 0.22
    face_corner_delta_threshold: float = 0.075
    face_smile_score_difference_threshold: float = 0.24

    # B：与抬臂检查相同，准备期不计入正式平衡样本。
    balance_warmup_seconds: float = 1.5
    balance_capture_seconds: float = 6.0
    balance_min_valid_samples: int = 20
    balance_min_valid_fraction: float = 0.55
    # 身体中心偏移和摆动范围均除以肩宽，保证阈值与画面尺度无关。
    balance_offset_threshold: float = 0.40
    balance_sway_range_threshold: float = 0.50
