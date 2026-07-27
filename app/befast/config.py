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

    # B：卒中静态平衡研究常采用 30 秒安静站立试次，并通过重复试次提高
    # 可靠性（Aryan et al. 2023; Ruhe et al. 2010）。这里不再使用固定的
    # “肩宽百分比阳性阈值”，而是建立同一人的多窗口稳健基线。
    balance_warmup_seconds: float = 1.5
    balance_capture_seconds: float = 30.0
    # 下列两项仅是 MoveNet 可解释性质量门槛，不参与异常 cutoff；目前没有论文
    # 能为本项目的相机、帧率和遮挡条件给出可直接移植的数值。
    balance_min_valid_samples: int = 30
    balance_min_valid_fraction: float = 0.75
    # 五个重复窗口取自静态姿势测量通常需要 3～5 次重复的可靠性建议。
    balance_baseline_windows: int = 5
    # 3.5 是 median/MAD 稳健异常分数的常用统计界值；它只表示相对个人
    # 基线的显著变化，不是卒中诊断阈值。
    balance_robust_z_threshold: float = 3.5
