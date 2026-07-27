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

    # E：中心和左右目标按配对试次重复三次。每次切换后先留出稳定时间，
    # 只分析随后保持注视的低帧率终点，不把它描述成扫视速度/潜伏期测量。
    eye_target_seconds: float = 2.0
    eye_settle_seconds: float = 0.5
    eye_min_samples_per_trial: int = 5
    eye_min_valid_fraction_per_trial: float = 0.60
    # 实际视频帧尺寸可用时，同时要求每只眼有足够像素；7.5% 脸宽规则不再
    # 单独决定 E 的可用性。该像素门槛仍是实现质量参数，需按设备验证。
    eye_min_eye_width_pixels: float = 24.0
    # MAD、重复差和头姿只负责拒绝不稳定测量，不是卒中阳性 cutoff。
    eye_max_gaze_mad: float = 0.08
    # 三次响应取中位数并允许一个离群试次；1.0 只要求最接近中位数的另
    # 一次响应处在约 3 倍量级内，不再要求每两次幅度都接近。
    eye_max_repeat_relative_error: float = 1.00
    eye_min_head_pose_fraction: float = 0.80
    eye_max_head_rotation_degrees: float = 8.0
    # 3.5 表示可见终点位移需明显高于同一试次的稳健噪声。下列相对差阈值
    # 仍是待临床标定的研究参数，但不再依赖眼裂绝对比例或拍摄距离。
    eye_response_snr_threshold: float = 3.5
    eye_directional_asymmetry_threshold: float = 0.45
    eye_conjugacy_relative_error_threshold: float = 0.35
    # CT/MRI 研究中约 12～14° 的共轭眼偏向具有较高特异度；这里只把
    # 12° 用作待外部验证的高幅度候选界值，不能视为可直接迁移的临床 cutoff。
    eye_rest_gaze_deviation_degrees_threshold: float = 12.0
    # 前端根据物理屏宽和观看距离把目标放到约 ±15°；默认值只用于预填表单。
    eye_target_visual_angle_degrees: float = 15.0
    eye_default_viewing_distance_cm: float = 50.0
    eye_default_screen_width_cm: float = 31.0

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
