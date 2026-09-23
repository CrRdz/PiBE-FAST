"""集中保存引导式 BE-FAST 筛查使用的时间窗口和经验阈值。"""

from dataclasses import dataclass


@dataclass(frozen=True)
class BefastConfig:
    """工程默认值；正式用于临床前必须用合规数据重新验证和校准。"""

    # MoveNet 关键点低于该置信度时按不可见处理。
    min_keypoint_score: float = 0.35

    # A：先展示完整动作；双臂自然下垂稳定后自动倒数，也可手动提前开始。
    arm_warmup_seconds: float = 1.5  # 兼容旧配置，不再用于阶段切换。
    arm_phase_countdown_seconds: float = 3.0
    arm_auto_ready_enabled: bool = True
    arm_preview_min_seconds: float = 2.0
    arm_auto_ready_seconds: float = 1.2
    arm_capture_seconds: float = 18.0  # 兼容旧调用；实际由下列阶段时长控制。
    arm_raise_timeout_seconds: float = 8.0
    arm_hold_seconds: float = 5.0
    arm_lower_timeout_seconds: float = 5.0
    arm_pose_sustain_seconds: float = 0.45
    # 样本数和有效帧比例同时达标，才允许输出阴性或阳性结果。
    arm_min_valid_samples: int = 20
    arm_min_valid_fraction: float = 0.55
    arm_min_endpoint_samples: int = 6
    # 完成门控：必须从自然下垂开始、双臂横向抬至肩高、保持、再放下。
    arm_start_wrist_below_shoulder: float = 0.45
    arm_raise_wrist_height_tolerance: float = 0.42
    arm_min_elbow_angle_degrees: float = 125.0
    arm_min_lateral_reach: float = 0.45
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
    # 下列门槛仅判断工程测量是否可用，不判断疾病阳性/阴性。由于自动 E
    # 已明确排除在最终安全状态之外，默认启用采集质量门控以拒绝明显无效记录；
    # 数值仍需目标设备数据复核，不能解释为临床 cutoff。
    eye_enable_unvalidated_quality_gates: bool = True
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
    # MediaPipe 可输出亚像素坐标，但零 MAD 不等于零测量误差。每段终点噪声
    # 至少按 0.5 个眼部像素换算，再与另一段按独立误差平方和合并。
    eye_landmark_noise_floor_pixels: float = 0.5
    # 摄像头坐标镜像必须由采集配置明确给出，不能从受试者是否正确跟随目标反推。
    eye_camera_mirrored: bool = False
    # 未经目标设备健康受试者数据估计前，候选阈值只供离线消融，默认不参与
    # 摄像头 E 阳性或阴性判定。连续指标仍写入报告，供技术验证预先估计阈值。
    eye_enable_unvalidated_warning_thresholds: bool = False
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
    # 3.5 源于 Iglewicz--Hoaglin 修正 Z 分数的潜在离群值建议。本实现还取
    # MAD 与 IQR 尺度中的较大值，因此不比原始 MAD 规则更敏感。横向摆动
    # 仅在速度和 R90 同时越界时触发；该值是工程起点，不是卒中诊断阈值。
    balance_robust_z_threshold: float = 3.5

    # Feature-level BE-FAST fusion remains a research-only observation layer.
    # These references mirror the default guided-speech assessment thresholds
    # and only normalize a feature vector; they never alter the final decision.
    feature_fusion_severity_clip: float = 5.0
    fusion_speech_character_error_rate_reference: float = 0.35
    fusion_speech_pause_fraction_reference: float = 0.55
    fusion_speech_min_characters_per_second: float = 1.0
    fusion_speech_max_characters_per_second: float = 8.0
    # Frozen threshold exported by models/mdsc_dysarthria_v1.json. Individual
    # S reports retain their model-specific threshold; this is the fallback for
    # older reports that include the probability but not threshold metadata.
    fusion_speech_mdsc_probability_reference: float = 0.7352820324592312
