"""Centralized runtime and heuristic configuration."""

from dataclasses import dataclass


# 姿态分类用的规则阈值，主要控制 standing / sitting / lying 的判定边界。
@dataclass(frozen=True)
class PoseClassifierConfig:
    # MoveNet 每个关键点都有 score。低于这个值的点会被当成“不可靠点”忽略。
    min_keypoint_score: float = 0.25
    # 至少要有多少个可靠关键点，才尝试分类；太少时返回 unknown。
    min_visible_keypoints: int = 5
    # 站立时人体通常“高大于宽”，这里要求 bbox 高/宽达到一定比例。
    standing_height_width_ratio: float = 1.25
    # 躯干与竖直方向的夹角，小于这个角度更像站立。
    standing_torso_max_degrees: float = 35.0
    # 允许关键点排序有一点误差：肩 -> 胯 -> 膝 -> 脚踝 应大致从上到下。
    standing_order_tolerance: float = 0.04
    # 躺下时人体通常“宽大于高”，这里要求 bbox 宽/高达到一定比例。
    lying_width_height_ratio: float = 1.15
    # 躯干与竖直方向的夹角，大于这个角度更像水平躺下。
    lying_torso_min_degrees: float = 60.0
    # 躺下时肩、胯、膝、踝在 y 方向的离散程度较小，表示整体趋向水平。
    lying_body_y_spread_ratio: float = 0.35
    # 坐姿时胯和膝盖的高度会比较接近，这里用 bbox 高度的比例做容忍。
    sitting_hip_knee_y_ratio: float = 0.22
    # 膝盖角度越小代表弯曲越明显；坐姿通常小于完全伸直的 180 度。
    sitting_knee_bend_max_degrees: float = 155.0
    # 坐姿躯干通常仍接近竖直，但允许比站立更倾斜。
    sitting_torso_max_degrees: float = 55.0
    # 关键点质量不足或没有规则匹配时输出 unknown，而不是强行分类。
    fallback_pose: str = "unknown"


# 摔倒状态机用的时间窗口和动作变化阈值，避免把普通躺下直接当成摔倒。
@dataclass(frozen=True)
class FallDetectorConfig:
    # 状态机最多保留最近多少秒的帧摘要，用来判断“之前是不是站/坐着”。
    history_seconds: float = 8.0
    # 快速摔倒转变必须发生在这个时间窗口内，窗口过大容易把慢慢躺下误判。
    transition_seconds: float = 1.6
    # 人体中心点 y 坐标下降幅度。归一化坐标里 y 越大越靠下。
    center_drop_threshold: float = 0.18
    # 躯干角度变化阈值：从接近竖直到接近水平需要有足够大的角度变化。
    torso_angle_delta_threshold: float = 40.0
    # 之前帧的躯干角度小于该值，才认为之前是比较竖直的姿态。
    vertical_torso_max_degrees: float = 40.0
    # 当前帧的躯干角度大于该值，才认为当前已经接近水平。
    horizontal_torso_min_degrees: float = 60.0
    # 出现快速下落后，还要持续 lying 这么久才最终确认 fall。
    lying_hold_seconds: float = 2.0
    # falling 候选状态最多保留多久；超过后还没躺下就取消候选。
    max_candidate_seconds: float = 5.0
    # fall 触发后，持续站/坐这么久才认为已经恢复。
    recovery_hold_seconds: float = 1.5


# 程序运行时的通用参数，包括预览分辨率、帧率、日志和事件片段长度。
@dataclass(frozen=True)
class RuntimeConfig:
    # 摄像头采集和事件视频推荐分辨率。视频文件输入时不会强行改原视频尺寸。
    frame_width: int = 640
    frame_height: int = 480
    # 推荐处理帧率；摄像头模式会尝试设置，实际值取决于硬件。
    frame_fps: int = 15
    # Web 预览把画面编码成 JPEG，质量越高越清晰但带宽和 CPU 消耗越高。
    jpeg_quality: int = 80
    # 命令行日志输出间隔，避免每一帧都刷屏。
    log_every_seconds: float = 1.0
    # fall 事件片段保存前后各多少秒。
    event_pre_seconds: float = 5.0
    event_post_seconds: float = 5.0
