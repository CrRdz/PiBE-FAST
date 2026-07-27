"""生成实时取景/动作引导；本模块不负责修改筛查状态机。"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from app.face_landmarker import FaceObservation

from .arms import arm_frame_metrics
from .balance import balance_frame_metrics
from .config import BefastConfig
from .eyes import eye_frame_metrics
from .face import face_frame_metrics


def guidance_value(
    stage: str,
    ready: bool,
    reason: str,
    ts: float,
    metrics: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """创建结构统一、数值已舍入的实时引导对象。"""

    return {
        "ready": bool(ready),
        "status": "correct" if ready else "adjust",
        "reason": str(reason),
        "stage": stage,
        "updated_at": round(float(ts), 4),
        "metrics": {
            str(key): round(float(value), 5)
            for key, value in (metrics or {}).items()
            if isinstance(value, (int, float))
        },
    }


def face_guidance(
    stage: str,
    ts: float,
    observation: FaceObservation | None,
    config: BefastConfig,
    face_phase: str = "neutral",
) -> dict[str, Any] | None:
    """根据当前 E/F 阶段判断人脸、眼睛或表情是否准备就绪。"""

    if stage in {"idle", "retry_eyes", "eyes"}:
        # E 需要同时看到完整眼角、虹膜和鼻尖，指标可计算即视为取景就绪。
        metrics = (
            eye_frame_metrics(
                observation,
                config.face_min_interocular_width,
                config.eye_min_eye_width_pixels,
            )
            if observation is not None
            else None
        )
        return guidance_value(
            stage,
            metrics is not None,
            "face_and_eyes_ready" if metrics is not None else "show_face_and_eyes",
            ts,
            metrics,
        )
    if stage not in {"ready_face", "retry_face", "face"}:
        # 非面部阶段返回 None，调用方保留最近一次有效引导。
        return None

    metrics = (
        face_frame_metrics(observation, config.face_min_interocular_width)
        if observation is not None
        else None
    )
    if metrics is None:
        return guidance_value(stage, False, "show_full_face", ts)
    smile_strength = max(metrics["left_smile"], metrics["right_smile"])
    if face_phase == "smile":
        # 微笑阶段要求表达强度达标；中性阶段则要求先放松表情。
        ready = smile_strength >= config.face_min_smile_score
        reason = "smile_detected" if ready else "smile_now"
    else:
        ready = smile_strength < config.face_min_smile_score
        reason = "neutral_face_ready" if ready else "relax_face_first"
    return guidance_value(stage, ready, reason, ts, metrics)


def pose_guidance(
    stage: str,
    ts: float,
    keypoints: Sequence[Mapping[str, float]],
    pose: str,
    config: BefastConfig,
) -> dict[str, Any] | None:
    """根据当前 A/B 阶段判断关键点可见性和动作准备状态。"""

    if stage in {"ready_arms", "retry_arms", "arms"}:
        metrics = arm_frame_metrics(keypoints, config.min_keypoint_score)
        if metrics is None:
            return guidance_value(stage, False, "show_shoulders_and_wrists", ts)
        # 这里不能要求两臂水平，否则会屏蔽 A 检查本来要发现的单侧无力。
        one_arm_raised = min(
            metrics["left_wrist_relative_y"],
            metrics["right_wrist_relative_y"],
        ) <= config.arm_max_initial_wrist_below_shoulder
        return guidance_value(
            stage,
            one_arm_raised,
            "arms_detected_hold_still" if one_arm_raised else "raise_both_arms",
            ts,
            metrics,
        )
    if stage not in {"ready_balance", "retry_balance", "balance"}:
        return None

    metrics = balance_frame_metrics(keypoints, config.min_keypoint_score)
    # 既要完整关键点，也要姿态分类为 standing，才能提示开始安全采样。
    ready = metrics is not None and pose == "standing"
    return guidance_value(
        stage,
        ready,
        "standing_pose_ready" if ready else "show_full_body_and_stand_safely",
        ts,
        metrics,
    )
