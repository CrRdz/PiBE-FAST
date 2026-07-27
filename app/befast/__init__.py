"""可解释、无需训练数据的 BE-FAST 筛查包公共入口。

具体实现按 B/E/F/A/S/T 和共享职责拆分；调用方仍可直接从
``app.befast`` 导入公共类，避免目录重构影响已有代码。
"""

from .arms import ArmDriftScreen, arm_frame_metrics
from .balance import (
    BALANCE_EVIDENCE_VERSION,
    BalanceScreen,
    PersonalBalanceBaseline,
    balance_frame_metrics,
    summarize_balance_window,
)
from .config import BefastConfig
from .eyes import EyeMovementScreen, eye_frame_metrics
from .face import FaceSmileScreen, face_frame_metrics
from .face_geometry import aligned_face_points
from .pose_geometry import Point, percentile, visible_point
from .result import MotionResult, motion_report_item, report_item
from .session import BefastSession

# 兼容旧单文件版本中直接导入私有辅助函数的代码；新代码应优先使用公共类。
_aligned_face_points = aligned_face_points
_arm_frame_metrics = arm_frame_metrics
_balance_frame_metrics = balance_frame_metrics
_eye_frame_metrics = eye_frame_metrics
_face_frame_metrics = face_frame_metrics
_item = report_item
_motion_item = motion_report_item
_percentile = percentile
_point = visible_point

__all__ = [
    "ArmDriftScreen",
    "BalanceScreen",
    "PersonalBalanceBaseline",
    "BALANCE_EVIDENCE_VERSION",
    "BefastConfig",
    "BefastSession",
    "EyeMovementScreen",
    "FaceSmileScreen",
    "MotionResult",
    "Point",
    "summarize_balance_window",
]
