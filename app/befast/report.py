"""把独立的 B/E/F/A/S 检查结果组装为前端使用的统一报告。"""

from __future__ import annotations

from typing import Any, Mapping

from .balance import balance_report_item
from .result import MotionResult, motion_report_item
from .speech import speech_report_item


def build_report_items(
    manual: Mapping[str, bool | None],
    manual_complete: bool,
    eye_result: MotionResult,
    face_result: MotionResult,
    arm_result: MotionResult,
    balance_result: MotionResult,
    speech_result: MotionResult | None = None,
    manual_completed: Mapping[str, bool] | None = None,
) -> dict[str, dict[str, Any]]:
    """按 BEFAS 固定代码映射结果，并标注每项数据来源。"""

    completed = manual_completed or {
        "B": manual_complete,
        "S": manual_complete,
    }
    # 保持固定顺序，便于前端展示和测试稳定比较。
    return {
        "B": balance_report_item(
            manual["balance_problem"] if completed.get("B", False) else None,
            balance_result,
        ),
        "E": motion_report_item(eye_result, "mediapipe_face"),
        "F": motion_report_item(face_result, "mediapipe_face"),
        "A": motion_report_item(arm_result, "pose"),
        "S": speech_report_item(speech_result or MotionResult()),
    }
