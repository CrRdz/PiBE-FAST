"""S（Speech）：把树莓派本地麦克风分析转换为统一报告项。"""

from __future__ import annotations

from typing import Any

from .result import MotionResult, motion_report_item


def speech_report_item(
    result: MotionResult,
) -> dict[str, Any]:
    """将麦克风、音频质量和离线识别结果转换为统一报告项。"""

    return motion_report_item(result, "microphone+offline_asr")
