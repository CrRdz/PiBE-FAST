"""把独立的 B/E/F/A/S 检查结果组装为前端使用的统一报告。"""

from __future__ import annotations

from typing import Any, Mapping

from .balance import balance_report_item
from .result import MotionResult, motion_report_item, report_item
from .speech import speech_report_item


def eye_report_item(
    reported_problem: bool | None,
    manual_completed: bool,
    motion_result: MotionResult,
) -> dict[str, Any]:
    """E 以突发视觉症状优先，摄像头终点检查只作研究测量。"""

    # 目前没有经过独立临床验证的摄像头 E 决策阈值。摄像头结果仍保留在
    # research fusion 中，但不得令最终安全状态变成 clear/urgent/incomplete。
    # 症状问卷单独记录是否完成，不把连续眼动指标解释成自动阴性结论。
    item = motion_report_item(motion_result, "mediapipe_eye_endpoint_research_only")
    item["decision_eligible"] = False
    item["decision_role"] = "research_measurement_only"
    # A technically qualified E trial is a measured research phenotype even
    # though no disease interpretation is available.  Keep acquisition and
    # interpretation on separate axes so fusion does not erase valid metrics.
    if (
        motion_result.reason == "eye_metrics_recorded_for_validation"
        and motion_result.metrics
        and motion_result.quality > 0.0
    ):
        item["acquisition_status"] = "measured"
        item["interpretation_status"] = "not_applicable"
    if manual_completed and reported_problem is False:
        item["reported_visual_problem"] = False
    # Keep questionnaire evidence separate from the camera record and its metrics.
    symptom_status = ("positive" if reported_problem else "negative") if manual_completed else "pending"
    symptom = report_item(
        symptom_status, "user_or_caregiver",
        "reported_visual_problem" if symptom_status == "positive" else
        "visual_problem_denied" if symptom_status == "negative" else "visual_question_unanswered",
    )
    symptom["decision_eligible"] = True
    return {
        **item, "status": symptom_status, "reason": symptom["reason"],
        "source": "user_or_caregiver", "decision_eligible": True,
        "decision_role": "symptom_report", "interpretation_status": symptom_status,
        "symptom_report": symptom, "camera_measurement": item,
    }


def build_report_items(
    manual: Mapping[str, bool | None],
    manual_complete: bool,
    eye_result: MotionResult,
    face_result: MotionResult,
    arm_result: MotionResult,
    balance_result: MotionResult,
    speech_result: MotionResult | None = None,
    manual_completed: Mapping[str, bool] | None = None,
    reported_functional_problems: Mapping[str, bool] | None = None,
    symptom_evidence: Mapping[str, list[dict[str, Any]]] | None = None,
) -> dict[str, dict[str, Any]]:
    """按 BEFAS 固定代码映射结果，并标注每项数据来源。"""

    completed = manual_completed or {
        "B": manual_complete,
        "S": manual_complete,
    }
    # 保持固定顺序，便于前端展示和测试稳定比较。
    items = {
        "B": balance_report_item(
            True if any(event.get("active", True) for event in (symptom_evidence or {}).get("B", []))
            else manual["balance_problem"] if completed.get("B", False) else None,
            balance_result,
        ),
        "E": eye_report_item(
            manual.get("eye_problem"),
            completed.get("E", False),
            eye_result,
        ),
        "F": motion_report_item(face_result, "mediapipe_face"),
        "A": motion_report_item(arm_result, "movenet_bilateral_arm_hold"),
        "S": speech_report_item(speech_result or MotionResult()),
    }
    for code, reported in (reported_functional_problems or {}).items():
        if code in {"F", "A", "S"} and reported is True:
            items[code]["reported_functional_problem"] = True
            items[code]["reported_problem_source"] = "user_or_caregiver"
    for code, evidence in (symptom_evidence or {}).items():
        items[code]["symptom_evidence"] = [dict(event) for event in evidence]
        items[code]["reported_functional_problem"] = any(event.get("active", True) for event in evidence)
    return items
