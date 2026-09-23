"""T（Time）：结合阳性体征和是否突然发生，给出筛查紧急程度。"""

from __future__ import annotations

from typing import Any, Mapping, Sequence


def screening_assessment(
    items: Mapping[str, Mapping[str, Any]],
    onset_by_component: Mapping[str, bool | None] | bool | None,
    required_components: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Return independent urgency and completeness axes plus a legacy decision."""

    # E camera data is never used as questionnaire completion or interpretation.
    items = {code: ({**item["symptom_report"],
                     "symptom_evidence": item.get("symptom_evidence", []),
                     "reported_functional_problem": item.get("reported_functional_problem", False),
                     "measurement_status": "not_applicable"}
                    if code == "E" and "symptom_report" in item else item)
             for code, item in items.items()}
    decision_items = {
        code: item
        for code, item in items.items()
        if item.get("decision_eligible", True) is not False
    }
    positives = [
        code
        for code, item in decision_items.items()
        if item.get("status") == "positive" or item.get("reported_functional_problem") is True
    ]

    def onset(code: str) -> bool | None:
        if isinstance(onset_by_component, Mapping):
            return onset_by_component.get(code)
        return onset_by_component

    def urgent_positive(code: str) -> bool:
        item = decision_items[code]
        evidence = [e for e in item.get("symptom_evidence", []) if e.get("active", True)]
        reported_urgent = any(event.get("new_or_sudden") is not False for event in evidence)
        # Legacy callers without event records retain their explicit onset contract.
        if item.get("reported_functional_problem") is True and not evidence:
            reported_urgent = onset(code) is not False
        measured_status = item.get("measurement_status", item.get("status")) if evidence else item.get("status")
        return reported_urgent or (measured_status == "positive" and onset(code) is not False)

    urgent = [code for code in positives if urgent_positive(code)]
    urgency = "urgent" if urgent else ("warning" if positives else "none")

    required = tuple(decision_items) if required_components is None else tuple(required_components)
    statuses = [str(decision_items.get(code, {}).get("completion_status", decision_items.get(code, {}).get("status", "pending"))) for code in required]
    incomplete = any(
        status not in {"positive", "negative", "safety_exempt"} for status in statuses
    ) or not statuses
    completeness = "incomplete" if incomplete else "complete"

    reasons = [
        f"{code}:reported_functional_problem"
        if items[code].get("reported_functional_problem") is True
        else f"{code}:{items[code]['reason']}"
        for code in positives
    ]
    if incomplete:
        if "insufficient" in statuses:
            reasons.append("motion_check_quality_insufficient")
        elif "skipped" in statuses:
            reasons.append("one_or_more_checks_skipped")
        else:
            reasons.append("complete_all_befast_checks")
    if not positives and not incomplete:
        reasons.append("no_obvious_befast_sign_detected")

    decision = (
        "emergency"
        if urgency == "urgent"
        else "warning"
        if urgency == "warning"
        else "incomplete"
        if incomplete
        else "clear"
    )
    return {
        "decision": decision,
        "urgency": urgency,
        "completeness": completeness,
        "required_components": list(required),
        "positive_components": positives,
        "urgent_components": urgent,
        "reasons": reasons,
    }


def screening_decision(
    items: Mapping[str, Mapping[str, Any]],
    new_or_sudden: Mapping[str, bool | None] | bool | None,
) -> tuple[str, list[str]]:
    """Backward-compatible scalar view of :func:`screening_assessment`."""

    assessment = screening_assessment(items, new_or_sudden)
    return str(assessment["decision"]), list(assessment["reasons"])
