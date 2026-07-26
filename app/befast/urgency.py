"""T（Time）：结合阳性体征和是否突然发生，给出筛查紧急程度。"""

from __future__ import annotations

from typing import Any, Mapping


def screening_decision(
    items: Mapping[str, Mapping[str, Any]],
    new_or_sudden: bool | None,
) -> tuple[str, list[str]]:
    """根据 B/E/F/A/S 报告项和起病方式生成最终决策及原因列表。"""

    # 收集所有阳性项，原因中保留字母代码，方便前端解释触发来源。
    positives = [code for code, item in items.items() if item["status"] == "positive"]
    if positives:
        reasons = [f"{code}:{items[code]['reason']}" for code in positives]
        # 只有“存在阳性体征且为新发/突发”时才升级到 emergency。
        if new_or_sudden is True:
            return "emergency", reasons
        return "warning", reasons

    statuses = [str(item["status"]) for item in items.values()]
    # clear 要求所有项目均为阴性；任何未完成状态都不能被当作排除卒中。
    if statuses and all(status == "negative" for status in statuses):
        return "clear", ["no_obvious_befast_sign_detected"]
    if "insufficient" in statuses:
        # 质量不足优先于 skipped，向用户突出重新采集的必要性。
        return "incomplete", ["motion_check_quality_insufficient"]
    if "skipped" in statuses:
        return "incomplete", ["one_or_more_checks_skipped"]
    return "incomplete", ["complete_all_befast_checks"]
