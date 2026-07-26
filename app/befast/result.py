"""定义所有自动检查共用的结果模型，以及面向 API 的序列化格式。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class MotionResult:
    """一次自动检查的状态、原因、质量和可解释指标。"""

    # status 使用 not_run/checking/positive/negative/insufficient/skipped。
    status: str = "not_run"
    reason: str = "not_run"
    affected_side: str | None = None
    quality: float = 0.0
    metrics: dict[str, float] = field(default_factory=dict)
    details: dict[str, str] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "reason": self.reason,
            "affected_side": self.affected_side,
            "quality": round(self.quality, 4),
            "metrics": {
                key: round(float(value), 5) for key, value in self.metrics.items()
            },
            "details": {str(key): str(value) for key, value in self.details.items()},
        }


def report_item(status: str, source: str, reason: str) -> dict[str, Any]:
    """生成没有运动指标的标准报告项，主要供人工观察结果使用。"""

    return {
        "status": status,
        "source": source,
        "reason": reason,
        "affected_side": None,
        "quality": None,
        "metrics": {},
    }


def motion_report_item(result: MotionResult, source: str) -> dict[str, Any]:
    """把内部运动结果转换为前端统一使用的报告项。"""

    # 对前端而言，尚未运行和正在检查分别表现为 pending 与 checking。
    if result.status in {"not_run", "checking"}:
        status = "pending" if result.status == "not_run" else "checking"
    else:
        status = result.status
    item = result.as_dict()
    item["status"] = status
    item["source"] = source
    return item
