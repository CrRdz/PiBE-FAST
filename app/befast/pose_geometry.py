"""A/B 姿态检查共用的关键点读取和稳健统计辅助函数。"""

from __future__ import annotations

from typing import Mapping, Sequence


Point = tuple[float, float]


def visible_point(
    points: Mapping[str, Mapping[str, float]], name: str, min_score: float
) -> Point | None:
    """只返回置信度达标的二维关键点，否则返回 ``None``。"""

    point = points.get(name)
    if point is None or float(point.get("score", 0.0)) < min_score:
        return None
    return float(point["x"]), float(point["y"])


def percentile(values: Sequence[float], fraction: float) -> float:
    """计算离散最近秩百分位数，用于抑制单帧抖动和极端值。"""

    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    # 先把 fraction 限制在 [0, 1]，再映射到最接近的样本下标。
    index = int(round((len(ordered) - 1) * max(0.0, min(1.0, fraction))))
    return ordered[index]
