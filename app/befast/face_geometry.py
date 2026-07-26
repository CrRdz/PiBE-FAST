"""E/F 面部检查共用的滚转校正与尺度归一化。"""

from __future__ import annotations

from math import atan2, cos, hypot, sin
from typing import Sequence

from app.face_landmarker import FaceObservation


Point = tuple[float, float]


def aligned_face_points(
    observation: FaceObservation,
    indices: Sequence[int],
    min_interocular_width: float,
) -> tuple[dict[int, Point], float] | None:
    """以双眼外眼角为基准校正画面内旋转，并返回眼距归一化尺度。"""

    # 33/263 分别是受试者右/左眼的外眼角。
    outer_right = observation.point(33)
    outer_left = observation.point(263)
    if outer_right is None or outer_left is None:
        return None
    dx = outer_left[0] - outer_right[0]
    dy = outer_left[1] - outer_right[1]
    scale = hypot(dx, dy)
    # 眼距过小通常意味着人脸太远，后续比值会被像素噪声放大。
    if scale < min_interocular_width:
        return None

    center_x = (outer_left[0] + outer_right[0]) / 2.0
    center_y = (outer_left[1] + outer_right[1]) / 2.0
    angle = atan2(dy, dx)
    cosine = cos(angle)
    sine = sin(angle)
    aligned: dict[int, Point] = {}
    # 先平移到双眼中心，再旋转到双眼连线水平；不改变原始观察对象。
    for index in indices:
        point = observation.point(index)
        if point is None:
            return None
        relative_x = point[0] - center_x
        relative_y = point[1] - center_y
        aligned[index] = (
            cosine * relative_x + sine * relative_y,
            -sine * relative_x + cosine * relative_y,
        )
    return aligned, scale
