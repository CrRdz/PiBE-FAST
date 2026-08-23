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
    # MediaPipe x/y 分别按画面宽/高归一化。直接在归一化坐标上旋转会在
    # 非正方形画面中产生各向异性误差；先统一到“画面宽度”为单位的欧氏坐标。
    y_scale = (
        observation.frame_height / observation.frame_width
        if observation.frame_width > 0 and observation.frame_height > 0
        else 1.0
    )

    def isotropic_xy(point: tuple[float, float, float]) -> Point:
        return float(point[0]), float(point[1]) * y_scale

    outer_right_xy = isotropic_xy(outer_right)
    outer_left_xy = isotropic_xy(outer_left)
    dx = outer_left_xy[0] - outer_right_xy[0]
    dy = outer_left_xy[1] - outer_right_xy[1]
    scale = hypot(dx, dy)
    # 眼距过小通常意味着人脸太远，后续比值会被像素噪声放大。
    if scale <= 0.0 or scale < min_interocular_width:
        return None

    center_x = (outer_left_xy[0] + outer_right_xy[0]) / 2.0
    center_y = (outer_left_xy[1] + outer_right_xy[1]) / 2.0
    angle = atan2(dy, dx)
    cosine = cos(angle)
    sine = sin(angle)
    aligned: dict[int, Point] = {}
    # 先平移到双眼中心，再旋转到双眼连线水平；不改变原始观察对象。
    for index in indices:
        point = observation.point(index)
        if point is None:
            return None
        point_x, point_y = isotropic_xy(point)
        relative_x = point_x - center_x
        relative_y = point_y - center_y
        aligned[index] = (
            cosine * relative_x + sine * relative_y,
            -sine * relative_x + cosine * relative_y,
        )
    return aligned, scale
