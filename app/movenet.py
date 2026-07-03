"""TensorFlow Lite MoveNet inference wrapper."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from app.keypoints import KEYPOINT_NAMES


def _load_interpreter_class() -> Any:
    # Raspberry Pi 上优先使用轻量的 tflite-runtime，开发机上可退回 TensorFlow 自带解释器。
    try:
        from tflite_runtime.interpreter import Interpreter

        return Interpreter
    except ImportError:
        try:
            from tensorflow.lite.python.interpreter import Interpreter

            return Interpreter
        except ImportError as exc:
            raise RuntimeError(
                "No TFLite interpreter found. Install either tflite-runtime "
                "on Raspberry Pi or tensorflow on a development machine."
            ) from exc


# MoveNet 推理封装，负责把 RGB 图像送入 TFLite 模型并输出 17 个关键点。
class MoveNet:
    """Runs a single-pose MoveNet TFLite model and returns normalized keypoints."""

    def __init__(self, model_path: str | Path, num_threads: int = 2) -> None:
        # 模型文件不提交到仓库，运行前需要放到 models/ 或通过 --model 指定。
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"MoveNet model not found: {self.model_path}. "
                "Place movenet_lightning.tflite under models/ or pass --model."
            )

        interpreter_class = _load_interpreter_class()
        self.interpreter = interpreter_class(
            model_path=str(self.model_path), num_threads=num_threads
        )
        self.interpreter.allocate_tensors()
        # 输入/输出 tensor 信息由模型决定，后面 resize 时要使用模型声明的尺寸。
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        input_shape = self.input_details[0]["shape"]
        self.input_height = int(input_shape[1])
        self.input_width = int(input_shape[2])
        self.input_dtype = self.input_details[0]["dtype"]

    def infer(self, rgb_frame: np.ndarray) -> list[dict[str, float]]:
        """Run inference on an RGB frame and return MoveNet's 17 keypoints.

        Coordinates are normalized to 0..1 in the original image coordinate space.
        The returned dictionaries use x/y order, while MoveNet outputs y/x/score.
        """

        # MoveNet Lightning 常见输入是 192x192 RGB，但这里不写死，直接读取模型 shape。
        resized = cv2.resize(
            rgb_frame,
            (self.input_width, self.input_height),
            interpolation=cv2.INTER_LINEAR,
        )
        input_data = np.expand_dims(resized, axis=0)

        # 有些 TFLite 模型输入是 uint8，有些是 float32；根据 dtype 自动适配。
        if np.issubdtype(self.input_dtype, np.floating):
            input_data = input_data.astype(self.input_dtype) / 255.0
        else:
            input_data = input_data.astype(self.input_dtype)

        self.interpreter.set_tensor(self.input_details[0]["index"], input_data)
        self.interpreter.invoke()
        output = self.interpreter.get_tensor(self.output_details[0]["index"])
        points = np.squeeze(output)

        # MoveNet SinglePose 输出 17 x 3，每行是 y, x, score。
        if points.shape != (17, 3):
            points = points.reshape((17, 3))

        keypoints: list[dict[str, float]] = []
        for name, point in zip(KEYPOINT_NAMES, points):
            y, x, score = point.tolist()
            # 统一输出为 x/y/score，并把坐标限制在 0..1，方便后续规则和绘制使用。
            keypoints.append(
                {
                    "name": name,
                    "x": float(np.clip(x, 0.0, 1.0)),
                    "y": float(np.clip(y, 0.0, 1.0)),
                    "score": float(score),
                }
            )
        return keypoints
