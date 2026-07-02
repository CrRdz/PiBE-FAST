"""TensorFlow Lite MoveNet inference wrapper."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from app.keypoints import KEYPOINT_NAMES


def _load_interpreter_class() -> Any:
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


class MoveNet:
    """Runs a single-pose MoveNet TFLite model and returns normalized keypoints."""

    def __init__(self, model_path: str | Path, num_threads: int = 2) -> None:
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

        resized = cv2.resize(
            rgb_frame,
            (self.input_width, self.input_height),
            interpolation=cv2.INTER_LINEAR,
        )
        input_data = np.expand_dims(resized, axis=0)

        if np.issubdtype(self.input_dtype, np.floating):
            input_data = input_data.astype(self.input_dtype) / 255.0
        else:
            input_data = input_data.astype(self.input_dtype)

        self.interpreter.set_tensor(self.input_details[0]["index"], input_data)
        self.interpreter.invoke()
        output = self.interpreter.get_tensor(self.output_details[0]["index"])
        points = np.squeeze(output)

        if points.shape != (17, 3):
            points = points.reshape((17, 3))

        keypoints: list[dict[str, float]] = []
        for name, point in zip(KEYPOINT_NAMES, points):
            y, x, score = point.tolist()
            keypoints.append(
                {
                    "name": name,
                    "x": float(np.clip(x, 0.0, 1.0)),
                    "y": float(np.clip(y, 0.0, 1.0)),
                    "score": float(score),
                }
            )
        return keypoints
