"""Thin MediaPipe Face Landmarker adapter for guided E/F checks."""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
import time
from typing import Any

import numpy as np


@dataclass(frozen=True)
class FaceObservation:
    """One detected face represented without exposing MediaPipe-specific types."""

    ts: float
    landmarks: tuple[tuple[float, float, float], ...]
    blendshapes: dict[str, float] = field(default_factory=dict)
    inference_ms: float = 0.0

    def point(self, index: int) -> tuple[float, float, float] | None:
        if index < 0 or index >= len(self.landmarks):
            return None
        return self.landmarks[index]


class MediaPipeFaceLandmarker:
    """Run the official Face Landmarker task in tracking-enabled video mode."""

    def __init__(
        self,
        model_path: str | Path,
        min_detection_confidence: float = 0.5,
        min_presence_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        cache_dir: str | Path = "data/cache/matplotlib",
    ) -> None:
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"MediaPipe face model not found: {self.model_path}. "
                "Download face_landmarker.task or pass --face-model."
            )

        # MediaPipe imports optional drawing helpers that initialize matplotlib.
        # A persistent writable cache prevents a slow font-cache rebuild each launch.
        cache_path = Path(cache_dir)
        cache_path.mkdir(parents=True, exist_ok=True)
        cache_root = cache_path.parent.resolve()
        os.environ.setdefault("MPLCONFIGDIR", str(cache_path.resolve()))
        os.environ.setdefault("XDG_CACHE_HOME", str(cache_root))

        try:
            import mediapipe as mp
        except ImportError as exc:
            raise RuntimeError(
                "MediaPipe is not installed. Run: python -m pip install mediapipe"
            ) from exc

        self.mp = mp
        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(
                model_asset_path=str(self.model_path.resolve()),
                delegate=mp.tasks.BaseOptions.Delegate.CPU,
            ),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=float(min_detection_confidence),
            min_face_presence_confidence=float(min_presence_confidence),
            min_tracking_confidence=float(min_tracking_confidence),
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=False,
        )
        self.landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(options)
        self.last_timestamp_ms = -1

    def infer(self, rgb_frame: np.ndarray, ts: float) -> FaceObservation | None:
        timestamp_ms = max(self.last_timestamp_ms + 1, int(float(ts) * 1000.0))
        self.last_timestamp_ms = timestamp_ms
        image = self.mp.Image(
            image_format=self.mp.ImageFormat.SRGB,
            data=np.ascontiguousarray(rgb_frame),
        )
        started = time.perf_counter()
        result = self.landmarker.detect_for_video(image, timestamp_ms)
        inference_ms = (time.perf_counter() - started) * 1000.0
        if not result.face_landmarks:
            return None

        landmarks = tuple(
            (float(point.x), float(point.y), float(point.z))
            for point in result.face_landmarks[0]
        )
        blendshapes: dict[str, float] = {}
        if result.face_blendshapes:
            for category in result.face_blendshapes[0]:
                name = str(category.category_name or category.display_name or "")
                if name:
                    blendshapes[name] = float(category.score)

        return FaceObservation(
            ts=float(ts),
            landmarks=landmarks,
            blendshapes=blendshapes,
            inference_ms=inference_ms,
        )

    def close(self) -> None:
        self.landmarker.close()

    def __enter__(self) -> "MediaPipeFaceLandmarker":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()
