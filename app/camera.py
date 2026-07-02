"""Frame sources for local videos, USB cameras, and optional Picamera2."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time

import cv2


@dataclass(frozen=True)
class Frame:
    image: object
    ts: float


class FrameSource:
    def __init__(
        self,
        source: str,
        width: int = 640,
        height: int = 480,
        fps: int = 15,
        camera_backend: str = "opencv",
    ) -> None:
        self.source = source
        self.width = width
        self.height = height
        self.requested_fps = fps
        self.camera_backend = camera_backend
        self.capture: cv2.VideoCapture | None = None
        self.picam2 = None
        self._fps = float(fps)

    @property
    def fps(self) -> float:
        return self._fps

    def open(self) -> "FrameSource":
        if self.source == "camera" and self.camera_backend == "picamera2":
            self._open_picamera2()
        else:
            self._open_opencv()
        return self

    def read(self) -> Frame | None:
        if self.picam2 is not None:
            rgb = self.picam2.capture_array("main")
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            return Frame(image=bgr, ts=time.time())

        if self.capture is None:
            raise RuntimeError("FrameSource is not open")
        ok, frame = self.capture.read()
        if not ok:
            return None
        return Frame(image=frame, ts=time.time())

    def release(self) -> None:
        if self.capture is not None:
            self.capture.release()
            self.capture = None
        if self.picam2 is not None:
            self.picam2.stop()
            self.picam2.close()
            self.picam2 = None

    def __enter__(self) -> "FrameSource":
        return self.open()

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()

    def _open_opencv(self) -> None:
        source_arg: str | int
        if self.source == "camera":
            source_arg = 0
        else:
            path = Path(self.source)
            if not path.exists():
                raise FileNotFoundError(f"Video source not found: {path}")
            source_arg = str(path)

        capture = cv2.VideoCapture(source_arg)
        if not capture.isOpened():
            raise RuntimeError(f"Unable to open video source: {self.source}")

        if self.source == "camera":
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            capture.set(cv2.CAP_PROP_FPS, self.requested_fps)

        fps = capture.get(cv2.CAP_PROP_FPS)
        self._fps = fps if fps and fps > 0 else float(self.requested_fps)
        self.capture = capture

    def _open_picamera2(self) -> None:
        try:
            from picamera2 import Picamera2
        except ImportError as exc:
            raise RuntimeError(
                "Picamera2 backend requested but picamera2 is not installed. "
                "Use --camera-backend opencv or install Picamera2 on Raspberry Pi."
            ) from exc

        picam2 = Picamera2()
        config = picam2.create_video_configuration(
            main={"size": (self.width, self.height), "format": "RGB888"},
            controls={"FrameRate": self.requested_fps},
        )
        picam2.configure(config)
        picam2.start()
        self.picam2 = picam2
        self._fps = float(self.requested_fps)

