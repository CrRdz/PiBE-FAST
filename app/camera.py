"""Frame sources for local videos, USB cameras, and optional Picamera2."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time

import cv2


# 单帧数据结构，保存图像本身和读取到这一帧时的时间戳。
@dataclass(frozen=True)
class Frame:
    image: object
    ts: float


# 视频源封装，统一本地视频、OpenCV 摄像头和 Picamera2 摄像头的读取方式。
class FrameSource:
    def __init__(
        self,
        source: str,
        width: int = 640,
        height: int = 480,
        fps: int = 15,
        camera_backend: str = "opencv",
        camera_index: int = 0,
    ) -> None:
        self.source = source
        self.width = width
        self.height = height
        self.requested_fps = fps
        self.camera_backend = camera_backend
        self.camera_index = int(camera_index)
        self.capture: cv2.VideoCapture | None = None
        self.picam2 = None
        self._fps = float(fps)

    @property
    def fps(self) -> float:
        # 对视频文件来说这是文件 FPS；对摄像头来说是请求值或驱动返回值。
        return self._fps

    def open(self) -> "FrameSource":
        # Raspberry Pi 官方摄像头可以走 Picamera2；普通 USB 摄像头/视频文件走 OpenCV。
        if self.source == "camera" and self.camera_backend == "picamera2":
            self._open_picamera2()
        else:
            self._open_opencv()
        return self

    def read(self) -> Frame | None:
        # Picamera2 输出 RGB；为了让后续主流程统一处理，这里转回 OpenCV 常用的 BGR。
        if self.picam2 is not None:
            rgb = self.picam2.capture_array("main")
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            return Frame(image=bgr, ts=time.time())

        # OpenCV 的 VideoCapture 同时支持本地视频文件和摄像头。
        if self.capture is None:
            raise RuntimeError("FrameSource is not open")
        ok, frame = self.capture.read()
        # 视频读到末尾或摄像头读取失败时返回 None，主循环会据此退出。
        if not ok:
            return None
        return Frame(image=frame, ts=time.time())

    def release(self) -> None:
        # 释放摄像头/视频文件句柄，否则下次运行可能占用设备。
        if self.capture is not None:
            self.capture.release()
            self.capture = None
        if self.picam2 is not None:
            self.picam2.stop()
            self.picam2.close()
            self.picam2 = None

    def switch_camera(self, camera_index: int) -> bool:
        """Switch an open OpenCV camera while keeping the frame source alive.

        Returns ``True`` when a different camera was opened. If opening the new
        device fails, the previous device is reopened before the error is raised.
        Video files and Picamera2 intentionally remain single-source.
        """

        target_index = int(camera_index)
        if self.source != "camera" or self.camera_backend != "opencv":
            return False
        if target_index == self.camera_index and self.capture is not None:
            return False

        previous_index = self.camera_index
        if self.capture is not None:
            self.capture.release()
            self.capture = None
        self.camera_index = target_index
        try:
            self._open_opencv()
        except Exception as switch_error:
            self.camera_index = previous_index
            try:
                self._open_opencv()
            except Exception as fallback_error:
                raise RuntimeError(
                    f"Unable to switch from camera {previous_index} to "
                    f"{target_index}; reopening camera {previous_index} also failed"
                ) from fallback_error
            raise RuntimeError(
                f"Unable to switch from camera {previous_index} to {target_index}; "
                f"camera {previous_index} was restored"
            ) from switch_error
        return True

    def __enter__(self) -> "FrameSource":
        return self.open()

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()

    def _open_opencv(self) -> None:
        source_arg: str | int
        if self.source == "camera":
            # OpenCV 里 0 通常表示默认摄像头，可用 --camera-index 选择其他设备。
            source_arg = self.camera_index
        else:
            # 本地视频要先确认文件存在，避免 OpenCV 给出不清楚的打开失败。
            path = Path(self.source)
            if not path.exists():
                raise FileNotFoundError(f"Video source not found: {path}")
            source_arg = str(path)

        capture = cv2.VideoCapture(source_arg)
        if not capture.isOpened():
            capture.release()
            if self.source == "camera":
                raise RuntimeError(
                    f"Unable to open camera index {self.camera_index} with OpenCV"
                )
            raise RuntimeError(f"Unable to open video source: {self.source}")

        if self.source == "camera":
            # 摄像头参数只是“请求”，实际是否生效取决于摄像头和驱动。
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            capture.set(cv2.CAP_PROP_FPS, self.requested_fps)

        fps = capture.get(cv2.CAP_PROP_FPS)
        # 有些摄像头驱动返回 0 FPS，这时退回到用户请求的 fps。
        self._fps = fps if fps and fps > 0 else float(self.requested_fps)
        self.capture = capture

    def _open_picamera2(self) -> None:
        # Picamera2 只在 Raspberry Pi 上常见，所以做成可选依赖。
        try:
            from picamera2 import Picamera2
        except ImportError as exc:
            raise RuntimeError(
                "Picamera2 backend requested but picamera2 is not installed. "
                "Use --camera-backend opencv or install Picamera2 on Raspberry Pi."
            ) from exc

        picam2 = Picamera2()
        # 主流里配置 RGB888，方便后面直接给 MoveNet 使用。
        config = picam2.create_video_configuration(
            main={"size": (self.width, self.height), "format": "RGB888"},
            controls={"FrameRate": self.requested_fps},
        )
        picam2.configure(config)
        picam2.start()
        self.picam2 = picam2
        self._fps = float(self.requested_fps)
