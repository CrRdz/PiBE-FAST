"""Optional fall-event video clip recorder."""

from __future__ import annotations

from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Deque

import cv2


class EventClipRecorder:
    """Stores a short MP4 clip around fall events when enabled."""

    def __init__(
        self,
        clips_dir: str | Path,
        fps: float,
        pre_seconds: float = 5.0,
        post_seconds: float = 5.0,
    ) -> None:
        self.clips_dir = Path(clips_dir)
        self.clips_dir.mkdir(parents=True, exist_ok=True)
        self.fps = max(float(fps), 1.0)
        self.pre_seconds = pre_seconds
        self.post_seconds = post_seconds
        self.buffer: Deque[tuple[float, object]] = deque(maxlen=max(1, int(self.fps * pre_seconds)))
        self.writer: cv2.VideoWriter | None = None
        self.record_until: float | None = None
        self.current_path: Path | None = None

    def update(self, frame, ts: float, fall: bool) -> Path | None:
        self.buffer.append((ts, frame.copy()))

        if fall:
            if self.writer is None:
                self._start(frame, ts)
                for _, buffered in list(self.buffer)[:-1]:
                    self._write(buffered)
            self.record_until = ts + self.post_seconds

        if self.writer is not None:
            self._write(frame)
            if self.record_until is not None and ts >= self.record_until:
                finished = self.current_path
                self.close()
                return finished

        return None

    def close(self) -> None:
        if self.writer is not None:
            self.writer.release()
            self.writer = None
        self.record_until = None
        self.current_path = None

    def _start(self, frame, ts: float) -> None:
        height, width = frame.shape[:2]
        stamp = datetime.fromtimestamp(ts, timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.current_path = self.clips_dir / f"fall-{stamp}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(self.current_path), fourcc, self.fps, (width, height))
        if not writer.isOpened():
            raise RuntimeError(f"Unable to open event clip writer: {self.current_path}")
        self.writer = writer

    def _write(self, frame) -> None:
        if self.writer is not None:
            self.writer.write(frame)
