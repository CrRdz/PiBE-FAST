"""Optional fall-event video clip recorder."""

from __future__ import annotations

from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Deque

import cv2


# 摔倒事件片段录像器，只在触发 fall 时保存前后几秒的视频。
class EventClipRecorder:
    """Stores a short MP4 clip around fall events when enabled."""

    def __init__(
        self,
        clips_dir: str | Path,
        fps: float,
        pre_seconds: float = 5.0,
        post_seconds: float = 5.0,
    ) -> None:
        # buffer 常驻保存最近 pre_seconds 秒的帧，用于 fall 触发后补上“事发前”片段。
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
        # 每帧都进环形缓冲，但只有 fall=True 时才真正打开 VideoWriter 写文件。
        self.buffer.append((ts, frame.copy()))

        if fall:
            if self.writer is None:
                self._start(frame, ts)
                # 先写入触发前缓存帧；[:-1] 是为了避免当前帧稍后重复写一次。
                for _, buffered in list(self.buffer)[:-1]:
                    self._write(buffered)
            self.record_until = ts + self.post_seconds

        if self.writer is not None:
            self._write(frame)
            # fall 消失后仍继续写到 record_until，这样能保留事发后几秒。
            if self.record_until is not None and ts >= self.record_until:
                finished = self.current_path
                self.close()
                return finished

        return None

    def close(self) -> None:
        # 关闭当前事件片段；下一次 fall 会重新创建新的文件。
        if self.writer is not None:
            self.writer.release()
            self.writer = None
        self.record_until = None
        self.current_path = None

    def _start(self, frame, ts: float) -> None:
        # 以 fall 触发时间命名文件，方便把视频和 JSONL 中的时间戳对上。
        height, width = frame.shape[:2]
        stamp = datetime.fromtimestamp(ts, timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.current_path = self.clips_dir / f"fall-{stamp}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(self.current_path), fourcc, self.fps, (width, height))
        if not writer.isOpened():
            raise RuntimeError(f"Unable to open event clip writer: {self.current_path}")
        self.writer = writer

    def _write(self, frame) -> None:
        # 对外统一走 update()，这里保持最小写入逻辑。
        if self.writer is not None:
            self.writer.write(frame)
