"""Flask MJPEG preview service."""

from __future__ import annotations

import threading
import time
from typing import Any

import cv2
from flask import Flask, Response, jsonify


# Web 预览共享状态，后台检测线程写入最新 JPEG 和状态，Flask 页面负责读取。
class PreviewState:
    def __init__(self, jpeg_quality: int = 80) -> None:
        self.jpeg_quality = jpeg_quality
        # 检测线程会写，Flask 请求线程会读，所以用锁保护共享状态。
        self.lock = threading.Lock()
        self.jpeg: bytes | None = None
        self.status: dict[str, Any] = {
            "pose": "starting",
            "fall": False,
            "fps": 0.0,
            "quality": 0.0,
        }

    def update(self, frame, status: dict[str, Any]) -> None:
        # Web 端用 MJPEG 预览，所以每次把最新画面压成 JPEG 字节。
        ok, encoded = cv2.imencode(
            ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]
        )
        if not ok:
            return
        with self.lock:
            self.jpeg = encoded.tobytes()
            self.status = status

    def snapshot(self) -> tuple[bytes | None, dict[str, Any]]:
        # 返回一份状态副本，避免调用方改动内部共享字典。
        with self.lock:
            return self.jpeg, dict(self.status)


def create_app(state: PreviewState, stop_event: threading.Event | None = None) -> Flask:
    # Flask 只负责展示，不参与检测计算；检测循环在 main.py 后台线程里跑。
    app = Flask(__name__)

    @app.get("/")
    def index() -> str:
        # 简单预览页：上方显示视频流，下方轮询 JSON 状态。
        return """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Fall Detection Preview</title>
  <style>
    body { margin: 0; font-family: system-ui, sans-serif; background: #101214; color: #f5f7fa; }
    main { max-width: 980px; margin: 0 auto; padding: 20px; }
    img { width: 100%; height: auto; background: #050607; border: 1px solid #30363d; }
    pre { padding: 12px; background: #171b20; border: 1px solid #30363d; overflow: auto; }
  </style>
</head>
<body>
  <main>
    <h1>Fall Detection Preview</h1>
    <img src="/video_feed" alt="live preview">
    <pre id="status">{}</pre>
  </main>
  <script>
    async function refreshStatus() {
      const response = await fetch('/api/status');
      document.getElementById('status').textContent =
        JSON.stringify(await response.json(), null, 2);
    }
    setInterval(refreshStatus, 500);
    refreshStatus();
  </script>
</body>
</html>
"""

    @app.get("/video_feed")
    def video_feed() -> Response:
        # multipart/x-mixed-replace 是浏览器显示 MJPEG 流的常见方式。
        return Response(
            _mjpeg_stream(state, stop_event),
            mimetype="multipart/x-mixed-replace; boundary=frame",
        )

    @app.get("/api/status")
    def api_status() -> Response:
        # 给页面和调试脚本读取当前 pose/fall/fps/quality。
        _, status = state.snapshot()
        return jsonify(status)

    return app


def _mjpeg_stream(state: PreviewState, stop_event: threading.Event | None):
    # 只推送最新帧，不排队历史帧，避免浏览器越看越延迟。
    while stop_event is None or not stop_event.is_set():
        jpeg, _ = state.snapshot()
        if jpeg is None:
            # 检测线程还没产出第一帧时，短暂等待。
            time.sleep(0.05)
            continue
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
        )
        time.sleep(0.03)
