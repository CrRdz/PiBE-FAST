"""Flask MJPEG preview service."""

from __future__ import annotations

import threading
import time
from typing import Any

import cv2
from flask import Flask, Response, jsonify


class PreviewState:
    def __init__(self, jpeg_quality: int = 80) -> None:
        self.jpeg_quality = jpeg_quality
        self.lock = threading.Lock()
        self.jpeg: bytes | None = None
        self.status: dict[str, Any] = {
            "pose": "starting",
            "fall": False,
            "fps": 0.0,
            "quality": 0.0,
        }

    def update(self, frame, status: dict[str, Any]) -> None:
        ok, encoded = cv2.imencode(
            ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]
        )
        if not ok:
            return
        with self.lock:
            self.jpeg = encoded.tobytes()
            self.status = status

    def snapshot(self) -> tuple[bytes | None, dict[str, Any]]:
        with self.lock:
            return self.jpeg, dict(self.status)


def create_app(state: PreviewState, stop_event: threading.Event | None = None) -> Flask:
    app = Flask(__name__)

    @app.get("/")
    def index() -> str:
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
        return Response(
            _mjpeg_stream(state, stop_event),
            mimetype="multipart/x-mixed-replace; boundary=frame",
        )

    @app.get("/api/status")
    def api_status() -> Response:
        _, status = state.snapshot()
        return jsonify(status)

    return app


def _mjpeg_stream(state: PreviewState, stop_event: threading.Event | None):
    while stop_event is None or not stop_event.is_set():
        jpeg, _ = state.snapshot()
        if jpeg is None:
            time.sleep(0.05)
            continue
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
        )
        time.sleep(0.03)

