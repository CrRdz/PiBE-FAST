"""Flask MJPEG preview service for the Pi BE-FAST screening demo."""

from __future__ import annotations

import threading
import time
from typing import Any

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template, request

from app.befast import BefastSession


class PreviewState:
    """Thread-safe bridge between the inference worker and Flask requests."""

    def __init__(self, jpeg_quality: int = 80) -> None:
        self.jpeg_quality = jpeg_quality
        self.lock = threading.Lock()
        self.jpeg: bytes | None = None
        self.has_live_frame = False
        self.camera_mode = "host"
        self.host_capture_source = "unknown"
        self.host_camera_ready = False
        self.client_frame: np.ndarray | None = None
        self.client_frame_ts: float | None = None
        self.client_frame_seq = 0
        self.client_frame_timeout = 2.0
        self.placeholder_jpeg = _make_placeholder_jpeg(jpeg_quality)
        self.status: dict[str, Any] = {
            "pose": "starting",
            "fps": 0.0,
            "quality": 0.0,
            "runtime": {
                "phase": "starting",
                "message": "正在启动摄像头与姿态模型……",
                "camera_ready": False,
                "model_ready": False,
                "face_model_ready": False,
                "has_live_frame": False,
                "error": None,
                "operation_mode": "standby",
                "inference_mode": "starting",
                "capture_origin": "server_host",
                "capture_source": "unknown",
                "client_camera_used": False,
                "camera_mode": "host",
                "client_camera_connected": False,
            },
            "befast": {
                "decision": "standby",
                "mode": "standby",
                "stage": "idle",
                "guidance": {"ready": False, "reason": "waiting_for_camera"},
            },
            "monitoring": {
                "enabled": True,
                "mode": "standby",
                "inference_mode": "starting",
                "standby_pose_fps": 2.0,
                "medical_role": "trigger_only_not_stroke_diagnosis",
            },
        }

    def update(self, frame, status: dict[str, Any]) -> None:
        ok, encoded = cv2.imencode(
            ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]
        )
        if not ok:
            return
        with self.lock:
            self.jpeg = encoded.tobytes()
            self.has_live_frame = True
            runtime = dict(self.status.get("runtime", {}))
            runtime["has_live_frame"] = True
            next_status = dict(status)
            next_status["runtime"] = runtime
            self.status = next_status

    def set_runtime(self, phase: str, message: str, **fields: Any) -> None:
        with self.lock:
            incoming_source = fields.get("capture_source")
            if (
                isinstance(incoming_source, str)
                and incoming_source not in {"browser_client", "unknown"}
            ):
                self.host_capture_source = incoming_source
            if "camera_ready" in fields:
                self.host_camera_ready = bool(fields["camera_ready"])
            runtime = dict(self.status.get("runtime", {}))
            runtime.update(fields)
            self._apply_camera_mode_locked(runtime)
            runtime.update(
                {
                    "phase": phase,
                    "message": message,
                }
            )
            self.status["runtime"] = runtime

    def set_camera_mode(self, mode: str) -> dict[str, Any]:
        if mode not in {"host", "client"}:
            raise ValueError("camera source must be 'host' or 'client'")
        with self.lock:
            if self.camera_mode != mode:
                self.camera_mode = mode
                self.jpeg = None
                self.has_live_frame = False
            runtime = dict(self.status.get("runtime", {}))
            self._apply_camera_mode_locked(runtime)
            self.status["runtime"] = runtime
            return dict(runtime)

    def submit_client_jpeg(self, payload: bytes, now: float | None = None) -> dict[str, Any]:
        if not payload:
            raise ValueError("empty client camera frame")
        if len(payload) > 2_000_000:
            raise ValueError("client camera frame exceeds 2 MB")
        encoded = np.frombuffer(payload, dtype=np.uint8)
        frame = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if frame is None or frame.ndim != 3:
            raise ValueError("unable to decode client camera JPEG")
        height, width = frame.shape[:2]
        if width < 160 or height < 120 or width > 1920 or height > 1080:
            raise ValueError("client camera frame dimensions must be 160x120..1920x1080")
        ts = time.time() if now is None else float(now)
        with self.lock:
            if self.camera_mode != "client":
                raise ValueError("select the client camera before uploading frames")
            self.client_frame = frame
            self.client_frame_ts = ts
            self.client_frame_seq += 1
            runtime = dict(self.status.get("runtime", {}))
            self._apply_camera_mode_locked(runtime, now=ts)
            self.status["runtime"] = runtime
            return {
                "sequence": self.client_frame_seq,
                "width": int(width),
                "height": int(height),
            }

    def select_input_frame(
        self,
        host_image: Any,
        host_ts: float,
        after_client_sequence: int,
        now: float | None = None,
    ) -> tuple[Any, float, int] | None:
        """Return the selected source frame, or None while waiting for a new client frame."""

        ts = time.time() if now is None else float(now)
        with self.lock:
            if self.camera_mode == "host":
                return host_image, float(host_ts), after_client_sequence
            runtime = dict(self.status.get("runtime", {}))
            self._apply_camera_mode_locked(runtime, now=ts)
            self.status["runtime"] = runtime
            if (
                self.client_frame is None
                or self.client_frame_ts is None
                or self.client_frame_seq <= after_client_sequence
                or ts - self.client_frame_ts > self.client_frame_timeout
            ):
                return None
            return (
                self.client_frame.copy(),
                float(self.client_frame_ts),
                self.client_frame_seq,
            )

    def _apply_camera_mode_locked(
        self, runtime: dict[str, Any], now: float | None = None
    ) -> None:
        ts = time.time() if now is None else float(now)
        runtime["camera_mode"] = self.camera_mode
        if self.camera_mode == "client":
            connected = bool(
                self.client_frame_ts is not None
                and ts - self.client_frame_ts <= self.client_frame_timeout
            )
            runtime.update(
                {
                    "capture_origin": "browser_client",
                    "capture_source": "browser_client",
                    "client_camera_used": True,
                    "client_camera_connected": connected,
                    "camera_ready": connected,
                    "has_live_frame": connected and self.has_live_frame,
                }
            )
        else:
            runtime.update(
                {
                    "capture_origin": "server_host",
                    "capture_source": self.host_capture_source,
                    "client_camera_used": False,
                    "client_camera_connected": False,
                    "camera_ready": self.host_camera_ready,
                    "has_live_frame": self.has_live_frame,
                }
            )

    def snapshot(self) -> tuple[bytes, dict[str, Any]]:
        with self.lock:
            jpeg = self.jpeg if self.jpeg is not None else self.placeholder_jpeg
            status = dict(self.status)
            status["runtime"] = dict(self.status.get("runtime", {}))
            return jpeg, status


def _make_placeholder_jpeg(jpeg_quality: int) -> bytes:
    frame = np.full((720, 1280, 3), (5, 8, 12), dtype=np.uint8)
    cv2.putText(
        frame,
        "Pi BE-FAST Screening Demo",
        (350, 335),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.25,
        (228, 237, 244),
        2,
        lineType=cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        "Waiting for camera...",
        (455, 390),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (126, 151, 172),
        2,
        lineType=cv2.LINE_AA,
    )
    ok, encoded = cv2.imencode(
        ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
    )
    if not ok:
        raise RuntimeError("Unable to create camera placeholder image")
    return encoded.tobytes()


def create_app(
    state: PreviewState,
    stop_event: threading.Event | None = None,
    befast_session: BefastSession | None = None,
) -> Flask:
    app = Flask(__name__)

    @app.after_request
    def configure_client_sensor_access(response: Response) -> Response:
        # Client-camera mode is opt-in and same-origin; microphone access remains disabled.
        response.headers["Permissions-Policy"] = "camera=(self), microphone=()"
        return response

    @app.get("/")
    def index() -> str:
        return render_template("index.html")

    @app.get("/video_feed")
    def video_feed() -> Response:
        return Response(
            _mjpeg_stream(state, stop_event),
            mimetype="multipart/x-mixed-replace; boundary=frame",
        )

    @app.get("/api/status")
    def api_status() -> Response:
        _, status = state.snapshot()
        if befast_session is not None:
            status["befast"] = befast_session.snapshot()
        return jsonify(status)

    @app.post("/api/camera/source")
    def api_camera_source() -> Response:
        payload = request.get_json(silent=True) or {}
        requested_source = str(payload.get("source", ""))
        if befast_session is not None:
            screen = befast_session.snapshot()
            _, current_status = state.snapshot()
            current_source = current_status.get("runtime", {}).get(
                "camera_mode", "host"
            )
            if screen.get("mode") == "screening" and requested_source != current_source:
                return jsonify(
                    {"error": "camera source can only change while screening is idle"}
                ), 409
        try:
            runtime = state.set_camera_mode(requested_source)
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        return jsonify({"runtime": runtime})

    @app.post("/api/camera/frame")
    def api_camera_frame() -> Response:
        if request.mimetype != "image/jpeg":
            return jsonify({"error": "Content-Type must be image/jpeg"}), 415
        try:
            frame = state.submit_client_jpeg(request.get_data(cache=False))
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        return jsonify(frame)

    @app.post("/api/monitoring/trigger")
    def api_monitoring_trigger() -> Response:
        if befast_session is None:
            return jsonify({"error": "BE-FAST session is unavailable"}), 503
        payload = request.get_json(silent=True) or {}
        source = str(payload.get("source", "user")).strip().lower()
        allowed_sources = {"user", "caregiver", "scheduled", "passive", "api"}
        if source not in allowed_sources:
            return jsonify({"error": "unsupported trigger source"}), 400
        reason = str(payload.get("reason", "screen_requested")).strip()
        if not reason or len(reason) > 120:
            return jsonify({"error": "reason must contain 1..120 characters"}), 400
        befast_session.start_screening(source=source, reason=reason)
        return jsonify({"befast": befast_session.snapshot()})

    @app.post("/api/monitoring/standby")
    def api_monitoring_standby() -> Response:
        if befast_session is None:
            return jsonify({"error": "BE-FAST session is unavailable"}), 503
        befast_session.reset()
        return jsonify({"befast": befast_session.snapshot()})

    @app.post("/api/befast/stage")
    def api_befast_stage() -> Response:
        if befast_session is None:
            return jsonify({"error": "BE-FAST session is unavailable"}), 503
        payload = request.get_json(silent=True) or {}
        try:
            befast_session.start_stage(str(payload.get("stage", "")))
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        return jsonify({"befast": befast_session.snapshot()})

    @app.post("/api/befast/skip")
    def api_befast_skip() -> Response:
        if befast_session is None:
            return jsonify({"error": "BE-FAST session is unavailable"}), 503
        try:
            skipped = befast_session.skip_current_stage()
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 409
        return jsonify(
            {"skipped": skipped, "befast": befast_session.snapshot()}
        )

    @app.post("/api/befast/manual")
    def api_befast_manual() -> Response:
        if befast_session is None:
            return jsonify({"error": "BE-FAST session is unavailable"}), 503
        payload = request.get_json(silent=True) or {}
        keys = BefastSession.MANUAL_KEYS + ("new_or_sudden",)
        invalid = [key for key in keys if not isinstance(payload.get(key), bool)]
        if invalid:
            return jsonify({"error": f"boolean fields required: {', '.join(invalid)}"}), 400
        observations = {key: payload[key] for key in BefastSession.MANUAL_KEYS}
        try:
            befast_session.submit_manual(
                observations,
                new_or_sudden=payload["new_or_sudden"],
                onset_time=payload.get("onset_time"),
            )
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        return jsonify({"befast": befast_session.snapshot()})

    @app.post("/api/befast/reset")
    def api_befast_reset() -> Response:
        if befast_session is None:
            return jsonify({"error": "BE-FAST session is unavailable"}), 503
        befast_session.reset()
        return jsonify({"befast": befast_session.snapshot()})

    return app


def _mjpeg_stream(state: PreviewState, stop_event: threading.Event | None):
    while True:
        jpeg, _ = state.snapshot()
        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
        time.sleep(0.5 if stop_event is not None and stop_event.is_set() else 0.03)
