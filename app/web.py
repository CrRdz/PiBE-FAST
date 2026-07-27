"""Flask MJPEG preview service for the Pi BE-FAST screening demo."""

from __future__ import annotations

import threading
import time
from typing import Any, Mapping

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template, request, send_file

from app.befast import BefastSession
from app.history import AbnormalHistoryStore, BEFAST_COMPONENTS
from app.passive_speech import EVIDENCE_VERSION, PassiveSpeechMonitor
from app.speech_audio import SpeechCaptureService


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
    history_store: AbnormalHistoryStore | None = None,
    speech_service: SpeechCaptureService | None = None,
    passive_speech_monitor: PassiveSpeechMonitor | None = None,
) -> Flask:
    app = Flask(__name__)
    speech_coordination_lock = threading.Lock()
    resume_passive_after_guided = False

    def remember_guided_passive_pause(should_resume: bool) -> None:
        nonlocal resume_passive_after_guided
        with speech_coordination_lock:
            resume_passive_after_guided = bool(should_resume)

    def restore_passive_after_guided() -> None:
        nonlocal resume_passive_after_guided
        with speech_coordination_lock:
            should_resume = resume_passive_after_guided
            resume_passive_after_guided = False
        if should_resume and passive_speech_monitor is not None:
            passive_speech_monitor.resume()

    def persist_current_positive(
        screen: Mapping[str, Any],
        audio_path: str | None = None,
    ) -> None:
        """Attach current frame and optional microphone audio to a positive report."""

        if history_store is None:
            return
        report = screen.get("current_report")
        if not isinstance(report, Mapping):
            return
        jpeg, status = state.snapshot()
        runtime = status.get("runtime", {})
        has_live_frame = bool(
            isinstance(runtime, Mapping) and runtime.get("has_live_frame")
        )
        captured_at = status.get("ts", time.time())
        try:
            history_store.save_positive_report(
                report,
                jpeg if has_live_frame else None,
                float(captured_at),
                audio_path=audio_path,
            )
        except Exception:
            app.logger.exception("Unable to save abnormal BE-FAST history")

    @app.after_request
    def configure_client_sensor_access(response: Response) -> Response:
        # Speech uses the Pi-attached ALSA microphone, not the browser microphone.
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
            screen = befast_session.snapshot()
            persist_current_positive(screen)
            status["befast"] = screen
        if speech_service is not None:
            status["speech"] = speech_service.snapshot()
        status["passive_speech"] = (
            passive_speech_monitor.snapshot()
            if passive_speech_monitor is not None
            else {
                "enabled": False,
                "state": "unavailable",
                "assessment": "unavailable",
                "medical_role": "change_detection_trigger_only",
                "clinical_validation": False,
                "evidence_version": EVIDENCE_VERSION,
            }
        )
        return jsonify(status)

    @app.get("/api/history")
    def api_history() -> Response:
        if history_store is None:
            return jsonify({"error": "abnormal history storage is unavailable"}), 503
        raw_components = str(request.args.get("component", "")).strip()
        components = tuple(
            value.strip().upper()
            for value in raw_components.split(",")
            if value.strip()
        )
        invalid_components = [
            value for value in components if value not in BEFAST_COMPONENTS
        ]
        if invalid_components:
            return jsonify(
                {
                    "error": (
                        "component must contain only B, E, F, A, or S"
                    )
                }
            ), 400
        try:
            limit = int(request.args.get("limit", 50))
            offset = int(request.args.get("offset", 0))
        except (TypeError, ValueError):
            return jsonify({"error": "limit and offset must be integers"}), 400
        try:
            sudden = _optional_query_bool(request.args.get("new_or_sudden"))
            affected_side = _optional_query_choice(
                request.args.get("affected_side"), {"left", "right"}
            )
            records, total = history_store.list_records(
                components=components,
                reason=_optional_query_text(request.args.get("reason")),
                affected_side=affected_side,
                new_or_sudden=sudden,
                limit=limit,
                offset=offset,
            )
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        return jsonify(
            {
                "items": records,
                "total": total,
                "limit": limit,
                "offset": offset,
                "filters": {
                    "component": list(components),
                    "reason": _optional_query_text(request.args.get("reason")),
                    "affected_side": affected_side,
                    "new_or_sudden": sudden,
                },
            }
        )

    @app.get("/api/history/<int:record_id>")
    def api_history_record(record_id: int) -> Response:
        if history_store is None:
            return jsonify({"error": "abnormal history storage is unavailable"}), 503
        record = history_store.get_record(record_id)
        if record is None:
            return jsonify({"error": "history record not found"}), 404
        return jsonify(record)

    @app.get("/api/history/<int:record_id>/frame")
    def api_history_frame(record_id: int) -> Response:
        if history_store is None:
            return jsonify({"error": "abnormal history storage is unavailable"}), 503
        path = history_store.get_frame_path(record_id)
        if path is None:
            return jsonify({"error": "history frame not found"}), 404
        return send_file(
            path.resolve(),
            mimetype="image/jpeg",
            conditional=True,
            max_age=0,
        )

    @app.get("/api/history/<int:record_id>/audio")
    def api_history_audio(record_id: int) -> Response:
        if history_store is None:
            return jsonify({"error": "abnormal history storage is unavailable"}), 503
        path = history_store.get_audio_path(record_id)
        if path is None:
            return jsonify({"error": "history audio not found"}), 404
        return send_file(
            path.resolve(),
            mimetype="audio/wav",
            conditional=True,
            max_age=0,
        )

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
            camera_switch_locked = (
                screen.get("mode") == "screening"
                and screen.get("stage") != "idle"
            )
            if camera_switch_locked and requested_source != current_source:
                return jsonify(
                    {
                        "error": (
                            "camera source can only change from standby "
                            "or the component menu"
                        )
                    }
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
        if speech_service is not None:
            speech_service.cancel()
        restore_passive_after_guided()
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

    @app.post("/api/befast/component")
    def api_befast_component() -> Response:
        if befast_session is None:
            return jsonify({"error": "BE-FAST session is unavailable"}), 503
        payload = request.get_json(silent=True) or {}
        component = str(payload.get("component", "")).strip().upper()
        try:
            if speech_service is not None and component in BEFAST_COMPONENTS:
                speech_service.cancel()
            befast_session.prepare_component(component)
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        return jsonify({"befast": befast_session.snapshot()})

    @app.post("/api/befast/menu")
    def api_befast_menu() -> Response:
        if befast_session is None:
            return jsonify({"error": "BE-FAST session is unavailable"}), 503
        if speech_service is not None:
            speech_service.cancel()
        befast_session.open_component_menu()
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

    @app.post("/api/befast/manual-item")
    def api_befast_manual_item() -> Response:
        if befast_session is None:
            return jsonify({"error": "BE-FAST session is unavailable"}), 503
        payload = request.get_json(silent=True) or {}
        invalid = [
            key
            for key in ("problem", "new_or_sudden")
            if not isinstance(payload.get(key), bool)
        ]
        if invalid:
            return jsonify(
                {"error": f"boolean fields required: {', '.join(invalid)}"}
            ), 400
        try:
            befast_session.submit_component_observation(
                str(payload.get("component", "")),
                problem=payload["problem"],
                new_or_sudden=payload["new_or_sudden"],
                onset_time=payload.get("onset_time"),
                viewing_distance_cm=payload.get("viewing_distance_cm"),
                screen_width_cm=payload.get("screen_width_cm"),
                achieved_target_visual_angle_degrees=payload.get(
                    "achieved_target_visual_angle_degrees"
                ),
            )
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        screen = befast_session.snapshot()
        persist_current_positive(screen)
        return jsonify({"befast": screen})

    @app.get("/api/speech/status")
    def api_speech_status() -> Response:
        if speech_service is None:
            return jsonify({"error": "speech capture service is unavailable"}), 503
        return jsonify({"speech": speech_service.snapshot()})

    @app.get("/api/speech/passive/status")
    def api_passive_speech_status() -> Response:
        if passive_speech_monitor is None:
            return jsonify({"error": "passive speech monitoring is unavailable"}), 503
        return jsonify({"passive_speech": passive_speech_monitor.snapshot()})

    @app.post("/api/speech/passive/pause")
    def api_passive_speech_pause() -> Response:
        if passive_speech_monitor is None:
            return jsonify({"error": "passive speech monitoring is unavailable"}), 503
        released = passive_speech_monitor.pause("manual_pause")
        if not released:
            passive_speech_monitor.resume()
            return jsonify({"error": "microphone capture did not stop in time"}), 409
        return jsonify({"passive_speech": passive_speech_monitor.snapshot()})

    @app.post("/api/speech/passive/resume")
    def api_passive_speech_resume() -> Response:
        if passive_speech_monitor is None:
            return jsonify({"error": "passive speech monitoring is unavailable"}), 503
        return jsonify({"passive_speech": passive_speech_monitor.resume()})

    @app.post("/api/speech/passive/reset-baseline")
    def api_passive_speech_reset_baseline() -> Response:
        if passive_speech_monitor is None:
            return jsonify({"error": "passive speech monitoring is unavailable"}), 503
        return jsonify({"passive_speech": passive_speech_monitor.reset_baseline()})

    @app.post("/api/speech/start")
    def api_speech_start() -> Response:
        if befast_session is None or speech_service is None:
            return jsonify({"error": "speech screening is unavailable"}), 503
        payload = request.get_json(silent=True) or {}
        sudden = payload.get("new_or_sudden")
        if not isinstance(sudden, bool):
            return jsonify({"error": "new_or_sudden must be a boolean"}), 400
        language = str(payload.get("language", "zh")).strip().lower()
        if language not in {"zh", "en"}:
            return jsonify({"error": "language must be 'zh' or 'en'"}), 400
        passive_paused = False
        should_resume_passive = False
        try:
            if passive_speech_monitor is not None:
                passive_before = passive_speech_monitor.snapshot()
                passive_paused = bool(passive_before.get("paused"))
                if not passive_paused:
                    passive_paused = passive_speech_monitor.pause(
                        "guided_speech_check"
                    )
                    should_resume_passive = passive_paused
                    if not passive_paused:
                        passive_speech_monitor.resume()
                        return jsonify(
                            {
                                "error": (
                                    "passive microphone capture did not stop in time"
                                )
                            }
                        ), 409
            speech_service.start(
                language=language,
                new_or_sudden=sudden,
                onset_time=payload.get("onset_time"),
            )
            befast_session.start_speech_recording()
            remember_guided_passive_pause(should_resume_passive)
        except ValueError as exc:
            speech_service.cancel()
            if should_resume_passive and passive_speech_monitor is not None:
                passive_speech_monitor.resume()
            return jsonify({"error": str(exc)}), 409
        except Exception:
            speech_service.cancel()
            if should_resume_passive and passive_speech_monitor is not None:
                passive_speech_monitor.resume()
            raise
        return jsonify(
            {
                "speech": speech_service.snapshot(),
                "befast": befast_session.snapshot(),
            }
        )

    @app.post("/api/speech/complete")
    def api_speech_complete() -> Response:
        if befast_session is None or speech_service is None:
            return jsonify({"error": "speech screening is unavailable"}), 503
        audio_path = None
        consumed = False
        try:
            result, audio_path, sudden, onset_time = speech_service.consume_result()
            consumed = True
            befast_session.submit_speech_result(
                result,
                new_or_sudden=sudden,
                onset_time=onset_time,
            )
            screen = befast_session.snapshot()
            persist_current_positive(
                screen,
                str(audio_path) if audio_path is not None else None,
            )
            return jsonify(
                {
                    "speech": speech_service.snapshot(),
                    "befast": screen,
                }
            )
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 409
        finally:
            SpeechCaptureService.discard_consumed_audio(audio_path)
            if consumed:
                restore_passive_after_guided()

    @app.post("/api/speech/cancel")
    def api_speech_cancel() -> Response:
        if befast_session is None or speech_service is None:
            return jsonify({"error": "speech screening is unavailable"}), 503
        speech_service.cancel()
        restore_passive_after_guided()
        screen = befast_session.snapshot()
        if (
            screen.get("active_component") == "S"
            and screen.get("stage") in {"speech_ready", "speech_recording"}
        ):
            befast_session.prepare_component("S")
        return jsonify(
            {
                "speech": speech_service.snapshot(),
                "befast": befast_session.snapshot(),
            }
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


def _optional_query_text(value: str | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _optional_query_bool(value: str | None) -> bool | None:
    text = _optional_query_text(value)
    if text is None:
        return None
    normalized = text.lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise ValueError("new_or_sudden must be true or false")


def _optional_query_choice(
    value: str | None, allowed: set[str]
) -> str | None:
    text = _optional_query_text(value)
    if text is None:
        return None
    normalized = text.lower()
    if normalized not in allowed:
        choices = ", ".join(sorted(allowed))
        raise ValueError(f"value must be one of: {choices}")
    return normalized
