"""CLI entry point for the PiBE-FAST screening service."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys
import threading
import time
from typing import Mapping

import cv2

from app.befast import BefastConfig, BefastSession
from app.camera import Frame, FrameSource
from app.config import PoseClassifierConfig, RuntimeConfig
from app.drawing import draw_befast_overlay
from app.event_recorder import EventClipRecorder
from app.face_landmarker import FaceObservation, MediaPipeFaceLandmarker
from app.history import AbnormalHistoryStore
from app.keypoint_logger import JsonlKeypointLogger
from app.monitoring import PassiveMonitor, PassiveMonitoringConfig
from app.movenet import MoveNet
from app.passive_speech import PassiveSpeechConfig, PassiveSpeechMonitor
from app.pose_classifier import PoseClassification, PoseClassifier
from app.speech_audio import (
    SpeechAudioConfig,
    SpeechCaptureService,
    WhisperCppRecognizer,
    default_microphone_capture,
)
from app.web import PreviewState, create_app


LOGGER = logging.getLogger("pibe_fast")


# 解析命令行参数，决定输入源、模型路径、Web 端口和是否保存事件片段。
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "PiBE-FAST: A Raspberry Pi-Based Multimodal System for Early "
            "Acute Stroke Screening"
        )
    )
    # source 可以是 "camera"，也可以是本地视频路径；第一阶段建议先用视频文件调规则。
    parser.add_argument(
        "--source",
        default="camera",
        help=(
            "'camera' (default: camera attached to the machine running this service) "
            "or a local video file path"
        ),
    )
    # 模型文件不放进仓库，运行前需要放到 models/ 或通过 --model 指定其他路径。
    parser.add_argument("--model", default="models/movenet_lightning.tflite")
    parser.add_argument(
        "--face-model",
        default="models/face_landmarker.task",
        help="MediaPipe Face Landmarker task model used by automated E/F checks",
    )
    parser.add_argument(
        "--face-fps",
        type=float,
        default=BefastConfig.face_inference_fps,
        help="Face inference rate during E/F stages; 5 FPS is the Pi-oriented default",
    )
    parser.add_argument(
        "--disable-face",
        action="store_true",
        help="Disable MediaPipe E/F inference (those stages return insufficient)",
    )
    # opencv 适合普通 USB 摄像头和本地视频；picamera2 适合 Raspberry Pi 官方摄像头。
    parser.add_argument("--camera-backend", choices=["opencv", "picamera2"], default="opencv")
    parser.add_argument(
        "--camera-index",
        type=int,
        default=0,
        help=(
            "host camera used for standby and A/B pose checks (default: 0)"
        ),
    )
    parser.add_argument(
        "--face-camera-index",
        type=int,
        default=None,
        help=(
            "host camera used for E/F face checks; by default this is index 1 "
            "on macOS and the same camera as --camera-index elsewhere"
        ),
    )
    parser.add_argument("--width", type=int, default=RuntimeConfig.frame_width)
    parser.add_argument("--height", type=int, default=RuntimeConfig.frame_height)
    parser.add_argument("--fps", type=int, default=RuntimeConfig.frame_fps)
    parser.add_argument(
        "--standby-pose-fps",
        type=float,
        default=RuntimeConfig.standby_pose_fps,
        help="Low-rate MoveNet cadence in standby; passive output only triggers a guided screen",
    )
    parser.add_argument(
        "--disable-passive-monitor",
        action="store_true",
        help="Disable low-rate fall monitoring while leaving manual screening available",
    )
    parser.add_argument(
        "--scheduled-screen-interval-hours",
        type=float,
        default=RuntimeConfig.scheduled_screen_interval_hours,
        help="Open a guided screen on this interval; 0 disables scheduled reminders",
    )

    # Web 预览默认监听 0.0.0.0，树莓派上可通过 SSH 端口转发到本机浏览器。
    parser.add_argument("--web-host", default="0.0.0.0")
    parser.add_argument("--web-port", type=int, default=8080)
    parser.add_argument(
        "--web-cert",
        help="TLS certificate PEM for HTTPS (required for phone browser camera access)",
    )
    parser.add_argument(
        "--web-key",
        help="TLS private-key PEM used with --web-cert",
    )
    parser.add_argument(
        "--debug-overlay",
        action="store_true",
        help="Draw the large diagnostic text box inside the video frame",
    )

    parser.add_argument("--no-web", action="store_true", help="Process frames without starting Flask")
    # 关闭 JSONL 可以减少磁盘写入；调试姿态规则时建议保持开启。
    parser.add_argument("--no-keypoint-log", action="store_true", help="Disable JSONL keypoint logging")
    parser.add_argument("--log-dir", default="data/keypoints")
    parser.add_argument(
        "--history-dir",
        default="data/history",
        help="Store positive report metadata plus captured JPEG/WAV evidence here",
    )
    parser.add_argument(
        "--speech-device",
        default="default",
        help=(
            "microphone used for S: ALSA name on Linux (default/plughw:1,0) "
            "or AVFoundation name/index on macOS"
        ),
    )
    parser.add_argument(
        "--speech-work-dir",
        default="data/speech",
        help="Temporary local WAV directory for microphone-based S checks",
    )
    parser.add_argument(
        "--speech-capture-seconds",
        type=float,
        default=SpeechAudioConfig.capture_seconds,
        help="Fixed microphone capture duration for one guided S check",
    )
    parser.add_argument(
        "--speech-model",
        default="models/ggml-base.bin",
        help="Local whisper.cpp GGML/GGUF model used to transcribe S",
    )
    parser.add_argument(
        "--whisper-cli",
        default="whisper-cli",
        help="Path or executable name for the local whisper.cpp CLI",
    )
    parser.add_argument(
        "--disable-speech",
        action="store_true",
        help="Disable Pi microphone capture and offline speech recognition",
    )
    parser.add_argument(
        "--disable-passive-speech",
        action="store_true",
        help="Disable long-running local natural-speech change monitoring",
    )
    parser.add_argument(
        "--passive-speech-window-seconds",
        type=float,
        default=PassiveSpeechConfig.window_seconds,
        help="Duration of each local passive speech-analysis window (minimum 3)",
    )
    parser.add_argument(
        "--passive-speech-interval-seconds",
        type=float,
        default=PassiveSpeechConfig.interval_seconds,
        help="Gap between passive speech-analysis windows",
    )
    parser.add_argument(
        "--passive-speech-baseline-windows",
        type=int,
        default=PassiveSpeechConfig.baseline_windows,
        help="Valid natural-speech windows required for the personal baseline",
    )
    # 默认不持续保存视频；只有显式开启后，才保存紧急筛查事件的短片段。
    parser.add_argument(
        "--save-event-clips",
        action="store_true",
        help="Save MP4 clips around newly triggered BE-FAST emergency alerts",
    )
    parser.add_argument("--clips-dir", default="data/clips")
    parser.add_argument("--max-frames", type=int, default=0, help="Stop after N frames, mainly for smoke tests")
    # TFLite 推理线程数，树莓派上可根据实时性和发热情况调节。
    parser.add_argument("--num-threads", type=int, default=2)
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


FACE_CAMERA_STAGES = frozenset(
    {"retry_eyes", "eyes", "ready_face", "retry_face", "face"}
)


def _resolved_face_camera_index(args: argparse.Namespace) -> int:
    """Resolve the E/F camera without forcing a second camera on Raspberry Pi."""

    configured = getattr(args, "face_camera_index", None)
    if configured is not None:
        return int(configured)
    if (
        sys.platform == "darwin"
        and getattr(args, "source", "camera") == "camera"
        and getattr(args, "camera_backend", "opencv") == "opencv"
    ):
        return 1
    return int(getattr(args, "camera_index", 0))


def _camera_index_for_assessment(
    args: argparse.Namespace, assessment: dict[str, object]
) -> int:
    """Choose the close-up E/F or full-body A/B host camera for this stage."""

    pose_camera_index = int(getattr(args, "camera_index", 0))
    if (
        getattr(args, "source", "camera") != "camera"
        or getattr(args, "camera_backend", "opencv") != "opencv"
    ):
        return pose_camera_index
    if (
        assessment.get("mode") == "screening"
        and str(assessment.get("stage", "idle")) in FACE_CAMERA_STAGES
    ):
        return _resolved_face_camera_index(args)
    return pose_camera_index


def _camera_role_for_assessment(assessment: dict[str, object]) -> str:
    if (
        assessment.get("mode") == "screening"
        and str(assessment.get("stage", "idle")) in FACE_CAMERA_STAGES
    ):
        return "face"
    return "pose"


def run_detection(
    args: argparse.Namespace,
    preview_state: PreviewState | None = None,
    stop_event: threading.Event | None = None,
    befast_session: BefastSession | None = None,
    history_store: AbnormalHistoryStore | None = None,
) -> None:
    """Run stage-aware inference, keeping Pi CPU load to one model at a time."""

    runtime_cfg = RuntimeConfig(
        frame_width=args.width, frame_height=args.height, frame_fps=args.fps
    )
    classifier = PoseClassifier(PoseClassifierConfig())
    befast = befast_session or BefastSession()
    passive_monitor = PassiveMonitor(
        PassiveMonitoringConfig(
            enabled=not bool(getattr(args, "disable_passive_monitor", False)),
            inference_fps=max(
                0.1,
                float(
                    getattr(args, "standby_pose_fps", RuntimeConfig.standby_pose_fps)
                ),
            ),
        )
    )
    scheduled_interval_seconds = max(
        0.0,
        float(
            getattr(
                args,
                "scheduled_screen_interval_hours",
                RuntimeConfig.scheduled_screen_interval_hours,
            )
        )
        * 3600.0,
    )
    next_scheduled_trigger: float | None = None
    logger = None if args.no_keypoint_log else JsonlKeypointLogger(args.log_dir)
    recorder = None
    pose_model = None
    face_model = None
    pose_error: str | None = None
    face_error: str | None = None
    last_face_ts = float("-inf")
    last_pose_guidance_ts = float("-inf")
    last_face: FaceObservation | None = None
    face_interval = 1.0 / max(float(getattr(args, "face_fps", 5.0)), 0.5)
    pose_guidance_interval = 1.0 / 5.0
    frame_count = 0
    fps = 0.0
    last_frame_time = time.perf_counter()
    last_log_time = 0.0
    emergency_was_active = False
    previous_operation_mode = "standby"
    last_client_frame_sequence = 0
    last_history_report_key: str | None = None
    initial_assessment = befast.snapshot()
    active_camera_index = _camera_index_for_assessment(args, initial_assessment)
    unavailable_camera_indices: set[int] = set()
    capture_source = (
        f"{getattr(args, 'camera_backend', 'opencv')}:{active_camera_index}"
        if args.source == "camera"
        else "local_video"
    )

    if preview_state is not None:
        preview_state.set_runtime(
            "starting",
            "正在加载 MoveNet 与 MediaPipe 人脸模型……",
            camera_ready=False,
            model_ready=False,
            face_model_ready=False,
            capture_origin="server_host",
            capture_source=capture_source,
            camera_role=_camera_role_for_assessment(initial_assessment),
            auto_camera_switch=bool(
                args.source == "camera"
                and getattr(args, "camera_backend", "opencv") == "opencv"
                and _resolved_face_camera_index(args)
                != int(getattr(args, "camera_index", 0))
            ),
            camera_fallback=False,
            client_camera_used=False,
            error=None,
        )

    try:
        try:
            pose_model = MoveNet(args.model, num_threads=args.num_threads)
        except Exception as exc:
            if preview_state is None:
                raise
            pose_error = str(exc)
            LOGGER.exception("MoveNet startup failed; camera preview will continue")

        # Namespaces created by older tests/tools have no face fields. Treat those
        # as face-disabled; parse_args always supplies disable_face=False.
        if not bool(getattr(args, "disable_face", True)):
            try:
                face_model = MediaPipeFaceLandmarker(
                    getattr(args, "face_model", "models/face_landmarker.task")
                )
            except Exception as exc:
                face_error = str(exc)
                LOGGER.exception("MediaPipe startup failed; E/F will be unavailable")

        with FrameSource(
            args.source,
            width=args.width,
            height=args.height,
            fps=args.fps,
            camera_backend=args.camera_backend,
            camera_index=active_camera_index,
        ) as source:
            if preview_state is not None:
                _publish_runtime(
                    preview_state,
                    pose_model is not None,
                    face_model is not None,
                    pose_error or face_error,
                    first_frame=True,
                )

            if args.save_event_clips:
                recorder = EventClipRecorder(
                    args.clips_dir,
                    fps=source.fps or args.fps,
                    pre_seconds=runtime_cfg.event_pre_seconds,
                    post_seconds=runtime_cfg.event_post_seconds,
                    event_prefix="befast",
                )

            while stop_event is None or not stop_event.is_set():
                assessment_before_frame = befast.snapshot()
                desired_camera_index = _camera_index_for_assessment(
                    args, assessment_before_frame
                )
                host_camera_selected = True
                if preview_state is not None:
                    _, preview_status = preview_state.snapshot()
                    host_camera_selected = (
                        preview_status.get("runtime", {}).get("camera_mode", "host")
                        == "host"
                    )
                if (
                    host_camera_selected
                    and desired_camera_index != active_camera_index
                    and desired_camera_index not in unavailable_camera_indices
                ):
                    desired_role = _camera_role_for_assessment(
                        assessment_before_frame
                    )
                    if preview_state is not None:
                        preview_state.set_runtime(
                            "switching_camera",
                            "正在按检测步骤切换摄像头……",
                            camera_ready=False,
                            capture_source=f"opencv:{desired_camera_index}",
                            camera_role=desired_role,
                        )
                    try:
                        source.switch_camera(desired_camera_index)
                    except Exception as exc:
                        unavailable_camera_indices.add(desired_camera_index)
                        LOGGER.exception(
                            "Camera %s unavailable for %s stage; restored camera %s",
                            desired_camera_index,
                            desired_role,
                            active_camera_index,
                        )
                        if preview_state is not None:
                            preview_state.set_runtime(
                                "running",
                                "目标摄像头不可用，已继续使用当前摄像头。",
                                camera_ready=True,
                                capture_source=f"opencv:{active_camera_index}",
                                camera_role=desired_role,
                                camera_fallback=True,
                                camera_switch_error=str(exc),
                            )
                    else:
                        active_camera_index = desired_camera_index
                        last_frame_time = time.perf_counter()
                        if preview_state is not None:
                            preview_state.set_runtime(
                                "running",
                                "摄像头已按当前检测步骤自动切换。",
                                camera_ready=True,
                                capture_source=f"opencv:{active_camera_index}",
                                camera_role=desired_role,
                                camera_fallback=False,
                                camera_switch_error=None,
                            )
                host_frame = source.read()
                if host_frame is None:
                    LOGGER.info("End of source: %s", args.source)
                    if preview_state is not None:
                        preview_state.set_runtime(
                            "stopped",
                            "摄像头或视频源已停止。请检查设备后重新启动服务。",
                            camera_ready=False,
                            model_ready=pose_model is not None,
                            face_model_ready=face_model is not None,
                        )
                    break

                if preview_state is not None:
                    selected = preview_state.select_input_frame(
                        host_frame.image,
                        host_frame.ts,
                        last_client_frame_sequence,
                    )
                    if selected is None:
                        continue
                    selected_image, selected_ts, last_client_frame_sequence = selected
                    frame = Frame(image=selected_image, ts=selected_ts)
                else:
                    frame = host_frame

                now = time.perf_counter()
                delta = max(now - last_frame_time, 1e-6)
                instant_fps = 1.0 / delta
                fps = instant_fps if fps <= 0 else 0.9 * fps + 0.1 * instant_fps
                last_frame_time = now

                assessment = befast.snapshot(frame.ts)
                if next_scheduled_trigger is None and scheduled_interval_seconds > 0:
                    next_scheduled_trigger = frame.ts + scheduled_interval_seconds
                if (
                    next_scheduled_trigger is not None
                    and frame.ts >= next_scheduled_trigger
                    and assessment["mode"] == "standby"
                ):
                    befast.start_screening(
                        source="scheduled",
                        reason="scheduled_screen_due",
                        now=frame.ts,
                    )
                    assessment = befast.snapshot(frame.ts)
                    next_scheduled_trigger = frame.ts + scheduled_interval_seconds

                operation_mode = str(assessment.get("mode", "standby"))
                if operation_mode == "standby" and previous_operation_mode != "standby":
                    passive_monitor.reset_alarm()
                previous_operation_mode = operation_mode

                stage = str(assessment.get("stage", "idle"))
                face_active_stage = operation_mode == "screening" and stage in {"eyes", "face"}
                face_guidance_stage = operation_mode == "screening" and stage in {
                    "retry_eyes",
                    "ready_face",
                    "retry_face",
                }
                pose_active_stage = operation_mode == "screening" and stage in {"arms", "balance"}
                pose_guidance_stage = operation_mode == "screening" and stage in {
                    "ready_arms",
                    "retry_arms",
                    "ready_balance",
                    "retry_balance",
                }
                face_stage = face_active_stage or face_guidance_stage
                pose_stage = pose_active_stage or pose_guidance_stage
                inference_mode = "preview_only"
                inference_performed = False
                keypoints = []
                idle_pose = (
                    "unavailable"
                    if pose_model is None
                    else (
                        "paused_for_face"
                        if face_stage
                        else (
                            "standby"
                            if operation_mode == "standby"
                            else "paused_between_stages"
                        )
                    )
                )
                idle_reason = (
                    "pose_model_unavailable"
                    if pose_model is None
                    else (
                        "face_stage"
                        if face_stage
                        else (
                            "low_load_standby"
                            if operation_mode == "standby"
                            else "waiting_for_guided_stage"
                        )
                    )
                )
                pose = PoseClassification(
                    pose=idle_pose,
                    confidence=0.0,
                    quality=0.0,
                    reason=idle_reason,
                )

                # E/F runs only during its guided stage. A/B gets full camera cadence.
                # Standby runs MoveNet at a low cadence and may only trigger a screen.
                if face_stage:
                    inference_mode = (
                        "active_face" if face_active_stage else "guidance_face"
                    )
                    if frame.ts - last_face_ts >= face_interval:
                        last_face_ts = frame.ts
                        inference_performed = True
                        observation = None
                        if face_model is not None:
                            try:
                                rgb = cv2.cvtColor(frame.image, cv2.COLOR_BGR2RGB)
                                observation = face_model.infer(rgb, frame.ts)
                                last_face = observation
                            except Exception as exc:
                                face_error = str(exc)
                                LOGGER.exception(
                                    "MediaPipe inference failed; disabling E/F backend"
                                )
                                face_model.close()
                                face_model = None
                        if face_active_stage:
                            befast.update_face(frame.ts, observation)
                        else:
                            befast.observe_face(frame.ts, observation)
                        assessment = befast.snapshot(frame.ts)
                elif pose_stage and pose_model is not None:
                    inference_mode = (
                        "active_pose" if pose_active_stage else "guidance_pose"
                    )
                    if (
                        pose_active_stage
                        or frame.ts - last_pose_guidance_ts >= pose_guidance_interval
                    ):
                        last_pose_guidance_ts = frame.ts
                        try:
                            rgb = cv2.cvtColor(frame.image, cv2.COLOR_BGR2RGB)
                            keypoints = pose_model.infer(rgb)
                            inference_performed = True
                            pose = classifier.classify(keypoints)
                            if pose_active_stage:
                                befast.update(frame.ts, keypoints, pose.pose)
                            else:
                                befast.observe_pose(frame.ts, keypoints, pose.pose)
                            assessment = befast.snapshot(frame.ts)
                        except Exception as exc:
                            pose_error = str(exc)
                            pose_model = None
                            LOGGER.exception(
                                "MoveNet inference failed; keeping camera/face preview alive"
                            )
                elif operation_mode == "standby":
                    inference_mode = (
                        "standby_pose"
                        if passive_monitor.config.enabled
                        else "standby_camera_only"
                    )
                    if pose_model is not None and passive_monitor.should_infer(frame.ts):
                        try:
                            rgb = cv2.cvtColor(frame.image, cv2.COLOR_BGR2RGB)
                            keypoints = pose_model.infer(rgb)
                            inference_performed = True
                            pose = classifier.classify(keypoints)
                            if passive_monitor.update(frame.ts, pose):
                                befast.start_screening(
                                    source="passive",
                                    reason="suspected_fall_trigger",
                                    now=frame.ts,
                                )
                                assessment = befast.snapshot(frame.ts)
                                operation_mode = "screening"
                                inference_mode = "passive_trigger"
                        except Exception as exc:
                            pose_error = str(exc)
                            pose_model = None
                            LOGGER.exception(
                                "Standby MoveNet inference failed; camera preview will continue"
                            )

                if preview_state is not None:
                    _publish_runtime(
                        preview_state,
                        pose_model is not None,
                        face_model is not None,
                        pose_error or face_error,
                        operation_mode=operation_mode,
                        inference_mode=inference_mode,
                        passive_enabled=passive_monitor.config.enabled,
                        standby_pose_fps=passive_monitor.config.inference_fps,
                    )

                # Long-term standby must not write one JSONL row per camera frame.
                if logger is not None and inference_performed:
                    logger.write_befast(
                        frame.ts,
                        keypoints,
                        pose.quality,
                        assessment,
                        pose.pose,
                        metrics=pose.metrics,
                    )

                annotated = draw_befast_overlay(
                    frame.image,
                    keypoints,
                    pose,
                    assessment,
                    fps,
                    face_observation=last_face if face_stage else None,
                    show_diagnostics=bool(getattr(args, "debug_overlay", False)),
                )

                if history_store is not None:
                    report = assessment.get("current_report")
                    report_key = history_store.report_key(
                        report if isinstance(report, Mapping) else None
                    )
                    if report_key != last_history_report_key:
                        try:
                            item = (
                                report.get("item")
                                if isinstance(report, Mapping)
                                else None
                            )
                            if (
                                isinstance(report, Mapping)
                                and isinstance(item, Mapping)
                                and item.get("status") == "positive"
                            ):
                                ok, encoded = cv2.imencode(
                                    ".jpg",
                                    frame.image,
                                    [
                                        int(cv2.IMWRITE_JPEG_QUALITY),
                                        int(runtime_cfg.jpeg_quality),
                                    ],
                                )
                                history_store.save_positive_report(
                                    report,
                                    encoded.tobytes() if ok else None,
                                    frame.ts,
                                )
                            last_history_report_key = report_key
                        except Exception:
                            LOGGER.exception(
                                "Unable to save abnormal BE-FAST history"
                            )

                status = {
                    "ts": round(frame.ts, 4),
                    "pose": pose.pose,
                    "pose_confidence": round(pose.confidence, 4),
                    "fps": round(fps, 2),
                    "quality": round(pose.quality, 4),
                    "face_inference_ms": (
                        round(last_face.inference_ms, 2)
                        if last_face is not None and face_stage
                        else None
                    ),
                    "befast": assessment,
                    "monitoring": passive_monitor.snapshot(
                        operation_mode,
                        inference_mode,
                    ),
                }

                if recorder is not None:
                    emergency_active = bool(assessment["emergency"])
                    emergency_triggered = emergency_active and not emergency_was_active
                    completed_clip = recorder.update(
                        annotated, frame.ts, emergency_triggered
                    )
                    emergency_was_active = emergency_active
                    if completed_clip is not None:
                        LOGGER.info("Saved event clip: %s", completed_clip)

                if preview_state is not None:
                    preview_state.update(annotated, status)

                if frame.ts - last_log_time >= runtime_cfg.log_every_seconds:
                    LOGGER.info(
                        "mode=%s inference=%s pose=%s befast=%s stage=%s fps=%.1f quality=%.2f",
                        operation_mode,
                        inference_mode,
                        pose.pose,
                        assessment["decision"],
                        assessment["stage"],
                        fps,
                        pose.quality,
                    )
                    last_log_time = frame.ts

                frame_count += 1
                if args.max_frames and frame_count >= args.max_frames:
                    LOGGER.info("Reached --max-frames=%s", args.max_frames)
                    break
    except Exception as exc:
        if preview_state is None:
            raise
        LOGGER.exception("Detection worker stopped")
        preview_state.set_runtime(
            "camera_error",
            "摄像头或视频源启动失败。请查看页面中的排障提示。",
            camera_ready=False,
            model_ready=pose_model is not None,
            face_model_ready=face_model is not None,
            error=str(exc),
            capture_origin="server_host",
            capture_source=capture_source,
            client_camera_used=False,
        )
    finally:
        if logger is not None:
            logger.flush()
            logger.close()
            LOGGER.info("Wrote keypoint log: %s", logger.path)
        if recorder is not None:
            recorder.close()
        if face_model is not None:
            face_model.close()
        if stop_event is not None:
            stop_event.set()


def _publish_runtime(
    preview_state: PreviewState,
    pose_ready: bool,
    face_ready: bool,
    error: str | None,
    first_frame: bool = False,
    operation_mode: str = "standby",
    inference_mode: str = "starting",
    passive_enabled: bool = True,
    standby_pose_fps: float = RuntimeConfig.standby_pose_fps,
) -> None:
    if pose_ready and face_ready:
        if first_frame:
            phase = "starting"
            message = "摄像头和两套模型已连接，正在等待第一帧……"
        elif operation_mode == "standby":
            phase = "standby"
            message = (
                f"树莓派正在低负载待机；MoveNet 约 {standby_pose_fps:.1f} FPS 被动观察，"
                "仅用于触发主动筛查。"
                if passive_enabled
                else "树莓派正在摄像头待机；被动模型已关闭，可随时手动启动筛查。"
            )
        else:
            phase = "running"
            message = "主动 BE-FAST 筛查已启动，请按当前步骤操作。"
    else:
        phase = "preview_only"
        missing = []
        if not face_ready:
            missing.append("E/F 人脸识别")
        if not pose_ready:
            missing.append("A/B 姿态识别")
        message = "摄像头正常；" + "、".join(missing) + "不可用。"
    preview_state.set_runtime(
        phase,
        message,
        camera_ready=True,
        model_ready=pose_ready,
        face_model_ready=face_ready,
        operation_mode=operation_mode,
        inference_mode=inference_mode,
        passive_monitor_enabled=passive_enabled,
        standby_pose_fps=round(float(standby_pose_fps), 2),
        error=error,
    )


# 程序入口：默认启动检测线程和 Flask Web 预览；--no-web 时只在命令行处理。
def _preflight_macos_camera(
    args: argparse.Namespace, preview_state: PreviewState
) -> bool:
    """Request/check AVFoundation camera access on the main thread on macOS."""

    if not (
        sys.platform == "darwin"
        and args.source == "camera"
        and args.camera_backend == "opencv"
    ):
        return True

    camera_index = int(getattr(args, "camera_index", 0))
    face_camera_index = _resolved_face_camera_index(args)
    preview_state.set_runtime(
        "starting",
        "正在请求并检查 macOS 摄像头权限……",
        camera_ready=False,
        model_ready=False,
        capture_origin="server_host",
        capture_source=f"opencv:{camera_index}",
        client_camera_used=False,
        error=None,
    )
    capture = cv2.VideoCapture(camera_index)
    opened = capture.isOpened()
    capture.release()
    if not opened:
        message = (
            f"macOS 未授权或无法打开摄像头 {camera_index}。请在“系统设置 → 隐私与安全性 "
            "→ 摄像头”中允许 Codex/终端/Python 使用摄像头，然后重新启动本服务。"
        )
        preview_state.set_runtime(
            "camera_error",
            message,
            camera_ready=False,
            model_ready=False,
            capture_origin="server_host",
            capture_source=f"opencv:{camera_index}",
            client_camera_used=False,
            error=f"OpenCV could not open camera index {camera_index} on macOS",
        )
        LOGGER.error(message)
        return False

    if face_camera_index != camera_index:
        face_capture = cv2.VideoCapture(face_camera_index)
        face_opened = face_capture.isOpened()
        face_capture.release()
        if not face_opened:
            LOGGER.warning(
                "E/F camera %s is unavailable; falling back to camera %s",
                face_camera_index,
                camera_index,
            )
            # Store a concrete fallback so stage routing does not retry index 1.
            args.face_camera_index = camera_index
            preview_state.set_runtime(
                "starting",
                "电脑前置摄像头不可用；E/F 将暂时使用与 A/B 相同的摄像头。",
                camera_ready=False,
                model_ready=False,
                capture_origin="server_host",
                capture_source=f"opencv:{camera_index}",
                camera_switch_error=(
                    f"OpenCV could not open E/F camera index {face_camera_index}"
                ),
            )
    return True


def main() -> None:
    args = parse_args()
    if bool(args.web_cert) != bool(args.web_key):
        raise SystemExit("--web-cert and --web-key must be provided together")
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    Path(args.clips_dir).mkdir(parents=True, exist_ok=True)
    history_store = AbnormalHistoryStore(args.history_dir)
    speech_service = None
    passive_speech_monitor = None
    if not args.disable_speech:
        microphone_capture = default_microphone_capture(device=args.speech_device)
        speech_service = SpeechCaptureService(
            args.speech_work_dir,
            capture_backend=microphone_capture,
            recognizer=WhisperCppRecognizer(
                model_path=args.speech_model,
                executable=args.whisper_cli,
            ),
            config=SpeechAudioConfig(
                capture_seconds=max(2.0, float(args.speech_capture_seconds))
            ),
        )
        speech_status = speech_service.snapshot()
        if not speech_status["capture_ready"]:
            LOGGER.warning(
                "Speech microphone backend unavailable: %s",
                speech_status["capture_reason"],
            )
        if not speech_status["recognizer_ready"]:
            LOGGER.warning(
                "Speech recognizer unavailable: %s",
                speech_status["recognizer_reason"],
            )
        if not args.disable_passive_speech:
            passive_speech_monitor = PassiveSpeechMonitor(
                args.speech_work_dir,
                capture_backend=microphone_capture,
                config=PassiveSpeechConfig(
                    window_seconds=max(
                        3.0, float(args.passive_speech_window_seconds)
                    ),
                    interval_seconds=max(
                        0.0, float(args.passive_speech_interval_seconds)
                    ),
                    baseline_windows=max(
                        1, int(args.passive_speech_baseline_windows)
                    ),
                ),
            )
            passive_status = passive_speech_monitor.start()
            if not passive_status["capture_ready"]:
                LOGGER.warning(
                    "Passive speech microphone backend unavailable: %s",
                    passive_status["capture_reason"],
                )

    if args.no_web:
        LOGGER.warning(
            "--no-web disables guided controls; use the Web UI for a complete BE-FAST screen"
        )
        try:
            run_detection(
                args,
                befast_session=BefastSession(),
                history_store=history_store,
            )
        finally:
            if passive_speech_monitor is not None:
                passive_speech_monitor.stop()
            if speech_service is not None:
                speech_service.cancel()
        return

    # Web 模式下，检测循环在后台线程运行，Flask 主线程负责提供页面和 MJPEG 视频流。
    stop_event = threading.Event()
    preview_state = PreviewState(jpeg_quality=RuntimeConfig.jpeg_quality)
    befast_session = BefastSession()
    worker = None
    if _preflight_macos_camera(args, preview_state):
        worker = threading.Thread(
            target=run_detection,
            args=(
                args,
                preview_state,
                stop_event,
                befast_session,
                history_store,
            ),
            daemon=True,
        )
        worker.start()

    scheme = "https" if args.web_cert else "http"
    LOGGER.info("Web preview: %s://%s:%s", scheme, args.web_host, args.web_port)
    app = create_app(
        preview_state,
        stop_event,
        befast_session,
        history_store,
        speech_service,
        passive_speech_monitor,
    )
    try:
        ssl_context = (args.web_cert, args.web_key) if args.web_cert else None
        app.run(
            host=args.web_host,
            port=args.web_port,
            threaded=True,
            ssl_context=ssl_context,
        )
    finally:
        stop_event.set()
        if passive_speech_monitor is not None:
            passive_speech_monitor.stop()
        if speech_service is not None:
            speech_service.cancel()
        if worker is not None:
            worker.join(timeout=2.0)


if __name__ == "__main__":
    main()
