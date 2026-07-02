"""CLI entry point for the real-time fall detection service."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import threading
import time

import cv2

from app.camera import FrameSource
from app.config import FallDetectorConfig, PoseClassifierConfig, RuntimeConfig
from app.drawing import draw_overlay
from app.event_recorder import EventClipRecorder
from app.fall_detector import FallDetector
from app.keypoint_logger import JsonlKeypointLogger
from app.movenet import MoveNet
from app.pose_classifier import PoseClassifier
from app.web import PreviewState, create_app


LOGGER = logging.getLogger("fall_detection")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Raspberry Pi MoveNet fall detection service")
    parser.add_argument("--source", required=True, help="'camera' or a local video file path")
    parser.add_argument("--model", default="models/movenet_lightning.tflite")
    parser.add_argument("--camera-backend", choices=["opencv", "picamera2"], default="opencv")
    parser.add_argument("--width", type=int, default=RuntimeConfig.frame_width)
    parser.add_argument("--height", type=int, default=RuntimeConfig.frame_height)
    parser.add_argument("--fps", type=int, default=RuntimeConfig.frame_fps)
    parser.add_argument("--web-host", default="0.0.0.0")
    parser.add_argument("--web-port", type=int, default=8080)
    parser.add_argument("--no-web", action="store_true", help="Process frames without starting Flask")
    parser.add_argument("--no-keypoint-log", action="store_true", help="Disable JSONL keypoint logging")
    parser.add_argument("--log-dir", default="data/keypoints")
    parser.add_argument("--save-event-clips", action="store_true", help="Save MP4 clips around detected falls")
    parser.add_argument("--clips-dir", default="data/clips")
    parser.add_argument("--max-frames", type=int, default=0, help="Stop after N frames, mainly for smoke tests")
    parser.add_argument("--num-threads", type=int, default=2)
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def run_detection(
    args: argparse.Namespace,
    preview_state: PreviewState | None = None,
    stop_event: threading.Event | None = None,
) -> None:
    runtime_cfg = RuntimeConfig(frame_width=args.width, frame_height=args.height, frame_fps=args.fps)
    model = MoveNet(args.model, num_threads=args.num_threads)
    classifier = PoseClassifier(PoseClassifierConfig())
    fall_detector = FallDetector(FallDetectorConfig())
    logger = None if args.no_keypoint_log else JsonlKeypointLogger(args.log_dir)
    recorder = None

    frame_count = 0
    fps = 0.0
    last_frame_time = time.perf_counter()
    last_log_time = 0.0

    try:
        with FrameSource(
            args.source,
            width=args.width,
            height=args.height,
            fps=args.fps,
            camera_backend=args.camera_backend,
        ) as source:
            if args.save_event_clips:
                recorder = EventClipRecorder(
                    args.clips_dir,
                    fps=source.fps or args.fps,
                    pre_seconds=runtime_cfg.event_pre_seconds,
                    post_seconds=runtime_cfg.event_post_seconds,
                )

            while stop_event is None or not stop_event.is_set():
                frame = source.read()
                if frame is None:
                    LOGGER.info("End of source: %s", args.source)
                    break

                now = time.perf_counter()
                delta = max(now - last_frame_time, 1e-6)
                instant_fps = 1.0 / delta
                fps = instant_fps if fps <= 0 else (0.9 * fps + 0.1 * instant_fps)
                last_frame_time = now

                rgb = cv2.cvtColor(frame.image, cv2.COLOR_BGR2RGB)
                keypoints = model.infer(rgb)
                pose = classifier.classify(keypoints)
                fall = fall_detector.update(
                    frame.ts, pose.pose, pose.metrics, quality=pose.quality
                )

                if logger is not None:
                    logger.write(
                        frame.ts,
                        pose.pose,
                        fall.fall,
                        keypoints,
                        pose.quality,
                        pose.metrics,
                    )

                annotated = draw_overlay(frame.image, keypoints, pose, fall, fps)
                status = {
                    "ts": round(frame.ts, 4),
                    "pose": pose.pose,
                    "pose_confidence": round(pose.confidence, 4),
                    "fall": fall.fall,
                    "fall_state": fall.state,
                    "fall_reason": fall.reason,
                    "fps": round(fps, 2),
                    "quality": round(pose.quality, 4),
                }

                if recorder is not None:
                    completed_clip = recorder.update(annotated, frame.ts, fall.fall)
                    if completed_clip is not None:
                        LOGGER.info("Saved event clip: %s", completed_clip)

                if preview_state is not None:
                    preview_state.update(annotated, status)

                if frame.ts - last_log_time >= runtime_cfg.log_every_seconds:
                    LOGGER.info(
                        "pose=%s fall=%s state=%s fps=%.1f quality=%.2f",
                        pose.pose,
                        fall.fall,
                        fall.state,
                        fps,
                        pose.quality,
                    )
                    last_log_time = frame.ts

                frame_count += 1
                if args.max_frames and frame_count >= args.max_frames:
                    LOGGER.info("Reached --max-frames=%s", args.max_frames)
                    break
    finally:
        if logger is not None:
            logger.flush()
            logger.close()
            LOGGER.info("Wrote keypoint log: %s", logger.path)
        if recorder is not None:
            recorder.close()
        if stop_event is not None:
            stop_event.set()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    Path(args.clips_dir).mkdir(parents=True, exist_ok=True)

    if args.no_web:
        run_detection(args)
        return

    stop_event = threading.Event()
    preview_state = PreviewState(jpeg_quality=RuntimeConfig.jpeg_quality)
    worker = threading.Thread(
        target=run_detection, args=(args, preview_state, stop_event), daemon=True
    )
    worker.start()

    LOGGER.info("Web preview: http://%s:%s", args.web_host, args.web_port)
    app = create_app(preview_state, stop_event)
    try:
        app.run(host=args.web_host, port=args.web_port, threaded=True)
    finally:
        stop_event.set()
        worker.join(timeout=2.0)


if __name__ == "__main__":
    main()

