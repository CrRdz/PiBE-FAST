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


# 解析命令行参数，决定输入源、模型路径、Web 端口和是否保存事件片段。
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Raspberry Pi MoveNet fall detection service")
    # source 可以是 "camera"，也可以是本地视频路径；第一阶段建议先用视频文件调规则。
    parser.add_argument("--source", required=True, help="'camera' or a local video file path")
    # 模型文件不放进仓库，运行前需要放到 models/ 或通过 --model 指定其他路径。
    parser.add_argument("--model", default="models/movenet_lightning.tflite")
    # opencv 适合普通 USB 摄像头和本地视频；picamera2 适合 Raspberry Pi 官方摄像头。
    parser.add_argument("--camera-backend", choices=["opencv", "picamera2"], default="opencv")
    parser.add_argument("--width", type=int, default=RuntimeConfig.frame_width)
    parser.add_argument("--height", type=int, default=RuntimeConfig.frame_height)
    parser.add_argument("--fps", type=int, default=RuntimeConfig.frame_fps)

    # Web 预览默认监听 0.0.0.0，树莓派上可通过 SSH 端口转发到本机浏览器。
    parser.add_argument("--web-host", default="0.0.0.0")
    parser.add_argument("--web-port", type=int, default=8080)

    parser.add_argument("--no-web", action="store_true", help="Process frames without starting Flask")
    # 关闭 JSONL 可以减少磁盘写入；调试姿态规则时建议保持开启。
    parser.add_argument("--no-keypoint-log", action="store_true", help="Disable JSONL keypoint logging")
    parser.add_argument("--log-dir", default="data/keypoints")
    # 默认不持续保存视频；只有显式开启后，才保存疑似摔倒前后的短片段。
    parser.add_argument("--save-event-clips", action="store_true", help="Save MP4 clips around detected falls")
    parser.add_argument("--clips-dir", default="data/clips")
    parser.add_argument("--max-frames", type=int, default=0, help="Stop after N frames, mainly for smoke tests")
    # TFLite 推理线程数，树莓派上可根据实时性和发热情况调节。
    parser.add_argument("--num-threads", type=int, default=2)
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def run_detection(
    args: argparse.Namespace,
    preview_state: PreviewState | None = None,
    stop_event: threading.Event | None = None,
) -> None:
    # 初始化核心组件：MoveNet 负责关键点，PoseClassifier 负责姿态，FallDetector 负责连续帧摔倒判断。
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
        # FrameSource 会根据 --source 自动选择本地视频、OpenCV 摄像头或 Picamera2 摄像头。
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

            # 主处理循环：每次读取一帧，立即推理、分类、判断 fall，并更新预览/日志。
            while stop_event is None or not stop_event.is_set():
                frame = source.read()
                if frame is None:
                    LOGGER.info("End of source: %s", args.source)
                    break

                now = time.perf_counter()
                delta = max(now - last_frame_time, 1e-6)
                instant_fps = 1.0 / delta
                # FPS 用指数滑动平均，预览里的数字会更稳定，不会每帧大幅跳动。
                fps = instant_fps if fps <= 0 else (0.9 * fps + 0.1 * instant_fps)
                last_frame_time = now

                # OpenCV 读到的是 BGR，MoveNet 输入使用 RGB。
                rgb = cv2.cvtColor(frame.image, cv2.COLOR_BGR2RGB)
                keypoints = model.infer(rgb)
                # pose.metrics 里包含 bbox、center_y、torso angle 等，fall detector 会继续使用。
                pose = classifier.classify(keypoints)
                fall = fall_detector.update(
                    frame.ts, pose.pose, pose.metrics, quality=pose.quality
                )

                if logger is not None:
                    # 每帧写一行 JSONL，后续可以离线分析误判帧，但不会保存完整原始视频。
                    logger.write(
                        frame.ts,
                        pose.pose,
                        fall.fall,
                        keypoints,
                        pose.quality,
                        pose.metrics,
                    )

                # 预览画面使用带骨架和文字的 annotated frame，原始帧不会被持续保存。
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
                    # 只有启用 --save-event-clips 且 fall 触发时，才会保存前后几秒事件视频。
                    completed_clip = recorder.update(annotated, frame.ts, fall.fall)
                    if completed_clip is not None:
                        LOGGER.info("Saved event clip: %s", completed_clip)

                if preview_state is not None:
                    # 检测线程只更新“最新帧”；Web 页面读取它，不堆积历史帧，避免延迟越来越大。
                    preview_state.update(annotated, status)

                if frame.ts - last_log_time >= runtime_cfg.log_every_seconds:
                    # 控制台只周期性打印摘要；逐帧细节在 JSONL 里。
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
                    # 主要给 smoke test 使用，例如只跑 100 帧确认流程能通。
                    LOGGER.info("Reached --max-frames=%s", args.max_frames)
                    break
    finally:
        # 无论正常结束还是异常退出，都尽量关闭日志、录像器和后台停止信号。
        if logger is not None:
            logger.flush()
            logger.close()
            LOGGER.info("Wrote keypoint log: %s", logger.path)
        if recorder is not None:
            recorder.close()
        if stop_event is not None:
            stop_event.set()


# 程序入口：默认启动检测线程和 Flask Web 预览；--no-web 时只在命令行处理。
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

    # Web 模式下，检测循环在后台线程运行，Flask 主线程负责提供页面和 MJPEG 视频流。
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
