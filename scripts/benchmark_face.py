"""Benchmark Face Landmarker without saving camera frames."""

from __future__ import annotations

import argparse
from statistics import median
import time

import cv2

from app.face_landmarker import MediaPipeFaceLandmarker


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/face_landmarker.task")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--frames", type=int, default=30)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    args = parser.parse_args()

    capture = cv2.VideoCapture(args.camera)
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open camera {args.camera}")
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    inference_ms: list[float] = []
    detections = 0
    started = time.perf_counter()
    try:
        with MediaPipeFaceLandmarker(args.model) as landmarker:
            for index in range(args.frames):
                ok, bgr = capture.read()
                if not ok:
                    raise RuntimeError("Camera stopped while benchmarking")
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                inference_started = time.perf_counter()
                observation = landmarker.infer(rgb, time.time())
                inference_ms.append(
                    (time.perf_counter() - inference_started) * 1000.0
                )
                if observation is not None:
                    detections += 1
    finally:
        capture.release()

    elapsed = time.perf_counter() - started
    print(f"frames={args.frames} detections={detections} elapsed_s={elapsed:.3f}")
    if inference_ms:
        print(
            f"inference_ms median={median(inference_ms):.2f} "
            f"min={min(inference_ms):.2f} max={max(inference_ms):.2f}"
        )
    print("privacy=frames_were_not_saved")


if __name__ == "__main__":
    main()
