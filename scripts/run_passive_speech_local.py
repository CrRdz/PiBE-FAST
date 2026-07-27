"""Run only the local long-running speech monitor, without camera models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from app.passive_speech import PassiveSpeechConfig, PassiveSpeechMonitor
from app.speech_audio import default_microphone_capture


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Continuously capture local microphone windows, build a personal "
            "natural-speech baseline, and print sustained feature changes."
        )
    )
    parser.add_argument("--device", default="default")
    parser.add_argument("--work-dir", default="data/speech")
    parser.add_argument(
        "--window-seconds",
        type=float,
        default=PassiveSpeechConfig.window_seconds,
        help="Local capture window duration (minimum 3 seconds)",
    )
    parser.add_argument(
        "--interval-seconds",
        type=float,
        default=PassiveSpeechConfig.interval_seconds,
    )
    parser.add_argument(
        "--baseline-windows",
        type=int,
        default=PassiveSpeechConfig.baseline_windows,
    )
    parser.add_argument(
        "--reset-baseline",
        action="store_true",
        help="Delete the saved acoustic baseline before monitoring",
    )
    args = parser.parse_args()

    monitor = PassiveSpeechMonitor(
        Path(args.work_dir),
        capture_backend=default_microphone_capture(args.device),
        config=PassiveSpeechConfig(
            window_seconds=max(3.0, float(args.window_seconds)),
            interval_seconds=max(0.0, float(args.interval_seconds)),
            baseline_windows=max(1, int(args.baseline_windows)),
        ),
    )
    if args.reset_baseline:
        monitor.reset_baseline()
    initial = monitor.start()
    print(
        json.dumps(
            {
                "event": "started",
                "capture_ready": initial["capture_ready"],
                "capture_reason": initial["capture_reason"],
                "baseline_windows": initial["baseline_windows"],
                "baseline_target": initial["baseline_target"],
                "raw_audio_retained": initial["raw_audio_retained"],
                "evidence_version": initial["evidence_version"],
                "clinical_validation": initial["clinical_validation"],
                "trigger_policy": initial["trigger_policy"],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    last_key: tuple[object, ...] | None = None
    try:
        while True:
            status = monitor.snapshot()
            key = (
                status["state"],
                status["assessment"],
                status["last_window_at"],
                status["last_error"],
            )
            if key != last_key:
                print(
                    json.dumps(
                        {
                            "event": "status",
                            "state": status["state"],
                            "assessment": status["assessment"],
                            "baseline_windows": status["baseline_windows"],
                            "baseline_target": status["baseline_target"],
                            "recent_anomaly_votes": status[
                                "recent_anomaly_votes"
                            ],
                            "recommend_guided_check": status[
                                "recommend_guided_check"
                            ],
                            "latest_window": status["latest_window"],
                            "error": status["last_error"],
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
                last_key = key
            time.sleep(0.25)
    except KeyboardInterrupt:
        pass
    finally:
        monitor.stop()
        print('{"event":"stopped"}', flush=True)


if __name__ == "__main__":
    main()
