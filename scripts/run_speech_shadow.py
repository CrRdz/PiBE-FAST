#!/usr/bin/env python3
"""M5 target-device prospective logger; it stores predictions, never raw audio."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile
import threading
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.speech_audio import SpeechAudioConfig, default_microphone_capture
from app.speech_representation import DysarthriaRepresentationModel


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/mdsc_dysarthria_v1.json", type=Path)
    parser.add_argument("--device", default="default")
    parser.add_argument("--window-seconds", type=float, default=5.0)
    parser.add_argument("--interval-seconds", type=float, default=2.0)
    parser.add_argument("--windows", type=int, default=20)
    parser.add_argument("--output", default=Path("data/speech/mdsc-shadow.jsonl"), type=Path)
    parser.add_argument("--wav", action="append", type=Path, help="Evaluate WAV instead of microphone; repeatable")
    parser.add_argument("--session-id", default="local-shadow")
    parser.add_argument("--label", choices=("unknown", "control", "dysarthria"), default="unknown")
    parser.add_argument("--environment", default="unspecified")
    parser.add_argument("--microphone", default="unspecified")
    args = parser.parse_args()
    model = DysarthriaRepresentationModel(args.model)
    ready, reason = model.availability()
    if not ready:
        raise SystemExit(reason)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    paths = args.wav or []
    capture = default_microphone_capture(args.device)
    with args.output.open("a", encoding="utf-8") as log:
        for index in range(max(1, len(paths) if paths else args.windows)):
            temporary: Path | None = None
            try:
                if paths:
                    wav_path = paths[index]
                else:
                    handle = tempfile.NamedTemporaryFile(prefix="mdsc-shadow-", suffix=".wav", delete=False)
                    handle.close()
                    temporary = Path(handle.name)
                    capture.capture(
                        temporary,
                        SpeechAudioConfig(capture_seconds=max(2.0, args.window_seconds)),
                        threading.Event(),
                    )
                    wav_path = temporary
                started = time.perf_counter()
                prediction = model.predict_wav(wav_path)
                record = {
                    "captured_at": round(time.time(), 3),
                    "window": index + 1,
                    "session_id": args.session_id,
                    "reference_label": args.label,
                    "environment": args.environment,
                    "microphone": args.microphone,
                    "inference_ms": round((time.perf_counter() - started) * 1000.0, 3),
                    "raw_audio_retained": bool(paths),
                    **prediction.as_dict(include_embedding=False),
                }
                line = json.dumps(record, ensure_ascii=False)
                print(line, flush=True)
                log.write(line + "\n")
                log.flush()
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
            if not paths and index + 1 < args.windows:
                time.sleep(max(0.0, args.interval_seconds))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
