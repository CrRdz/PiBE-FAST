#!/usr/bin/env python3
"""Summarize M5 target-microphone shadow sessions without raw audio."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", required=True, type=Path)
    parser.add_argument("--output", default=Path("training/reports/mdsc_shadow.json"), type=Path)
    args = parser.parse_args()
    records = [json.loads(line) for line in args.log.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not records:
        raise ValueError("shadow log is empty")
    sessions: dict[str, list[dict[str, object]]] = {}
    for record in records:
        sessions.setdefault(str(record.get("session_id", "unknown")), []).append(record)
    summaries = []
    for session_id, items in sorted(sessions.items()):
        probabilities = [float(item["probability"]) for item in items]
        latencies = sorted(float(item["inference_ms"]) for item in items)
        predicted = [bool(item["predicted_dysarthria"]) for item in items]
        summaries.append(
            {
                "session_id": session_id,
                "reference_label": str(items[0].get("reference_label", "unknown")),
                "environment": str(items[0].get("environment", "unspecified")),
                "microphone": str(items[0].get("microphone", "unspecified")),
                "windows": len(items),
                "mean_probability": mean(probabilities),
                "min_probability": min(probabilities),
                "max_probability": max(probabilities),
                "positive_fraction": sum(predicted) / len(predicted),
                "inference_ms_p50": latencies[len(latencies) // 2],
                "inference_ms_p95": latencies[min(len(latencies) - 1, int(0.95 * len(latencies)))],
            }
        )
    known = [item for item in summaries if item["reference_label"] != "unknown"]
    correct = [
        (item["positive_fraction"] >= 0.5) == (item["reference_label"] == "dysarthria")
        for item in known
    ]
    report = {
        "sessions": summaries,
        "session_count": len(summaries),
        "labeled_session_count": len(known),
        "descriptive_session_accuracy_at_model_threshold": sum(correct) / len(correct) if correct else None,
        "raw_audio_retained_by_microphone_runner": False,
        "warning": "Descriptive M5 shadow summary only; define and freeze a prospective protocol before performance claims.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
