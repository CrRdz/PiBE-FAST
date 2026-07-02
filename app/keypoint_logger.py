"""JSONL keypoint logging."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Mapping, Sequence


class JsonlKeypointLogger:
    def __init__(self, log_dir: str | Path, prefix: str = "session") -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.path = self.log_dir / f"{prefix}-{stamp}.jsonl"
        self.file = self.path.open("a", encoding="utf-8")

    def write(
        self,
        ts: float,
        pose: str,
        fall: bool,
        keypoints: Sequence[Mapping[str, float]],
        quality: float,
        metrics: Mapping[str, float] | None = None,
    ) -> None:
        record = {
            "ts": round(float(ts), 4),
            "pose": pose,
            "fall": bool(fall),
            "quality": round(float(quality), 4),
            "keypoints": [_clean_keypoint(keypoint) for keypoint in keypoints],
        }
        if metrics:
            record["metrics"] = {
                str(key): round(float(value), 5)
                for key, value in metrics.items()
                if isinstance(value, (int, float))
            }
        self.file.write(json.dumps(record, separators=(",", ":")) + "\n")

    def flush(self) -> None:
        self.file.flush()

    def close(self) -> None:
        self.file.close()

    def __enter__(self) -> "JsonlKeypointLogger":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def _clean_keypoint(keypoint: Mapping[str, float]) -> dict[str, float | str]:
    return {
        "name": str(keypoint.get("name", "")),
        "x": round(float(keypoint.get("x", 0.0)), 5),
        "y": round(float(keypoint.get("y", 0.0)), 5),
        "score": round(float(keypoint.get("score", 0.0)), 5),
    }

