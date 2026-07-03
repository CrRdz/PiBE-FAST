"""JSONL keypoint logging."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Mapping, Sequence


# JSONL 骨架日志写入器，每帧写一行，便于后续调参和离线分析。
class JsonlKeypointLogger:
    def __init__(self, log_dir: str | Path, prefix: str = "session") -> None:
        # 每次运行创建一个新的 JSONL 文件，避免不同实验的数据混在一起。
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
        # JSONL 是“一行一个 JSON 对象”，适合实时追加，也方便后续用脚本逐行分析。
        record = {
            "ts": round(float(ts), 4),
            "pose": pose,
            "fall": bool(fall),
            "quality": round(float(quality), 4),
            "keypoints": [_clean_keypoint(keypoint) for keypoint in keypoints],
        }
        if metrics:
            # metrics 保存分类器中间指标，例如 bbox、躯干角度、人体中心点。
            record["metrics"] = {
                str(key): round(float(value), 5)
                for key, value in metrics.items()
                if isinstance(value, (int, float))
            }
        self.file.write(json.dumps(record, separators=(",", ":")) + "\n")

    def flush(self) -> None:
        # 手动刷盘，确保程序退出前日志尽量落到文件里。
        self.file.flush()

    def close(self) -> None:
        # 关闭文件句柄，避免数据残留在缓冲区。
        self.file.close()

    def __enter__(self) -> "JsonlKeypointLogger":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def _clean_keypoint(keypoint: Mapping[str, float]) -> dict[str, float | str]:
    # 控制小数位，日志更小也更容易人工查看。
    return {
        "name": str(keypoint.get("name", "")),
        "x": round(float(keypoint.get("x", 0.0)), 5),
        "y": round(float(keypoint.get("y", 0.0)), 5),
        "score": round(float(keypoint.get("score", 0.0)), 5),
    }
