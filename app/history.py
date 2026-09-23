"""Persistent abnormal BE-FAST reports with captured frame/audio evidence."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sqlite3
import threading
from typing import Any, Mapping, Sequence


BEFAST_COMPONENTS = frozenset({"B", "E", "F", "A", "S"})


class AbnormalHistoryStore:
    """Store positive reports in SQLite, with optional JPEG and WAV evidence."""

    def __init__(self, root_dir: str | Path) -> None:
        # Flask resolves relative send_file paths from app.root_path rather than
        # the process working directory. Normalize once so saved evidence can
        # always be served from the same location it was written to.
        self.root_dir = Path(root_dir).expanduser().resolve()
        self.frames_dir = self.root_dir / "frames"
        self.audio_dir = self.root_dir / "audio"
        self.db_path = self.root_dir / "history.sqlite3"
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self.audio_dir.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self._initialize()

    @staticmethod
    def report_key(report: Mapping[str, Any] | None) -> str | None:
        """Return a stable key for one completed single-check report."""

        if not isinstance(report, Mapping):
            return None
        component = str(report.get("component", "")).strip().upper()
        if component not in BEFAST_COMPONENTS:
            return None
        try:
            completed_at = float(report["completed_at"])
            attempt = int(report["attempt"])
        except (KeyError, TypeError, ValueError):
            return None
        if report.get("session_id") and report.get("revision"):
            return f"{report['session_id']}:{component}:{report['revision']}"
        return f"{completed_at:.4f}:{component}:{attempt}"

    def save_positive_report(
        self,
        report: Mapping[str, Any],
        frame_jpeg: bytes | None,
        captured_at: float,
        audio_path: str | Path | None = None,
    ) -> dict[str, Any] | None:
        """Persist a positive report once and attach frame/audio evidence."""

        item = report.get("item")
        if not isinstance(item, Mapping):
            return None
        positive = item.get("status") == "positive" or item.get("reported_functional_problem") is True
        if not positive and report.get("kind") != "symptom_correction":
            return None
        event_key = self.report_key(report)
        if event_key is None:
            return None

        component = str(report["component"]).strip().upper()
        completed_at = float(report["completed_at"])
        attempt = int(report["attempt"])
        metrics = item.get("metrics")
        metrics_json = json.dumps(
            metrics if isinstance(metrics, Mapping) else {},
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        details = dict(item.get("details") or {})
        if report.get("session_id"):
            details["evidence_report"] = dict(report)
        details_json = json.dumps(
            details if isinstance(details, Mapping) else {},
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        sudden = report.get("new_or_sudden")
        sudden_value = None if sudden is None else int(bool(sudden))

        with self.lock:
            connection = self._connect()
            frame_path: Path | None = None
            frame_temporary_path: Path | None = None
            stored_audio_path: Path | None = None
            audio_temporary_path: Path | None = None
            try:
                connection.execute("BEGIN IMMEDIATE")
                existing = connection.execute(
                    "SELECT * FROM abnormal_history WHERE event_key = ?",
                    (event_key,),
                ).fetchone()
                if existing is None:
                    cursor = connection.execute(
                        """
                        INSERT INTO abnormal_history (
                            event_key, component, attempt, status, decision,
                            reason, affected_side, source, quality, metrics_json,
                            details_json, new_or_sudden, onset_time, completed_at,
                            captured_at
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            event_key,
                            component,
                            attempt,
                            "positive" if positive else "corrected",
                            str(report.get("decision", "warning")),
                            str(item.get("reason", "unknown")),
                            _optional_text(item.get("affected_side")),
                            str(item.get("source", "unknown")),
                            _optional_float(item.get("quality")),
                            metrics_json,
                            details_json,
                            sudden_value,
                            _optional_text(report.get("onset_time")),
                            completed_at,
                            float(captured_at),
                        ),
                    )
                    record_id = int(cursor.lastrowid)
                    existing_filename = None
                    existing_audio_filename = None
                else:
                    record_id = int(existing["id"])
                    existing_filename = existing["frame_filename"]
                    existing_audio_filename = existing["audio_filename"]
                    connection.execute(
                        "UPDATE abnormal_history SET details_json = ? WHERE id = ?",
                        (details_json, record_id),
                    )

                if frame_jpeg and not existing_filename:
                    filename = self._frame_filename(
                        record_id, component, completed_at
                    )
                    frame_path = self.frames_dir / filename
                    frame_temporary_path = frame_path.with_suffix(".tmp")
                    frame_temporary_path.write_bytes(bytes(frame_jpeg))
                    frame_temporary_path.replace(frame_path)
                    connection.execute(
                        """
                        UPDATE abnormal_history
                        SET frame_filename = ?, captured_at = ?
                        WHERE id = ?
                        """,
                        (filename, float(captured_at), record_id),
                    )

                source_audio = Path(audio_path) if audio_path is not None else None
                if (
                    source_audio is not None
                    and source_audio.is_file()
                    and not existing_audio_filename
                ):
                    audio_filename = self._audio_filename(
                        record_id, component, completed_at
                    )
                    stored_audio_path = self.audio_dir / audio_filename
                    audio_temporary_path = stored_audio_path.with_suffix(".tmp")
                    shutil.copyfile(source_audio, audio_temporary_path)
                    audio_temporary_path.replace(stored_audio_path)
                    connection.execute(
                        """
                        UPDATE abnormal_history
                        SET audio_filename = ?, captured_at = ?
                        WHERE id = ?
                        """,
                        (audio_filename, float(captured_at), record_id),
                    )

                connection.commit()
                row = connection.execute(
                    "SELECT * FROM abnormal_history WHERE id = ?",
                    (record_id,),
                ).fetchone()
                return self._public_record(row) if row is not None else None
            except Exception:
                connection.rollback()
                if frame_path is not None:
                    frame_path.unlink(missing_ok=True)
                if frame_temporary_path is not None:
                    frame_temporary_path.unlink(missing_ok=True)
                if stored_audio_path is not None:
                    stored_audio_path.unlink(missing_ok=True)
                if audio_temporary_path is not None:
                    audio_temporary_path.unlink(missing_ok=True)
                raise
            finally:
                connection.close()

    def list_records(
        self,
        *,
        components: Sequence[str] = (),
        reason: str | None = None,
        affected_side: str | None = None,
        new_or_sudden: bool | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[dict[str, Any]], int]:
        """Return newest-first abnormal records matching the supplied filters."""

        normalized_components = tuple(
            str(component).strip().upper() for component in components
        )
        invalid = [
            component
            for component in normalized_components
            if component not in BEFAST_COMPONENTS
        ]
        if invalid:
            raise ValueError(f"unsupported BE-FAST component: {invalid[0]}")
        if not 1 <= int(limit) <= 200:
            raise ValueError("limit must be between 1 and 200")
        if int(offset) < 0:
            raise ValueError("offset must be zero or greater")

        clauses: list[str] = []
        values: list[Any] = []
        if normalized_components:
            placeholders = ",".join("?" for _ in normalized_components)
            clauses.append(f"component IN ({placeholders})")
            values.extend(normalized_components)
        if reason:
            clauses.append("reason = ?")
            values.append(str(reason))
        if affected_side:
            clauses.append("affected_side = ?")
            values.append(str(affected_side).strip().lower())
        if new_or_sudden is not None:
            clauses.append("new_or_sudden = ?")
            values.append(int(bool(new_or_sudden)))

        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        with self.lock:
            connection = self._connect()
            try:
                total = int(
                    connection.execute(
                        f"SELECT COUNT(*) FROM abnormal_history{where}",
                        values,
                    ).fetchone()[0]
                )
                rows = connection.execute(
                    f"""
                    SELECT * FROM abnormal_history{where}
                    ORDER BY completed_at DESC, id DESC
                    LIMIT ? OFFSET ?
                    """,
                    [*values, int(limit), int(offset)],
                ).fetchall()
                return [self._public_record(row) for row in rows], total
            finally:
                connection.close()

    def get_record(self, record_id: int) -> dict[str, Any] | None:
        with self.lock:
            connection = self._connect()
            try:
                row = connection.execute(
                    "SELECT * FROM abnormal_history WHERE id = ?",
                    (int(record_id),),
                ).fetchone()
                return self._public_record(row) if row is not None else None
            finally:
                connection.close()

    def get_frame_path(self, record_id: int) -> Path | None:
        with self.lock:
            connection = self._connect()
            try:
                row = connection.execute(
                    "SELECT frame_filename FROM abnormal_history WHERE id = ?",
                    (int(record_id),),
                ).fetchone()
            finally:
                connection.close()
        if row is None or not row["frame_filename"]:
            return None
        filename = str(row["frame_filename"])
        if Path(filename).name != filename:
            return None
        path = self.frames_dir / filename
        return path if path.is_file() else None

    def get_audio_path(self, record_id: int) -> Path | None:
        with self.lock:
            connection = self._connect()
            try:
                row = connection.execute(
                    "SELECT audio_filename FROM abnormal_history WHERE id = ?",
                    (int(record_id),),
                ).fetchone()
            finally:
                connection.close()
        if row is None or not row["audio_filename"]:
            return None
        filename = str(row["audio_filename"])
        if Path(filename).name != filename:
            return None
        path = self.audio_dir / filename
        return path if path.is_file() else None

    def _initialize(self) -> None:
        with self.lock:
            connection = self._connect()
            try:
                connection.execute("PRAGMA journal_mode = WAL")
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS abnormal_history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        event_key TEXT NOT NULL UNIQUE,
                        component TEXT NOT NULL,
                        attempt INTEGER NOT NULL,
                        status TEXT NOT NULL,
                        decision TEXT NOT NULL,
                        reason TEXT NOT NULL,
                        affected_side TEXT,
                        source TEXT NOT NULL,
                        quality REAL,
                        metrics_json TEXT NOT NULL,
                        details_json TEXT NOT NULL DEFAULT '{}',
                        new_or_sudden INTEGER,
                        onset_time TEXT,
                        completed_at REAL NOT NULL,
                        captured_at REAL NOT NULL,
                        frame_filename TEXT,
                        audio_filename TEXT
                    )
                    """
                )
                columns = {
                    str(row["name"])
                    for row in connection.execute(
                        "PRAGMA table_info(abnormal_history)"
                    ).fetchall()
                }
                if "details_json" not in columns:
                    connection.execute(
                        """
                        ALTER TABLE abnormal_history
                        ADD COLUMN details_json TEXT NOT NULL DEFAULT '{}'
                        """
                    )
                if "audio_filename" not in columns:
                    connection.execute(
                        """
                        ALTER TABLE abnormal_history
                        ADD COLUMN audio_filename TEXT
                        """
                    )
                connection.execute(
                    """
                    CREATE INDEX IF NOT EXISTS abnormal_history_component_time_idx
                    ON abnormal_history (component, completed_at DESC)
                    """
                )
                connection.execute(
                    """
                    CREATE INDEX IF NOT EXISTS abnormal_history_reason_idx
                    ON abnormal_history (reason)
                    """
                )
                connection.commit()
            finally:
                connection.close()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path, timeout=10.0)
        connection.row_factory = sqlite3.Row
        return connection

    @staticmethod
    def _frame_filename(
        record_id: int, component: str, completed_at: float
    ) -> str:
        stamp = datetime.fromtimestamp(
            completed_at, timezone.utc
        ).strftime("%Y%m%dT%H%M%S%fZ")
        return f"{record_id:08d}-{component}-{stamp}.jpg"

    @staticmethod
    def _audio_filename(
        record_id: int, component: str, completed_at: float
    ) -> str:
        stamp = datetime.fromtimestamp(
            completed_at, timezone.utc
        ).strftime("%Y%m%dT%H%M%S%fZ")
        return f"{record_id:08d}-{component}-{stamp}.wav"

    @staticmethod
    def _public_record(row: sqlite3.Row) -> dict[str, Any]:
        sudden = row["new_or_sudden"]
        frame_url = (
            f"/api/history/{int(row['id'])}/frame"
            if row["frame_filename"]
            else None
        )
        audio_url = (
            f"/api/history/{int(row['id'])}/audio"
            if row["audio_filename"]
            else None
        )
        return {
            "id": int(row["id"]),
            "component": str(row["component"]),
            "attempt": int(row["attempt"]),
            "status": str(row["status"]),
            "decision": str(row["decision"]),
            "reason": str(row["reason"]),
            "affected_side": row["affected_side"],
            "source": str(row["source"]),
            "quality": row["quality"],
            "metrics": json.loads(str(row["metrics_json"])),
            "details": json.loads(str(row["details_json"])),
            "new_or_sudden": None if sudden is None else bool(sudden),
            "onset_time": row["onset_time"],
            "completed_at": float(row["completed_at"]),
            "captured_at": float(row["captured_at"]),
            "frame_url": frame_url,
            "audio_url": audio_url,
        }


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
