"""Durable Pi-side queue for triggers awaiting the local guided-check computer."""

from __future__ import annotations

import json
import threading
import time
import uuid
from pathlib import Path
from typing import Any


class EdgeTriggerQueue:
    """Persist unresolved trigger records across disconnects and restarts."""

    def __init__(
        self,
        path: str | Path,
        *,
        repeat_prompt_seconds: float = 30.0,
        attention_required_seconds: float = 120.0,
    ) -> None:
        if repeat_prompt_seconds <= 0:
            raise ValueError("repeat_prompt_seconds must be positive")
        if attention_required_seconds < repeat_prompt_seconds:
            raise ValueError(
                "attention_required_seconds must be at least repeat_prompt_seconds"
            )
        self.path = Path(path)
        self.repeat_prompt_seconds = float(repeat_prompt_seconds)
        self.attention_required_seconds = float(attention_required_seconds)
        self.lock = threading.RLock()
        self.records: list[dict[str, Any]] = []
        self._load()

    def enqueue(
        self,
        source: str,
        reason: str,
        *,
        created_at: float | None = None,
    ) -> dict[str, Any]:
        with self.lock:
            timestamp = time.time() if created_at is None else float(created_at)
            record = {
                "id": uuid.uuid4().hex,
                "source": str(source),
                "reason": str(reason),
                "created_at": round(timestamp, 4),
                "state": "unresolved",
                "operational_level": "pending_response",
                "attempts": 0,
                "prompt_due_at": round(
                    timestamp + self.repeat_prompt_seconds,
                    4,
                ),
                "attention_required_at": round(
                    timestamp + self.attention_required_seconds,
                    4,
                ),
                "prompt_due_recorded_at": None,
                "attention_required_recorded_at": None,
                "acknowledged_at": None,
            }
            self.records.append(record)
            self._save_locked()
            return dict(record)

    def mark_delivery_attempt(self, record_id: str) -> dict[str, Any]:
        with self.lock:
            record = self._find_locked(record_id)
            record["attempts"] = int(record["attempts"]) + 1
            self._save_locked()
            return dict(record)

    def acknowledge(
        self, record_id: str, *, acknowledged_at: float | None = None
    ) -> dict[str, Any]:
        with self.lock:
            record = self._find_locked(record_id)
            record["state"] = "acknowledged"
            record["operational_level"] = "acknowledged"
            record["acknowledged_at"] = round(
                time.time() if acknowledged_at is None else acknowledged_at, 4
            )
            self._save_locked()
            return dict(record)

    def evaluate_timeouts(self, *, now: float | None = None) -> list[dict[str, Any]]:
        """Advance local no-response states without claiming an alert was sent.

        These are operational timers, not clinical severity rules.  A trigger can
        therefore require caregiver attention while its medical urgency remains
        unchanged and no external delivery channel is configured.
        """

        evaluated_at = time.time() if now is None else float(now)
        changed = False
        with self.lock:
            for record in self.records:
                if record.get("state") != "unresolved":
                    continue
                created_at = float(record.get("created_at", evaluated_at))
                had_prompt_deadline = "prompt_due_at" in record
                had_attention_deadline = "attention_required_at" in record
                prompt_due_at = float(
                    record.setdefault(
                        "prompt_due_at",
                        round(created_at + self.repeat_prompt_seconds, 4),
                    )
                )
                attention_at = float(
                    record.setdefault(
                        "attention_required_at",
                        round(created_at + self.attention_required_seconds, 4),
                    )
                )
                if not had_prompt_deadline or not had_attention_deadline:
                    changed = True
                previous = record.get("operational_level", "pending_response")
                if evaluated_at >= attention_at:
                    level = "caregiver_attention_required"
                    if record.get("prompt_due_recorded_at") is None:
                        record["prompt_due_recorded_at"] = round(evaluated_at, 4)
                        changed = True
                    if record.get("attention_required_recorded_at") is None:
                        record["attention_required_recorded_at"] = round(evaluated_at, 4)
                        changed = True
                elif evaluated_at >= prompt_due_at:
                    level = "repeat_prompt_due"
                    if record.get("prompt_due_recorded_at") is None:
                        record["prompt_due_recorded_at"] = round(evaluated_at, 4)
                        changed = True
                else:
                    level = "pending_response"
                if level != previous:
                    record["operational_level"] = level
                    changed = True
            if changed:
                self._save_locked()
            return [
                dict(record)
                for record in self.records
                if record.get("state") == "unresolved"
            ]

    def unresolved(self, *, now: float | None = None) -> list[dict[str, Any]]:
        if now is not None:
            return self.evaluate_timeouts(now=now)
        with self.lock:
            return [
                dict(record)
                for record in self.records
                if record.get("state") == "unresolved"
            ]

    def _find_locked(self, record_id: str) -> dict[str, Any]:
        for record in self.records:
            if record["id"] == record_id:
                return record
        raise KeyError(record_id)

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            records = payload.get("records", [])
            if isinstance(records, list):
                self.records = [dict(record) for record in records]
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            self.records = []

    def _save_locked(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(
            json.dumps({"schema_version": 2, "records": self.records}, indent=2)
            + "\n",
            encoding="utf-8",
        )
        temporary.replace(self.path)
