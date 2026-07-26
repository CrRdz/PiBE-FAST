import tempfile
import unittest
import wave
from pathlib import Path

import cv2
import numpy as np

from app.history import AbnormalHistoryStore


def positive_report(
    component="F",
    completed_at=100.0,
    attempt=1,
    reason="asymmetric_mouth_corner_movement",
    affected_side="right",
    new_or_sudden=True,
):
    return {
        "id": attempt,
        "component": component,
        "attempt": attempt,
        "completed_at": completed_at,
        "decision": "emergency" if new_or_sudden else "warning",
        "item": {
            "status": "positive",
            "source": "mediapipe_face",
            "reason": reason,
            "affected_side": affected_side,
            "quality": 0.92,
            "metrics": {"difference": 0.31},
            "details": {},
        },
        "new_or_sudden": new_or_sudden,
        "onset_time": "2026-07-26T22:30",
    }


def jpeg_bytes():
    ok, encoded = cv2.imencode(
        ".jpg", np.zeros((120, 160, 3), dtype=np.uint8)
    )
    if not ok:
        raise RuntimeError("test JPEG encoding failed")
    return encoded.tobytes()


class AbnormalHistoryStoreTest(unittest.TestCase):
    def test_relative_root_is_normalized_for_flask_evidence_routes(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            relative_root = Path(directory).relative_to(Path.cwd())
            store = AbnormalHistoryStore(relative_root)
            saved = store.save_positive_report(
                positive_report(),
                jpeg_bytes(),
                captured_at=101.0,
            )

            self.assertTrue(store.root_dir.is_absolute())
            self.assertTrue(store.get_frame_path(saved["id"]).is_absolute())

    def test_persists_frame_and_filters_by_befast_component(self):
        with tempfile.TemporaryDirectory() as directory:
            store = AbnormalHistoryStore(directory)
            saved = store.save_positive_report(
                positive_report(),
                jpeg_bytes(),
                captured_at=101.0,
            )

            records, total = store.list_records(
                components=("F",),
                affected_side="right",
                new_or_sudden=True,
            )
            other_records, other_total = store.list_records(components=("A",))

            self.assertIsNotNone(saved)
            self.assertEqual(total, 1)
            self.assertEqual(records[0]["component"], "F")
            self.assertEqual(records[0]["reason"], "asymmetric_mouth_corner_movement")
            self.assertEqual(records[0]["metrics"]["difference"], 0.31)
            self.assertTrue(records[0]["frame_url"].endswith("/frame"))
            self.assertIsNotNone(store.get_frame_path(records[0]["id"]))
            self.assertEqual(other_total, 0)
            self.assertEqual(other_records, [])

            reopened = AbnormalHistoryStore(directory)
            self.assertEqual(reopened.list_records()[1], 1)

    def test_duplicate_report_is_saved_once_and_can_attach_frame_later(self):
        with tempfile.TemporaryDirectory() as directory:
            store = AbnormalHistoryStore(directory)
            report = positive_report()

            first = store.save_positive_report(report, None, captured_at=100.5)
            second = store.save_positive_report(
                report, jpeg_bytes(), captured_at=101.0
            )

            records, total = store.list_records()
            self.assertEqual(total, 1)
            self.assertEqual(first["id"], second["id"])
            self.assertEqual(records[0]["frame_url"], second["frame_url"])
            self.assertIsNotNone(store.get_frame_path(second["id"]))

    def test_non_positive_report_is_not_saved(self):
        with tempfile.TemporaryDirectory() as directory:
            store = AbnormalHistoryStore(directory)
            report = positive_report()
            report["item"]["status"] = "negative"

            saved = store.save_positive_report(
                report, jpeg_bytes(), captured_at=101.0
            )

            self.assertIsNone(saved)
            self.assertEqual(store.list_records()[1], 0)

    def test_persists_abnormal_speech_audio_and_transcript(self):
        with tempfile.TemporaryDirectory() as directory:
            source_audio = Path(directory) / "capture.wav"
            with wave.open(str(source_audio), "wb") as target:
                target.setnchannels(1)
                target.setsampwidth(2)
                target.setframerate(16000)
                target.writeframes(np.zeros(16000, dtype="<i2").tobytes())
            report = positive_report(
                component="S",
                reason="speech_content_mismatch",
                affected_side=None,
            )
            report["item"]["details"] = {
                "transcript": "今天",
                "recognizer": "test_offline_asr",
            }
            store = AbnormalHistoryStore(directory)

            saved = store.save_positive_report(
                report,
                None,
                captured_at=101.0,
                audio_path=source_audio,
            )

            self.assertIsNotNone(saved)
            self.assertEqual(saved["details"]["transcript"], "今天")
            self.assertTrue(saved["audio_url"].endswith("/audio"))
            stored_audio = store.get_audio_path(saved["id"])
            self.assertIsNotNone(stored_audio)
            self.assertGreater(stored_audio.stat().st_size, 44)


if __name__ == "__main__":
    unittest.main()
