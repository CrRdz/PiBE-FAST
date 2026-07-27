import tempfile
import unittest
import wave
from pathlib import Path

import cv2
import numpy as np

from app.befast import BefastSession
from app.history import AbnormalHistoryStore
from app.speech_audio import SpeechAudioConfig, SpeechCaptureService, Transcript
from app.web import PreviewState, _mjpeg_stream, create_app


class _TestSpeechCapture:
    def capture(self, output_path, config, cancel_event):
        samples = (
            np.sin(
                2.0
                * np.pi
                * 220.0
                * np.arange(int(config.sample_rate * 3.0))
                / config.sample_rate
            )
            * 0.25
            * 32767
        ).astype("<i2")
        with wave.open(str(Path(output_path)), "wb") as target:
            target.setnchannels(1)
            target.setsampwidth(2)
            target.setframerate(config.sample_rate)
            target.writeframes(samples.tobytes())


class _TestSpeechRecognizer:
    name = "test_offline_asr"

    def __init__(self):
        self.text = "今天天气很好，我们一起去公园散步"

    def availability(self):
        return True, None

    def recognize(self, wav_path, language):
        return Transcript(text=self.text, engine=self.name)


class _TestPassiveSpeechMonitor:
    def __init__(self):
        self.paused = False
        self.pause_reason = None
        self.baseline_windows = 2

    def snapshot(self):
        return {
            "enabled": True,
            "state": "paused" if self.paused else "listening",
            "assessment": "calibrating",
            "paused": self.paused,
            "pause_reason": self.pause_reason,
            "capture_ready": True,
            "baseline_windows": self.baseline_windows,
            "baseline_target": 6,
            "recommend_guided_check": False,
        }

    def pause(self, reason="manual_pause", timeout=4.0):
        self.paused = True
        self.pause_reason = reason
        return True

    def resume(self):
        self.paused = False
        self.pause_reason = None
        return self.snapshot()

    def reset_baseline(self):
        self.baseline_windows = 0
        return self.snapshot()


class BefastWebApiTest(unittest.TestCase):
    def setUp(self):
        self.history_directory = tempfile.TemporaryDirectory()
        self.history_store = AbnormalHistoryStore(self.history_directory.name)
        self.session = BefastSession()
        self.state = PreviewState()
        self.speech_recognizer = _TestSpeechRecognizer()
        self.speech_service = SpeechCaptureService(
            Path(self.history_directory.name) / "speech-work",
            capture_backend=_TestSpeechCapture(),
            recognizer=self.speech_recognizer,
            config=SpeechAudioConfig(capture_seconds=3.0),
        )
        self.passive_speech_monitor = _TestPassiveSpeechMonitor()
        self.app = create_app(
            self.state,
            befast_session=self.session,
            history_store=self.history_store,
            speech_service=self.speech_service,
            passive_speech_monitor=self.passive_speech_monitor,
        )
        self.client = self.app.test_client()

    def tearDown(self):
        self.speech_service.cancel()
        self.history_directory.cleanup()

    def test_index_contains_guided_screen(self):
        response = self.client.get("/")

        self.assertEqual(response.status_code, 200)
        self.assertIn("Pi BE-FAST 筛查 Demo".encode(), response.data)
        self.assertIn("Pi BE-FAST Screening Demo".encode(), response.data)
        self.assertIn("红绿提示".encode(), response.data)
        self.assertIn("setLanguage('zh')".encode(), response.data)
        self.assertIn("setLanguage('en')".encode(), response.data)
        self.assertIn("maybeAdvance".encode(), response.data)
        self.assertIn("advanceCurrentStage".encode(), response.data)
        self.assertIn("重新检测本项".encode(), response.data)
        self.assertIn("选择单项检测".encode(), response.data)
        self.assertIn("每次完成都会立即生成报告".encode(), response.data)
        self.assertIn(b"/api/befast/component", response.data)
        self.assertIn(b"/api/befast/manual-item", response.data)
        self.assertIn(b"/api/speech/start", response.data)
        self.assertIn(b"/api/speech/complete", response.data)
        self.assertIn(b"/api/speech/passive/pause", response.data)
        self.assertIn(b"/api/speech/passive/resume", response.data)
        self.assertIn(b"/api/speech/passive/reset-baseline", response.data)
        self.assertIn(b'id="passiveSpeechPanel"', response.data)
        self.assertIn(b'id="speechSheet"', response.data)
        self.assertNotIn(b'id="speech_problem"', response.data)
        self.assertIn("跳过本项".encode(), response.data)
        self.assertIn(b"/api/befast/skip", response.data)
        self.assertIn(b"user_skipped", response.data)
        self.assertNotIn(b"transitionKey", response.data)
        self.assertIn("手机 / 当前设备".encode(), response.data)
        self.assertIn("自动切换（E/F 电脑 · A/B 手机）".encode(), response.data)
        self.assertIn("电脑前置摄像头 · E/F".encode(), response.data)
        self.assertIn("asymmetric_eye_excursion".encode(), response.data)
        self.assertIn("水平移动范围明显小于另一只眼".encode(), response.data)
        self.assertIn("E · 看左侧".encode(), response.data)
        self.assertIn(b"eye_target_remaining", response.data)
        self.assertIn(b"getUserMedia", response.data)
        self.assertIn(b"/api/camera/frame", response.data)
        self.assertIn("异常历史".encode(), response.data)
        self.assertIn(b'id="persistentHistorySheet"', response.data)
        self.assertIn(b"openPersistentHistory", response.data)
        self.assertIn(b"/api/history?", response.data)
        self.assertIn(b"historyComponentFilter", response.data)
        self.assertIn(b"screen.stage === 'idle'", response.data)
        self.assertNotIn(b'class="brand-mark"', response.data)
        self.assertIn(b"flex-direction: column", response.data)
        self.assertNotIn(
            "本 Demo 不能诊断或排除脑卒中；突然出现任何异常请立即拨打 120。".encode(),
            response.data,
        )

    def test_browser_camera_is_same_origin_and_microphone_is_disabled(self):
        response = self.client.get("/")

        self.assertEqual(
            response.headers["Permissions-Policy"],
            "camera=(self), microphone=()",
        )

    def test_can_switch_to_client_camera_and_submit_a_frame(self):
        selected = self.client.post("/api/camera/source", json={"source": "client"})
        ok, jpeg = cv2.imencode(".jpg", np.zeros((240, 320, 3), dtype=np.uint8))
        self.assertTrue(ok)

        uploaded = self.client.post(
            "/api/camera/frame",
            data=jpeg.tobytes(),
            content_type="image/jpeg",
        )
        frame = self.state.select_input_frame(
            np.ones((120, 160, 3), dtype=np.uint8),
            host_ts=1.0,
            after_client_sequence=0,
        )

        self.assertEqual(selected.status_code, 200)
        self.assertEqual(uploaded.status_code, 200)
        self.assertEqual(uploaded.get_json()["width"], 320)
        self.assertIsNotNone(frame)
        self.assertEqual(frame[0].shape[:2], (240, 320))
        self.assertEqual(frame[2], 1)

    def test_camera_source_can_change_from_component_menu(self):
        self.session.start_screening(now=1.0)

        response = self.client.post(
            "/api/camera/source", json={"source": "client"}
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["runtime"]["camera_mode"], "client")

    def test_camera_source_cannot_change_during_active_capture(self):
        self.session.start_stage("eyes", now=1.0)

        response = self.client.post(
            "/api/camera/source", json={"source": "client"}
        )

        self.assertEqual(response.status_code, 409)

    def test_status_exposes_camera_and_model_lifecycle(self):
        response = self.client.get("/api/status")

        runtime = response.get_json()["runtime"]
        self.assertEqual(response.status_code, 200)
        self.assertEqual(runtime["phase"], "starting")
        self.assertFalse(runtime["camera_ready"])
        self.assertFalse(runtime["model_ready"])
        self.assertEqual(runtime["capture_origin"], "server_host")
        self.assertFalse(runtime["client_camera_used"])

    def test_can_pause_resume_and_reset_passive_speech_monitoring(self):
        paused = self.client.post("/api/speech/passive/pause")
        reset = self.client.post("/api/speech/passive/reset-baseline")
        resumed = self.client.post("/api/speech/passive/resume")
        status = self.client.get("/api/status").get_json()["passive_speech"]

        self.assertEqual(paused.status_code, 200)
        self.assertTrue(paused.get_json()["passive_speech"]["paused"])
        self.assertEqual(reset.get_json()["passive_speech"]["baseline_windows"], 0)
        self.assertEqual(resumed.status_code, 200)
        self.assertFalse(status["paused"])

    def test_video_stream_immediately_returns_placeholder(self):
        stream = _mjpeg_stream(PreviewState(), stop_event=None)
        try:
            chunk = next(stream)
        finally:
            stream.close()

        self.assertIn(b"Content-Type: image/jpeg", chunk)
        self.assertGreater(len(chunk), 1000)

    def test_can_start_eye_stage(self):
        response = self.client.post("/api/befast/stage", json={"stage": "eyes"})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["befast"]["stage"], "eyes")

    def test_can_select_an_independent_speech_component_and_get_report(self):
        selected = self.client.post(
            "/api/befast/component", json={"component": "S"}
        )
        started = self.client.post(
            "/api/speech/start",
            json={
                "language": "zh",
                "new_or_sudden": False,
            },
        )
        self.assertTrue(self.passive_speech_monitor.paused)
        self.assertEqual(
            self.passive_speech_monitor.pause_reason,
            "guided_speech_check",
        )
        self.assertTrue(self.speech_service.wait(timeout=2.0))
        submitted = self.client.post("/api/speech/complete")

        self.assertEqual(selected.status_code, 200)
        self.assertEqual(selected.get_json()["befast"]["stage"], "speech_ready")
        self.assertEqual(started.status_code, 200)
        report = submitted.get_json()["befast"]["current_report"]
        self.assertEqual(submitted.status_code, 200)
        self.assertEqual(report["component"], "S")
        self.assertEqual(report["item"]["status"], "negative")
        self.assertEqual(
            report["item"]["details"]["transcript"],
            "今天天气很好，我们一起去公园散步",
        )
        self.assertFalse(self.passive_speech_monitor.paused)

    def test_guided_speech_preserves_a_manual_passive_pause(self):
        self.client.post("/api/speech/passive/pause")
        self.client.post("/api/befast/component", json={"component": "S"})
        started = self.client.post(
            "/api/speech/start",
            json={
                "language": "zh",
                "new_or_sudden": False,
            },
        )
        self.assertTrue(self.speech_service.wait(timeout=2.0))
        completed = self.client.post("/api/speech/complete")

        self.assertEqual(started.status_code, 200)
        self.assertEqual(completed.status_code, 200)
        self.assertTrue(self.passive_speech_monitor.paused)
        self.assertEqual(
            self.passive_speech_monitor.pause_reason,
            "manual_pause",
        )

    def test_can_skip_current_check_and_report_it(self):
        self.session.start_screening(now=1.0)

        response = self.client.post("/api/befast/skip")
        payload = response.get_json()

        self.assertEqual(response.status_code, 200)
        self.assertEqual(payload["skipped"], "eyes")
        self.assertEqual(payload["befast"]["stage"], "ready_face")
        self.assertEqual(payload["befast"]["items"]["E"]["status"], "skipped")

    def test_user_trigger_switches_standby_to_screening(self):
        response = self.client.post(
            "/api/monitoring/trigger",
            json={"source": "user", "reason": "felt_unwell"},
        )

        payload = response.get_json()["befast"]
        self.assertEqual(response.status_code, 200)
        self.assertEqual(payload["mode"], "screening")
        self.assertEqual(payload["trigger"]["reason"], "felt_unwell")

    def test_return_to_standby_clears_active_screen(self):
        self.session.start_screening(now=1.0)

        response = self.client.post("/api/monitoring/standby")

        payload = response.get_json()["befast"]
        self.assertEqual(response.status_code, 200)
        self.assertEqual(payload["mode"], "standby")
        self.assertIsNone(payload["trigger"])

    def test_rejects_incomplete_manual_payload(self):
        response = self.client.post(
            "/api/befast/manual", json={"new_or_sudden": True}
        )

        self.assertEqual(response.status_code, 400)

    def test_microphone_sudden_speech_problem_triggers_emergency(self):
        self.speech_recognizer.text = "今天"
        self.client.post("/api/befast/component", json={"component": "S"})
        started = self.client.post(
            "/api/speech/start",
            json={
                "language": "zh",
                "new_or_sudden": True,
                "onset_time": "2026-07-19T10:30",
            },
        )
        self.assertTrue(self.speech_service.wait(timeout=2.0))
        response = self.client.post("/api/speech/complete")

        payload = response.get_json()["befast"]
        self.assertEqual(started.status_code, 200)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(payload["decision"], "emergency")
        self.assertTrue(payload["emergency"])

    def test_positive_speech_saves_audio_frame_and_history_can_filter(self):
        self.state.update(
            np.zeros((240, 320, 3), dtype=np.uint8),
            {
                "ts": 100.0,
                "runtime": {},
                "befast": {},
                "monitoring": {},
            },
        )
        self.speech_recognizer.text = "今天"
        self.client.post("/api/befast/component", json={"component": "S"})
        started = self.client.post(
            "/api/speech/start",
            json={
                "language": "zh",
                "new_or_sudden": True,
                "onset_time": "2026-07-26T22:30",
            },
        )
        self.assertTrue(self.speech_service.wait(timeout=2.0))
        submitted = self.client.post("/api/speech/complete")
        filtered = self.client.get(
            "/api/history?component=F,S&new_or_sudden=true&limit=10"
        )
        excluded = self.client.get("/api/history?component=F")

        payload = filtered.get_json()
        record = payload["items"][0]
        self.assertEqual(started.status_code, 200)
        self.assertEqual(submitted.status_code, 200)
        self.assertEqual(payload["total"], 1)
        self.assertEqual(payload["filters"]["component"], ["F", "S"])
        self.assertEqual(record["component"], "S")
        self.assertEqual(record["reason"], "speech_content_mismatch")
        self.assertEqual(record["details"]["transcript"], "今天")
        self.assertTrue(record["audio_url"].endswith("/audio"))
        self.assertEqual(excluded.get_json()["total"], 0)

        detail = self.client.get(f"/api/history/{record['id']}")
        frame = self.client.get(record["frame_url"])
        audio = self.client.get(record["audio_url"])
        self.assertEqual(detail.status_code, 200)
        self.assertEqual(detail.get_json()["id"], record["id"])
        self.assertEqual(frame.status_code, 200)
        self.assertEqual(frame.mimetype, "image/jpeg")
        self.assertGreater(len(frame.data), 100)
        frame.close()
        self.assertEqual(audio.status_code, 200)
        self.assertEqual(audio.mimetype, "audio/wav")
        self.assertGreater(len(audio.data), 44)
        audio.close()

    def test_history_rejects_invalid_befast_filter(self):
        response = self.client.get("/api/history?component=X")

        self.assertEqual(response.status_code, 400)


if __name__ == "__main__":
    unittest.main()
