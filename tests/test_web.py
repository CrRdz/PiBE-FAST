import unittest

import cv2
import numpy as np

from app.befast import BefastSession
from app.web import PreviewState, _mjpeg_stream, create_app


class BefastWebApiTest(unittest.TestCase):
    def setUp(self):
        self.session = BefastSession()
        self.state = PreviewState()
        self.app = create_app(self.state, befast_session=self.session)
        self.client = self.app.test_client()

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

    def test_camera_source_cannot_change_during_active_screen(self):
        self.session.start_screening(now=1.0)

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

    def test_manual_sudden_speech_problem_triggers_emergency(self):
        response = self.client.post(
            "/api/befast/manual",
            json={
                "balance_problem": False,
                "speech_problem": True,
                "new_or_sudden": True,
                "onset_time": "2026-07-19T10:30",
            },
        )

        payload = response.get_json()["befast"]
        self.assertEqual(response.status_code, 200)
        self.assertEqual(payload["decision"], "emergency")
        self.assertTrue(payload["emergency"])


if __name__ == "__main__":
    unittest.main()
