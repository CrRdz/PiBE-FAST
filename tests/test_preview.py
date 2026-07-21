import argparse
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from app.camera import Frame, FrameSource
from app.befast import BefastSession
from app.main import (
    _camera_index_for_assessment,
    _preflight_macos_camera,
    _resolved_face_camera_index,
    parse_args,
    run_detection,
)
from app.web import PreviewState


class _OneFrameSource:
    fps = 15.0

    def __init__(self):
        self.frames = [
            Frame(image=np.zeros((240, 320, 3), dtype=np.uint8), ts=100.0)
        ]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return None

    def read(self):
        return self.frames.pop(0) if self.frames else None


class _TimestampFrameSource(_OneFrameSource):
    def __init__(self, timestamps):
        self.frames = [
            Frame(image=np.zeros((240, 320, 3), dtype=np.uint8), ts=ts)
            for ts in timestamps
        ]


class CameraPreviewFallbackTest(unittest.TestCase):
    @staticmethod
    def runtime_args(**overrides):
        values = {
            "source": "camera",
            "model": "models/movenet_lightning.tflite",
            "face_model": "models/face_landmarker.task",
            "face_fps": 5.0,
            "disable_face": False,
            "standby_pose_fps": 2.0,
            "disable_passive_monitor": False,
            "scheduled_screen_interval_hours": 0.0,
            "camera_backend": "opencv",
            "width": 320,
            "height": 240,
            "fps": 15,
            "no_keypoint_log": True,
            "log_dir": "data/keypoints",
            "save_event_clips": False,
            "clips_dir": "data/clips",
            "max_frames": 1,
            "num_threads": 2,
        }
        values.update(overrides)
        return argparse.Namespace(**values)

    def test_cli_defaults_to_camera_on_the_service_host(self):
        with patch("app.main.sys.argv", ["pibe-fast"]):
            args = parse_args()

        self.assertEqual(args.source, "camera")
        self.assertEqual(args.camera_index, 0)
        self.assertIsNone(args.face_camera_index)

    def test_default_macos_stage_camera_routing(self):
        args = self.runtime_args(camera_index=0, face_camera_index=None)

        with patch("app.main.sys.platform", "darwin"):
            self.assertEqual(_resolved_face_camera_index(args), 1)
            self.assertEqual(
                _camera_index_for_assessment(
                    args, {"mode": "screening", "stage": "eyes"}
                ),
                1,
            )
            self.assertEqual(
                _camera_index_for_assessment(
                    args, {"mode": "screening", "stage": "face"}
                ),
                1,
            )
            self.assertEqual(
                _camera_index_for_assessment(
                    args, {"mode": "screening", "stage": "arms"}
                ),
                0,
            )
            self.assertEqual(
                _camera_index_for_assessment(
                    args, {"mode": "screening", "stage": "balance"}
                ),
                0,
            )

    def test_macos_preflight_explains_camera_permission_failure(self):
        args = argparse.Namespace(
            source="camera", camera_backend="opencv", camera_index=2
        )
        state = PreviewState()
        capture = MagicMock()
        capture.isOpened.return_value = False

        with (
            patch("app.main.sys.platform", "darwin"),
            patch("app.main.cv2.VideoCapture", return_value=capture),
            patch("app.main.LOGGER.error"),
        ):
            ready = _preflight_macos_camera(args, state)

        _, status = state.snapshot()
        self.assertFalse(ready)
        self.assertEqual(status["runtime"]["phase"], "camera_error")
        self.assertIn("系统设置", status["runtime"]["message"])
        self.assertIn("2", status["runtime"]["message"])
        capture.release.assert_called_once()

    def test_macos_preflight_falls_back_when_face_camera_is_unavailable(self):
        args = argparse.Namespace(
            source="camera",
            camera_backend="opencv",
            camera_index=0,
            face_camera_index=1,
        )
        state = PreviewState()
        pose_capture = MagicMock()
        pose_capture.isOpened.return_value = True
        face_capture = MagicMock()
        face_capture.isOpened.return_value = False

        with (
            patch("app.main.sys.platform", "darwin"),
            patch(
                "app.main.cv2.VideoCapture",
                side_effect=[pose_capture, face_capture],
            ),
            patch("app.main.LOGGER.warning"),
        ):
            ready = _preflight_macos_camera(args, state)

        self.assertTrue(ready)
        self.assertEqual(args.face_camera_index, 0)
        pose_capture.release.assert_called_once()
        face_capture.release.assert_called_once()

    def test_model_startup_error_still_publishes_camera_frame(self):
        args = self.runtime_args(model="missing.tflite", disable_face=True)
        state = PreviewState()

        with (
            patch("app.main.MoveNet", side_effect=FileNotFoundError("missing model")),
            patch("app.main.FrameSource", return_value=_OneFrameSource()),
            patch("app.main.LOGGER.exception"),
        ):
            run_detection(args, preview_state=state)

        jpeg, status = state.snapshot()
        self.assertIsNotNone(jpeg)
        self.assertTrue(status["runtime"]["has_live_frame"])
        self.assertTrue(status["runtime"]["camera_ready"])
        self.assertFalse(status["runtime"]["model_ready"])
        self.assertEqual(status["runtime"]["phase"], "preview_only")
        self.assertEqual(status["runtime"]["capture_origin"], "server_host")
        self.assertEqual(status["runtime"]["capture_source"], "opencv:0")
        self.assertFalse(status["runtime"]["client_camera_used"])
        self.assertEqual(status["pose"], "unavailable")

    def test_eye_stage_runs_face_model_and_pauses_movenet(self):
        args = self.runtime_args()
        state = PreviewState()
        session = BefastSession()
        session.start_stage("eyes", now=100.0)
        pose_backend = MagicMock()
        face_backend = MagicMock()
        face_backend.infer.return_value = None

        with (
            patch("app.main.MoveNet", return_value=pose_backend),
            patch("app.main.MediaPipeFaceLandmarker", return_value=face_backend),
            patch("app.main.FrameSource", return_value=_OneFrameSource()),
        ):
            run_detection(args, preview_state=state, befast_session=session)

        pose_backend.infer.assert_not_called()
        face_backend.infer.assert_called_once()
        face_backend.close.assert_called_once()

    def test_standby_throttles_movenet_to_configured_rate(self):
        args = self.runtime_args(disable_face=True, max_frames=3)
        state = PreviewState()
        pose_backend = MagicMock()
        pose_backend.infer.return_value = []

        with (
            patch("app.main.MoveNet", return_value=pose_backend),
            patch(
                "app.main.FrameSource",
                return_value=_TimestampFrameSource([100.0, 100.1, 100.5]),
            ),
        ):
            run_detection(args, preview_state=state)

        _, status = state.snapshot()
        self.assertEqual(pose_backend.infer.call_count, 2)
        self.assertEqual(status["befast"]["mode"], "standby")
        self.assertEqual(status["monitoring"]["inference_mode"], "standby_pose")
        self.assertEqual(status["monitoring"]["inference_count"], 2)

    def test_guidance_mode_keeps_pose_model_paused_during_eye_setup(self):
        args = self.runtime_args(disable_face=True)
        state = PreviewState()
        session = BefastSession()
        session.start_screening(now=100.0)
        pose_backend = MagicMock()

        with (
            patch("app.main.MoveNet", return_value=pose_backend),
            patch("app.main.FrameSource", return_value=_OneFrameSource()),
        ):
            run_detection(args, preview_state=state, befast_session=session)

        pose_backend.infer.assert_not_called()
        _, status = state.snapshot()
        self.assertEqual(status["monitoring"]["inference_mode"], "guidance_face")


class FrameSourceCameraSwitchTest(unittest.TestCase):
    def test_switch_camera_releases_previous_device_and_opens_target(self):
        first_capture = MagicMock()
        first_capture.isOpened.return_value = True
        first_capture.get.return_value = 15.0
        second_capture = MagicMock()
        second_capture.isOpened.return_value = True
        second_capture.get.return_value = 15.0
        source = FrameSource("camera", camera_index=0)

        with patch(
            "app.camera.cv2.VideoCapture",
            side_effect=[first_capture, second_capture],
        ) as video_capture:
            source.open()
            switched = source.switch_camera(1)

        self.assertTrue(switched)
        self.assertEqual(source.camera_index, 1)
        self.assertIs(source.capture, second_capture)
        self.assertEqual(video_capture.call_args_list[1].args, (1,))
        first_capture.release.assert_called_once()

    def test_failed_switch_restores_previous_device(self):
        first_capture = MagicMock()
        first_capture.isOpened.return_value = True
        first_capture.get.return_value = 15.0
        failed_capture = MagicMock()
        failed_capture.isOpened.return_value = False
        restored_capture = MagicMock()
        restored_capture.isOpened.return_value = True
        restored_capture.get.return_value = 15.0
        source = FrameSource("camera", camera_index=0)

        with patch(
            "app.camera.cv2.VideoCapture",
            side_effect=[first_capture, failed_capture, restored_capture],
        ):
            source.open()
            with self.assertRaisesRegex(RuntimeError, "camera 0 was restored"):
                source.switch_camera(1)

        self.assertEqual(source.camera_index, 0)
        self.assertIs(source.capture, restored_capture)
        failed_capture.release.assert_called_once()


if __name__ == "__main__":
    unittest.main()
