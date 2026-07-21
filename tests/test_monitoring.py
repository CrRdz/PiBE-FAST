import unittest
from unittest.mock import MagicMock

from app.fall_detector import FallDetection
from app.monitoring import PassiveMonitor, PassiveMonitoringConfig
from app.pose_classifier import PoseClassification


class PassiveMonitorTest(unittest.TestCase):
    @staticmethod
    def pose():
        return PoseClassification(
            pose="standing",
            confidence=0.9,
            quality=0.8,
            metrics={"center_y": 0.5, "torso_vertical_degrees": 5.0},
        )

    def test_standby_inference_is_throttled(self):
        monitor = PassiveMonitor(PassiveMonitoringConfig(inference_fps=2.0))

        self.assertTrue(monitor.should_infer(10.0))
        self.assertFalse(monitor.update(10.0, self.pose()))
        self.assertFalse(monitor.should_infer(10.2))
        self.assertTrue(monitor.should_infer(10.5))

    def test_confirmed_fall_only_emits_a_trigger_edge(self):
        monitor = PassiveMonitor(
            PassiveMonitoringConfig(inference_fps=2.0, trigger_cooldown_seconds=60.0)
        )
        monitor.fall_detector = MagicMock()
        monitor.fall_detector.update.return_value = FallDetection(
            fall=True,
            state="fall",
            confidence=1.0,
            reason="lying_after_fast_transition",
        )

        self.assertTrue(monitor.update(10.0, self.pose()))
        self.assertFalse(monitor.update(11.0, self.pose()))
        self.assertEqual(
            monitor.snapshot("standby", "standby_pose")["medical_role"],
            "trigger_only_not_stroke_diagnosis",
        )

    def test_disabled_monitor_never_requests_inference(self):
        monitor = PassiveMonitor(PassiveMonitoringConfig(enabled=False))

        self.assertFalse(monitor.should_infer(10.0))
        self.assertFalse(monitor.snapshot("standby", "standby_camera_only")["enabled"])


if __name__ == "__main__":
    unittest.main()
