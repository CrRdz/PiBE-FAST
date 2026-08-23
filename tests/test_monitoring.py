import unittest
from unittest.mock import MagicMock

from app.befast import BefastConfig, PersonalBalanceBaseline
from app.fall_detector import FallDetection
from app.keypoints import KEYPOINT_NAMES
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

    @staticmethod
    def standing_keypoints(body_shift=0.0):
        keypoints = [
            {"name": name, "x": 0.5, "y": 0.5, "score": 0.01}
            for name in KEYPOINT_NAMES
        ]
        values = {
            "left_shoulder": (0.40 + body_shift, 0.30),
            "right_shoulder": (0.60 + body_shift, 0.30),
            "left_hip": (0.44 + body_shift, 0.52),
            "right_hip": (0.56 + body_shift, 0.52),
            "left_ankle": (0.44, 0.90),
            "right_ankle": (0.56, 0.90),
        }
        for keypoint in keypoints:
            if keypoint["name"] in values:
                keypoint["x"], keypoint["y"] = values[keypoint["name"]]
                keypoint["score"] = 0.95
        return keypoints

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

    def test_continuous_standing_builds_baseline_then_triggers_on_sway_change(self):
        balance_config = BefastConfig(
            balance_warmup_seconds=0.0,
            balance_capture_seconds=0.2,
            balance_min_valid_samples=2,
            balance_min_valid_fraction=0.5,
            balance_baseline_windows=2,
        )
        baseline = PersonalBalanceBaseline(balance_config)
        monitor = PassiveMonitor(
            PassiveMonitoringConfig(
                inference_fps=10.0,
                trigger_cooldown_seconds=0.0,
            ),
            balance_config=balance_config,
            balance_baseline=baseline,
        )

        for ts in (0.0, 0.1, 0.2):
            self.assertFalse(
                monitor.update(ts, self.pose(), self.standing_keypoints())
            )
        first_window = monitor.snapshot("standby", "standby_pose")
        self.assertFalse(first_window["balance_change"]["baseline_ready"])

        for index, ts in enumerate((1.0, 1.1, 1.3)):
            self.assertFalse(
                monitor.update(
                    ts,
                    self.pose(),
                    self.standing_keypoints(
                        body_shift=0.002 if index % 2 else -0.002
                    ),
                )
            )
        calibrated = monitor.snapshot("standby", "standby_pose")
        self.assertTrue(calibrated["balance_change"]["baseline_ready"])

        triggered = False
        for index, ts in enumerate((2.0, 2.1, 2.3)):
            triggered = monitor.update(
                ts,
                self.pose(),
                self.standing_keypoints(
                    body_shift=0.06 if index % 2 else -0.06
                ),
            ) or triggered

        self.assertTrue(triggered)
        self.assertEqual(
            monitor.last_trigger_reason,
            "sustained_personal_balance_change",
        )
        self.assertEqual(
            monitor.snapshot("standby", "passive_trigger")["balance_change"][
                "latest_result"
            ]["reason"],
            "increased_mediolateral_sway_velocity_and_range",
        )


if __name__ == "__main__":
    unittest.main()
