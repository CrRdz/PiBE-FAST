from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from app.befast import (
    BefastConfig,
    BefastSession,
    MotionResult,
    PersonalBalanceBaseline,
)
from app.face_landmarker import FaceObservation
from app.keypoints import KEYPOINT_NAMES


def make_keypoints(overrides=None):
    keypoints = [
        {"name": name, "x": 0.5, "y": 0.5, "score": 0.01}
        for name in KEYPOINT_NAMES
    ]
    by_name = {keypoint["name"]: keypoint for keypoint in keypoints}
    for name, values in (overrides or {}).items():
        by_name[name].update(values)
        by_name[name]["score"] = values.get("score", 0.95)
    return keypoints


def standing_points(left_wrist_y=0.30, right_wrist_y=0.30, body_shift=0.0):
    return make_keypoints(
        {
            "left_shoulder": {"x": 0.40 + body_shift, "y": 0.30},
            "right_shoulder": {"x": 0.60 + body_shift, "y": 0.30},
            "left_elbow": {"x": 0.34 + body_shift, "y": 0.30},
            "right_elbow": {"x": 0.66 + body_shift, "y": 0.30},
            "left_wrist": {"x": 0.28 + body_shift, "y": left_wrist_y},
            "right_wrist": {"x": 0.72 + body_shift, "y": right_wrist_y},
            "left_hip": {"x": 0.44 + body_shift, "y": 0.52},
            "right_hip": {"x": 0.56 + body_shift, "y": 0.52},
            "left_knee": {"x": 0.44, "y": 0.70},
            "right_knee": {"x": 0.56, "y": 0.70},
            "left_ankle": {"x": 0.44, "y": 0.90},
            "right_ankle": {"x": 0.56, "y": 0.90},
        }
    )


def face_observation(
    gaze_ratio=0.5,
    mouth_corner_difference=0.0,
    left_smile=0.0,
    right_smile=0.0,
    interocular_width=0.30,
):
    landmarks = [(0.5, 0.5, 0.0) for _ in range(478)]
    half_eye_distance = interocular_width / 2.0
    landmarks[33] = (0.5 - half_eye_distance, 0.40, 0.0)
    landmarks[263] = (0.5 + half_eye_distance, 0.40, 0.0)
    eye_width = interocular_width / 3.0
    landmarks[133] = (landmarks[33][0] + eye_width, 0.40, 0.0)
    landmarks[362] = (landmarks[263][0] - eye_width, 0.40, 0.0)
    landmarks[468] = (landmarks[33][0] + eye_width * gaze_ratio, 0.40, 0.0)
    landmarks[473] = (landmarks[362][0] + eye_width * gaze_ratio, 0.40, 0.0)
    landmarks[1] = (0.5, 0.52, 0.0)
    landmarks[61] = (0.43, 0.66, 0.0)
    landmarks[291] = (
        0.57,
        0.66 + mouth_corner_difference * interocular_width,
        0.0,
    )
    return FaceObservation(
        ts=0.0,
        landmarks=tuple(landmarks),
        blendshapes={
            "mouthSmileLeft": left_smile,
            "mouthSmileRight": right_smile,
        },
        inference_ms=12.0,
    )


class BefastSessionTest(unittest.TestCase):
    def setUp(self):
        self.config = BefastConfig(
            eye_target_seconds=0.3,
            eye_min_samples_per_target=3,
            face_neutral_seconds=0.3,
            face_smile_seconds=0.3,
            face_min_samples_per_phase=3,
            arm_warmup_seconds=0.1,
            arm_capture_seconds=0.8,
            arm_min_valid_samples=5,
            balance_warmup_seconds=0.1,
            balance_capture_seconds=0.8,
            balance_min_valid_samples=5,
        )
        balance_baseline = PersonalBalanceBaseline(self.config)
        baseline_rolls = (-0.4, -0.2, 0.0, 0.2, 0.4)
        baseline_velocities = (0.0, 0.002, 0.004, 0.006, 0.008)
        for roll, velocity in zip(baseline_rolls, baseline_velocities):
            balance_baseline.assess(
                {
                    "median_trunk_roll_degrees": roll,
                    "ml_sway_mean_velocity": velocity,
                }
            )
        self.session = BefastSession(self.config, balance_baseline)

    def run_eye_screen(self, gaze_factory):
        self.session.start_stage("eyes", now=0.0)
        for index in range(20):
            ts = index * 0.05
            self.session.update_face(ts, gaze_factory(ts))
        return self.session.snapshot(now=1.0)

    def run_face_screen(self, face_factory):
        self.session.start_stage("face", now=1.0)
        for index in range(14):
            ts = 1.0 + index * 0.05
            self.session.update_face(ts, face_factory(ts))
        return self.session.snapshot(now=1.8)

    def run_arm_screen(self, frame_factory):
        self.session.start_stage("arms", now=0.0)
        for index in range(11):
            ts = index * 0.1
            self.session.update(ts, frame_factory(index), "standing")
        return self.session.snapshot(now=1.1)

    def run_balance_screen(self, frame_factory):
        self.session.start_stage("balance", now=2.0)
        for index in range(11):
            ts = 2.0 + index * 0.1
            self.session.update(ts, frame_factory(index), "standing")
        return self.session.snapshot(now=3.1)

    def test_device_starts_in_low_load_standby(self):
        result = self.session.snapshot(now=10.0)

        self.assertEqual(result["mode"], "standby")
        self.assertEqual(result["decision"], "standby")
        self.assertEqual(result["stage"], "idle")

    def test_trigger_opens_a_fresh_guided_screen(self):
        self.session.start_screening(
            source="caregiver",
            reason="noticed_sudden_change",
            now=10.0,
        )

        result = self.session.snapshot(now=11.0)

        self.assertEqual(result["mode"], "screening")
        self.assertEqual(result["stage"], "idle")
        self.assertEqual(result["trigger"]["source"], "caregiver")
        self.assertEqual(result["trigger"]["reason"], "noticed_sudden_change")

        self.session.reset()
        self.assertEqual(self.session.snapshot()["mode"], "standby")

    def test_default_eye_targets_remain_visible_for_three_seconds_each(self):
        session = BefastSession()
        session.start_stage("eyes", now=100.0)

        center = session.snapshot(now=102.9)
        left = session.snapshot(now=103.1)
        right = session.snapshot(now=106.1)

        self.assertEqual(center["eye_target"], "center")
        self.assertEqual(center["eye_target_remaining"], 0.1)
        self.assertEqual(left["eye_target"], "left")
        self.assertEqual(left["eye_target_remaining"], 2.9)
        self.assertEqual(right["eye_target"], "right")
        self.assertEqual(right["eye_target_remaining"], 2.9)

    def test_live_guidance_marks_visible_face_as_ready(self):
        self.session.start_screening(now=10.0)

        self.session.observe_face(10.1, face_observation())
        result = self.session.snapshot(now=10.1)

        self.assertTrue(result["guidance"]["ready"])
        self.assertEqual(result["guidance"]["reason"], "face_and_eyes_ready")

    def test_arm_setup_guidance_requires_visible_raised_arm(self):
        self.session.start_stage("arms", now=0.0)
        self.session.stage = "ready_arms"

        self.session.observe_pose(0.1, standing_points(), "standing")
        result = self.session.snapshot(now=0.1)

        self.assertTrue(result["guidance"]["ready"])
        self.assertEqual(result["guidance"]["reason"], "arms_detected_hold_still")

    def test_symmetric_arm_hold_is_negative(self):
        result = self.run_arm_screen(lambda _: standing_points())

        self.assertEqual(result["items"]["A"]["status"], "negative")
        self.assertEqual(result["stage"], "ready_balance")

    def test_persistent_lower_left_arm_is_positive(self):
        result = self.run_arm_screen(
            lambda _: standing_points(left_wrist_y=0.38, right_wrist_y=0.30)
        )

        self.assertEqual(result["items"]["A"]["status"], "positive")
        self.assertEqual(result["items"]["A"]["affected_side"], "left")

    def test_asymmetric_drift_is_positive(self):
        def frame(index):
            left_y = 0.30 if index < 5 else 0.37
            return standing_points(left_wrist_y=left_y, right_wrist_y=0.30)

        result = self.run_arm_screen(frame)

        self.assertEqual(result["items"]["A"]["status"], "positive")
        self.assertIn(
            result["items"]["A"]["reason"],
            {"persistent_arm_height_asymmetry", "asymmetric_arm_drift"},
        )

    def test_stable_standing_balance_is_negative(self):
        result = self.run_balance_screen(lambda _: standing_points())

        # B remains pending until the manual dizziness/coordination question is answered.
        self.assertEqual(result["items"]["B"]["status"], "pending")
        self.session.submit_manual(
            {
                "balance_problem": False,
            },
            new_or_sudden=False,
        )
        checked = self.session.snapshot(now=3.2)
        self.assertEqual(checked["items"]["B"]["status"], "negative")
        self.assertIn(
            "ml_sway_mean_velocity",
            checked["items"]["B"]["metrics"],
        )

    def test_increased_lateral_sway_from_personal_baseline_is_positive(self):
        result = self.run_balance_screen(
            lambda index: standing_points(
                body_shift=0.05 if index % 2 else -0.05
            )
        )
        self.session.submit_manual(
            {"balance_problem": False},
            new_or_sudden=True,
        )
        checked = self.session.snapshot(now=3.2)

        self.assertEqual(result["items"]["B"]["status"], "positive")
        self.assertEqual(checked["items"]["B"]["status"], "positive")
        self.assertEqual(
            checked["items"]["B"]["reason"],
            "increased_mediolateral_sway_velocity",
        )

    def test_zero_dispersion_baseline_does_not_invent_a_noise_floor(self):
        baseline = PersonalBalanceBaseline(
            BefastConfig(balance_baseline_windows=2)
        )
        stable = {
            "median_trunk_roll_degrees": 0.0,
            "ml_sway_mean_velocity": 0.0,
        }
        baseline.assess(stable)
        baseline.assess(stable)

        result = baseline.assess(
            {
                "median_trunk_roll_degrees": 1.0,
                "ml_sway_mean_velocity": 0.1,
            }
        )

        self.assertEqual(result["status"], "unscorable")
        self.assertEqual(
            set(result["unscorable_domains"]),
            {"trunk_orientation", "mediolateral_sway"},
        )

    def test_personal_balance_baseline_persists_between_runs(self):
        config = BefastConfig(balance_baseline_windows=2)
        with TemporaryDirectory() as directory:
            path = Path(directory) / "balance-baseline.json"
            baseline = PersonalBalanceBaseline(config, path)
            baseline.assess(
                {
                    "median_trunk_roll_degrees": -0.2,
                    "ml_sway_mean_velocity": 0.01,
                }
            )

            restored = PersonalBalanceBaseline(config, path)
            self.assertEqual(restored.snapshot()["windows"], 1)
            restored.assess(
                {
                    "median_trunk_roll_degrees": 0.2,
                    "ml_sway_mean_velocity": 0.02,
                }
            )
            self.assertTrue(restored.ready)

    def test_automated_face_asymmetry_with_sudden_onset_is_emergency(self):
        self.run_face_screen(
            lambda ts: face_observation(
                mouth_corner_difference=0.12 if ts >= 1.3 else 0.0,
                left_smile=0.15 if ts >= 1.3 else 0.0,
                right_smile=0.65 if ts >= 1.3 else 0.0,
            )
        )
        self.session.submit_manual(
            {
                "balance_problem": False,
            },
            new_or_sudden=True,
            onset_time="2026-07-19T10:30",
        )

        result = self.session.snapshot()

        self.assertEqual(result["decision"], "emergency")
        self.assertTrue(result["emergency"])
        self.assertEqual(result["items"]["F"]["status"], "positive")

    def test_complete_negative_screen_is_clear(self):
        self.run_eye_screen(
            lambda ts: face_observation(
                gaze_ratio=0.5 if ts < 0.3 else (0.25 if ts < 0.6 else 0.75)
            )
        )
        self.run_face_screen(
            lambda ts: face_observation(
                left_smile=0.60 if ts >= 1.3 else 0.0,
                right_smile=0.61 if ts >= 1.3 else 0.0,
            )
        )
        self.run_arm_screen(lambda _: standing_points())
        self.run_balance_screen(lambda _: standing_points())
        self.session.submit_manual(
            {
                "balance_problem": False,
            },
            new_or_sudden=False,
        )
        self.session.prepare_component("S", now=3.3)
        self.session.start_speech_recording(now=3.4)
        self.session.submit_speech_result(
            MotionResult(
                status="negative",
                reason="no_clear_speech_abnormality",
                quality=0.9,
            ),
            new_or_sudden=False,
            now=3.5,
        )

        result = self.session.snapshot()

        self.assertEqual(result["decision"], "clear")
        self.assertTrue(all(item["status"] == "negative" for item in result["items"].values()))

    def test_low_quality_motion_is_not_treated_as_normal(self):
        result = self.run_arm_screen(lambda _: make_keypoints())

        self.assertEqual(result["items"]["A"]["status"], "insufficient")
        self.assertEqual(result["decision"], "incomplete")
        self.assertEqual(result["stage"], "retry_arms")

    def test_reduced_eye_target_response_is_positive(self):
        result = self.run_eye_screen(lambda _: face_observation(gaze_ratio=0.5))

        self.assertEqual(result["items"]["E"]["status"], "positive")
        self.assertEqual(
            result["items"]["E"]["reason"], "reduced_visual_target_response"
        )

    def test_symmetric_smile_is_negative(self):
        result = self.run_face_screen(
            lambda ts: face_observation(
                left_smile=0.58 if ts >= 1.3 else 0.0,
                right_smile=0.60 if ts >= 1.3 else 0.0,
            )
        )

        self.assertEqual(result["items"]["F"]["status"], "negative")

    def test_face_too_small_is_insufficient(self):
        result = self.run_face_screen(
            lambda _: face_observation(interocular_width=0.04)
        )

        self.assertEqual(result["items"]["F"]["status"], "insufficient")

    def test_face_can_restart_after_multiple_insufficient_retries(self):
        for attempt in range(3):
            start = float(attempt * 2)
            self.session.start_stage("face", now=start)
            for index in range(14):
                self.session.update_face(start + index * 0.05, None)

            result = self.session.snapshot(now=start + 0.8)
            self.assertEqual(result["stage"], "retry_face")
            self.assertEqual(result["retry_counts"]["face"], attempt + 1)

            self.session.observe_face(start + 0.9, face_observation())
            self.assertTrue(self.session.snapshot(now=start + 0.9)["guidance"]["ready"])

    def test_skipped_checks_advance_and_remain_skipped_in_report(self):
        self.session.start_screening(now=0.0)

        expected = (
            ("eyes", "ready_face", "E"),
            ("face", "ready_arms", "F"),
            ("arms", "ready_balance", "A"),
            ("balance", "review", "B"),
        )
        for index, (check, next_stage, item_code) in enumerate(expected):
            skipped = self.session.skip_current_stage(now=float(index + 1))
            snapshot = self.session.snapshot(now=float(index + 1))
            self.assertEqual(skipped, check)
            self.assertEqual(snapshot["stage"], next_stage)
            self.assertEqual(snapshot["items"][item_code]["status"], "skipped")
            self.assertEqual(snapshot["items"][item_code]["reason"], "user_skipped")

        self.session.submit_manual(
            {"balance_problem": False},
            new_or_sudden=False,
        )
        self.session.prepare_component("S", now=6.0)
        self.session.submit_speech_result(
            MotionResult(
                status="negative",
                reason="no_clear_speech_abnormality",
                quality=0.9,
            ),
            new_or_sudden=False,
            now=6.1,
        )
        report = self.session.snapshot()

        self.assertEqual(report["decision"], "incomplete")
        self.assertIn("one_or_more_checks_skipped", report["reasons"])
        self.assertEqual(report["items"]["S"]["status"], "negative")

    def test_skip_is_rejected_outside_an_active_automated_check(self):
        with self.assertRaises(ValueError):
            self.session.skip_current_stage(now=1.0)

        self.session.start_screening(now=2.0)
        for index in range(4):
            self.session.skip_current_stage(now=3.0 + index)
        with self.assertRaises(ValueError):
            self.session.skip_current_stage(now=8.0)

    def test_independent_component_finishes_with_an_immediate_report(self):
        self.session.prepare_component("A", now=0.0)
        self.assertEqual(self.session.snapshot(now=0.1)["stage"], "ready_arms")

        result = self.run_arm_screen(lambda _: standing_points())

        self.assertEqual(result["current_report"]["component"], "A")
        self.assertEqual(result["current_report"]["attempt"], 1)
        self.assertEqual(result["current_report"]["item"]["status"], "negative")
        self.assertEqual(result["attempt_counts"]["A"], 1)
        self.assertEqual(len(result["reports"]), 1)

    def test_independent_component_can_be_repeated_without_losing_history(self):
        for _ in range(2):
            self.session.prepare_component("A", now=0.0)
            self.run_arm_screen(lambda _: standing_points())

        result = self.session.snapshot(now=2.0)

        self.assertEqual(result["attempt_counts"]["A"], 2)
        self.assertEqual([report["attempt"] for report in result["reports"]], [1, 2])
        self.assertEqual(result["current_report"]["attempt"], 2)

    def test_speech_component_produces_its_own_report(self):
        self.session.prepare_component("S", now=1.0)
        self.session.start_speech_recording(now=1.5)
        self.session.submit_speech_result(
            MotionResult(
                status="positive",
                reason="speech_content_mismatch",
                quality=0.88,
                details={"transcript": "今天天气"},
            ),
            new_or_sudden=True,
            onset_time="2026-07-24T10:30",
            now=2.0,
        )

        result = self.session.snapshot(now=2.0)

        self.assertEqual(result["stage"], "report")
        self.assertEqual(result["current_report"]["component"], "S")
        self.assertEqual(result["current_report"]["decision"], "emergency")
        self.assertEqual(result["current_report"]["item"]["status"], "positive")

    def test_balance_negative_observation_continues_to_safe_pose_check(self):
        self.session.prepare_component("B", now=1.0)
        self.session.submit_component_observation(
            "B",
            problem=False,
            new_or_sudden=False,
            now=2.0,
        )

        result = self.session.snapshot(now=2.0)

        self.assertEqual(result["stage"], "ready_balance")
        self.assertIsNone(result["current_report"])
        self.assertEqual(result["reports"], [])


if __name__ == "__main__":
    unittest.main()
