"""Tests for the shadow-only feature-level BE-FAST fusion payload."""

from __future__ import annotations

import unittest

from app.befast.config import BefastConfig
from app.befast.fusion import (
    FUSION_FEATURE_NAMES,
    FUSION_MODEL_FEATURE_NAMES,
    build_feature_fusion,
)
from app.befast.result import MotionResult, motion_report_item, report_item


def measured_item(metrics: dict[str, float], quality: float = 1.0) -> dict[str, object]:
    return motion_report_item(
        MotionResult(status="negative", quality=quality, metrics=metrics), "test"
    )


class FeatureFusionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = BefastConfig()

    def test_builds_fixed_vector_with_quality_gating_and_masks(self):
        items = {
            "B": measured_item(
                {
                    "trunk_orientation_change_score": 3.5,
                    "mediolateral_sway_change_score": 1.75,
                },
                quality=0.8,
            ),
            "E": measured_item(
                {
                    "left_gaze_range": 0.6,
                    "right_gaze_range": 0.5,
                    "inter_eye_range_asymmetry": 0.2,
                    "binocular_directional_asymmetry": 0.225,
                    "max_conjugacy_error": 0.175,
                    "conjugate_rest_gaze_deviation_degrees": 6.0,
                    "minimum_trial_valid_fraction": 0.9,
                    "minimum_response_snr": 7.0,
                    "max_repeat_relative_error": 0.1,
                },
                quality=0.9,
            ),
            "F": measured_item(
                {
                    "mouth_corner_delta": 0.075,
                    "smile_score_difference": 0.12,
                    "smile_strength": 0.44,
                },
                quality=0.6,
            ),
            "A": measured_item(
                {
                    "level_difference": 0.30,
                    "left_drop": 0.1,
                    "right_drop": 0.0,
                    "drift_difference": 0.22,
                    "action_completion_score": 0.7,
                },
                quality=0.8,
            ),
            "S": measured_item(
                {
                    "character_error_rate": 0.175,
                    "characters_per_second": 4.0,
                    "pause_fraction": 0.275,
                },
                quality=0.75,
            ),
        }

        fusion = build_feature_fusion(items, self.config)

        self.assertEqual(fusion["mode"], "research_only_no_decision")
        self.assertEqual(fusion["status"], "ready")
        self.assertEqual(fusion["feature_names"], list(FUSION_FEATURE_NAMES))
        self.assertEqual(
            fusion["model_feature_names"], list(FUSION_MODEL_FEATURE_NAMES)
        )
        self.assertEqual(
            len(fusion["model_vector"]), len(FUSION_MODEL_FEATURE_NAMES)
        )
        self.assertFalse(any(fusion["missing"].values()))
        self.assertAlmostEqual(fusion["quality"]["B"], 0.8)
        self.assertAlmostEqual(fusion["quality"]["E"], 0.9)
        self.assertAlmostEqual(fusion["quality"]["F"], 0.6)
        self.assertAlmostEqual(fusion["quality"]["A"], 0.7)
        self.assertAlmostEqual(fusion["gated_features"]["A_level_difference"], 0.21)
        self.assertAlmostEqual(fusion["domain_severity"]["B"], 0.8)
        self.assertAlmostEqual(fusion["domain_severity"]["F"], 0.6)
        self.assertAlmostEqual(fusion["aggregate"]["arm_face_laterality_agreement"], 1.0)

    def test_manual_only_item_remains_missing_not_a_sensor_zero(self):
        items = {
            "B": report_item("positive", "manual", "reported_balance_problem"),
            "E": report_item("pending", "manual", "not_fully_checked"),
            "F": report_item("pending", "test", "not_run"),
            "A": report_item("pending", "test", "not_run"),
            "S": report_item("pending", "test", "not_run"),
        }

        fusion = build_feature_fusion(items, self.config)

        self.assertEqual(fusion["status"], "awaiting_completed_measurement")
        self.assertTrue(all(fusion["missing"].values()))
        self.assertEqual(fusion["aggregate"]["reliable_domain_count"], 0)
