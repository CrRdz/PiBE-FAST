import unittest

import numpy as np

from scripts.analyze_eyes_healthy import _enrich, _icc_2_1, _retry_summary


class HealthyEyeExperimentAnalysisTest(unittest.TestCase):
    def test_icc_is_one_for_identical_session_rankings(self):
        matrix = np.asarray([[0.1, 0.1], [0.2, 0.2], [0.3, 0.3]])

        self.assertAlmostEqual(_icc_2_1(matrix), 1.0)

    def test_enrich_recovers_endpoint_order_and_direction_comparison(self):
        metrics = {"coordinate_orientation": 1.0}
        for trial in (2, 4, 6, 8, 10, 12):
            for eye in ("left", "right"):
                metrics[f"trial_{trial}_{eye}_iris_position"] = 0.5
        for trial in (3, 9, 11):
            for eye in ("left", "right"):
                metrics[f"trial_{trial}_{eye}_iris_position"] = 0.25
        for trial in (5, 7, 13):
            for eye in ("left", "right"):
                metrics[f"trial_{trial}_{eye}_iris_position"] = 0.75
        for eye in ("left", "right"):
            for direction in ("left", "right"):
                metrics[f"{eye}_{direction}_response"] = 0.25
                for repetition in (1, 2, 3):
                    metrics[f"{eye}_{direction}_repeat_{repetition}_response"] = 0.25

        row = {
            "participant_id": "P001",
            "session_id": "S1",
            "attempt": "1",
            "condition": "baseline",
            "distance_cm": "60",
            "light_lux": "400",
            "glasses": "habitual",
            "head_condition": "still",
            "result_index": "1",
            "log_path": "unused.jsonl",
        }
        enriched = _enrich(
            row,
            {"status": "insufficient", "reason": "eye_metrics_recorded_for_validation", "quality": 1.0, "metrics": metrics},
        )

        self.assertTrue(enriched["endpoint_order_correct"])
        self.assertTrue(enriched["single_all_directions_correct"])
        self.assertTrue(enriched["triple_all_directions_correct"])
        self.assertAlmostEqual(enriched["left_left_separation"], 0.25)
        self.assertAlmostEqual(enriched["right_right_separation"], 0.25)

    def test_retry_summary_counts_recovery_without_dropping_failure(self):
        rows = [
            {
                "participant_id": "P001",
                "session_id": "S1",
                "condition": "low_light",
                "attempt": 1,
                "status": "insufficient",
                "measurement_available": False,
            },
            {
                "participant_id": "P001",
                "session_id": "S1",
                "condition": "low_light",
                "attempt": 2,
                "status": "insufficient",
                "measurement_available": True,
            },
        ]

        summary = _retry_summary(rows)

        self.assertEqual(summary["first_attempt_insufficient_groups"], 1)
        self.assertEqual(summary["successful_retries"], 1)
        self.assertEqual(summary["retry_success_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
