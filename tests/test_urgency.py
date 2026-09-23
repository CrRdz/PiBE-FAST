"""Safety-path tests independent of sensor implementations."""

from __future__ import annotations

import unittest

from app.befast.urgency import screening_assessment, screening_decision


class UrgencyTest(unittest.TestCase):
    def test_positive_with_unknown_onset_is_emergency(self):
        decision, _ = screening_decision(
            {"F": {"status": "positive", "reason": "facial_asymmetry"}},
            None,
        )
        self.assertEqual(decision, "emergency")

    def test_explicitly_longstanding_positive_is_warning(self):
        decision, _ = screening_decision(
            {"F": {"status": "positive", "reason": "facial_asymmetry"}},
            False,
        )
        self.assertEqual(decision, "warning")

    def test_missing_measurement_cannot_be_clear(self):
        decision, _ = screening_decision(
            {
                "B": {"status": "negative", "reason": "not_triggered"},
                "E": {"status": "insufficient", "reason": "quality_failure"},
            },
            False,
        )
        self.assertEqual(decision, "incomplete")

    def test_research_only_eye_measurement_does_not_block_clear(self):
        decision, _ = screening_decision(
            {
                "B": {"status": "negative", "reason": "stable"},
                "E": {
                    "status": "insufficient",
                    "reason": "eye_metrics_recorded_for_validation",
                    "decision_eligible": False,
                },
                "F": {"status": "negative", "reason": "symmetric"},
                "A": {"status": "negative", "reason": "symmetric"},
                "S": {"status": "negative", "reason": "no_speech_change"},
            },
            False,
        )
        self.assertEqual(decision, "clear")

    def test_reported_visual_problem_remains_safety_eligible(self):
        decision, _ = screening_decision(
            {
                "E": {
                    "status": "positive",
                    "reason": "reported_visual_problem",
                    "decision_eligible": True,
                }
            },
            True,
        )
        self.assertEqual(decision, "emergency")

    def test_onset_is_resolved_per_positive_component(self):
        assessment = screening_assessment(
            {
                "F": {"status": "positive", "reason": "facial_asymmetry"},
                "S": {"status": "positive", "reason": "speech_change"},
            },
            {"F": False, "S": True},
        )

        self.assertEqual(assessment["urgency"], "urgent")
        self.assertEqual(assessment["urgent_components"], ["S"])

    def test_warning_does_not_hide_incomplete_acquisition(self):
        assessment = screening_assessment(
            {
                "F": {"status": "positive", "reason": "longstanding_asymmetry"},
                "A": {"status": "insufficient", "reason": "arm_not_visible"},
            },
            {"F": False, "A": None},
        )

        self.assertEqual(assessment["decision"], "warning")
        self.assertEqual(assessment["urgency"], "warning")
        self.assertEqual(assessment["completeness"], "incomplete")
        self.assertIn("motion_check_quality_insufficient", assessment["reasons"])


if __name__ == "__main__":
    unittest.main()
