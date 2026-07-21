import unittest

import numpy as np

from app.drawing import draw_befast_overlay
from app.pose_classifier import PoseClassification


class BefastDrawingTest(unittest.TestCase):
    def setUp(self):
        self.frame = np.zeros((240, 480, 3), dtype=np.uint8)
        self.pose = PoseClassification(
            pose="standby",
            confidence=0.0,
            quality=0.0,
            reason="test",
        )
        self.assessment = {
            "decision": "warning",
            "mode": "screening",
            "stage": "arms",
            "progress": 0.5,
            "items": {},
        }

    def test_large_diagnostic_box_is_hidden_by_default(self):
        output = draw_befast_overlay(
            self.frame,
            [],
            self.pose,
            self.assessment,
            fps=15.0,
        )

        self.assertTrue(np.array_equal(output, self.frame))

    def test_large_diagnostic_box_can_be_enabled_explicitly(self):
        output = draw_befast_overlay(
            self.frame,
            [],
            self.pose,
            self.assessment,
            fps=15.0,
            show_diagnostics=True,
        )

        self.assertFalse(np.array_equal(output, self.frame))


if __name__ == "__main__":
    unittest.main()
