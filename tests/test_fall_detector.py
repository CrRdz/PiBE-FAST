import unittest

from app.fall_detector import FallDetector


def metrics(center_y, torso_angle):
    return {"center_y": center_y, "torso_vertical_degrees": torso_angle}


class FallDetectorTest(unittest.TestCase):
    def test_lying_without_transition_is_not_fall(self):
        detector = FallDetector()

        result = None
        for ts in (0.0, 1.0, 2.0, 3.0):
            result = detector.update(ts, "lying", metrics(0.70, 80.0))

        self.assertIsNotNone(result)
        self.assertFalse(result.fall)
        self.assertEqual(result.state, "normal")

    def test_detects_fall_after_fast_transition_and_lying_hold(self):
        detector = FallDetector()

        self.assertFalse(detector.update(0.0, "standing", metrics(0.40, 10.0)).fall)
        self.assertFalse(detector.update(0.5, "standing", metrics(0.42, 12.0)).fall)
        transition = detector.update(1.0, "lying", metrics(0.66, 75.0))
        self.assertFalse(transition.fall)
        self.assertEqual(transition.state, "falling")

        self.assertFalse(detector.update(2.0, "lying", metrics(0.68, 80.0)).fall)
        result = detector.update(3.1, "lying", metrics(0.69, 82.0))

        self.assertTrue(result.fall)
        self.assertEqual(result.state, "fall")

    def test_recovers_after_stable_upright_pose(self):
        detector = FallDetector()
        detector.update(0.0, "standing", metrics(0.40, 10.0))
        detector.update(1.0, "lying", metrics(0.66, 75.0))
        detector.update(2.0, "lying", metrics(0.68, 80.0))
        self.assertTrue(detector.update(3.1, "lying", metrics(0.69, 82.0)).fall)

        still_active = detector.update(4.0, "standing", metrics(0.42, 12.0))
        recovered = detector.update(5.6, "standing", metrics(0.41, 11.0))

        self.assertTrue(still_active.fall)
        self.assertFalse(recovered.fall)
        self.assertEqual(recovered.state, "recovered")


if __name__ == "__main__":
    unittest.main()

