import unittest

from app.keypoints import KEYPOINT_NAMES
from app.pose_classifier import PoseClassifier


def make_keypoints(overrides):
    keypoints = [
        {"name": name, "x": 0.5, "y": 0.5, "score": 0.01}
        for name in KEYPOINT_NAMES
    ]
    by_name = {keypoint["name"]: keypoint for keypoint in keypoints}
    for name, values in overrides.items():
        by_name[name].update(values)
        by_name[name]["score"] = values.get("score", 0.9)
    return keypoints


class PoseClassifierTest(unittest.TestCase):
    def setUp(self):
        self.classifier = PoseClassifier()

    def test_classifies_standing(self):
        keypoints = make_keypoints(
            {
                "left_shoulder": {"x": 0.45, "y": 0.20},
                "right_shoulder": {"x": 0.55, "y": 0.20},
                "left_hip": {"x": 0.46, "y": 0.45},
                "right_hip": {"x": 0.54, "y": 0.45},
                "left_knee": {"x": 0.47, "y": 0.68},
                "right_knee": {"x": 0.53, "y": 0.68},
                "left_ankle": {"x": 0.48, "y": 0.92},
                "right_ankle": {"x": 0.52, "y": 0.92},
            }
        )

        result = self.classifier.classify(keypoints)

        self.assertEqual(result.pose, "standing")
        self.assertGreater(result.confidence, 0.8)

    def test_classifies_sitting(self):
        keypoints = make_keypoints(
            {
                "left_shoulder": {"x": 0.45, "y": 0.24},
                "right_shoulder": {"x": 0.55, "y": 0.24},
                "left_hip": {"x": 0.46, "y": 0.56},
                "right_hip": {"x": 0.54, "y": 0.56},
                "left_knee": {"x": 0.35, "y": 0.59},
                "right_knee": {"x": 0.65, "y": 0.59},
                "left_ankle": {"x": 0.35, "y": 0.82},
                "right_ankle": {"x": 0.65, "y": 0.82},
            }
        )

        result = self.classifier.classify(keypoints)

        self.assertEqual(result.pose, "sitting")
        self.assertGreater(result.confidence, 0.7)

    def test_classifies_lying(self):
        keypoints = make_keypoints(
            {
                "left_shoulder": {"x": 0.18, "y": 0.48},
                "right_shoulder": {"x": 0.18, "y": 0.58},
                "left_hip": {"x": 0.42, "y": 0.49},
                "right_hip": {"x": 0.42, "y": 0.59},
                "left_knee": {"x": 0.63, "y": 0.50},
                "right_knee": {"x": 0.63, "y": 0.58},
                "left_ankle": {"x": 0.85, "y": 0.50},
                "right_ankle": {"x": 0.85, "y": 0.57},
            }
        )

        result = self.classifier.classify(keypoints)

        self.assertEqual(result.pose, "lying")
        self.assertGreater(result.confidence, 0.8)

    def test_returns_unknown_for_low_confidence(self):
        keypoints = make_keypoints({})

        result = self.classifier.classify(keypoints)

        self.assertEqual(result.pose, "unknown")
        self.assertEqual(result.confidence, 0.0)


if __name__ == "__main__":
    unittest.main()

