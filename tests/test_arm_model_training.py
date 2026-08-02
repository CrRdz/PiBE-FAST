import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from app.action_quality_model import ACTION_QUALITY_FEATURES, summarize_action_quality
from app.arm_model import (
    ARM_MODEL_FEATURES,
    LinearBinaryModel,
    summarize_arm_samples,
)
from app.compensation_model import (
    COMPENSATION_MODEL_FEATURES,
    COMPENSATION_TARGETS,
    compensation_frame_features,
)
from app.befast.arms import ArmDriftScreen
from app.befast.config import BefastConfig
from training.modeling import (
    export_model,
    fit_logistic,
    select_threshold,
    subject_folds,
)
from training.kinematic_features import (
    KINEMATIC_FEATURES,
    summarize_angle_segment,
)
from training.train_kinematic_stroke import aggregate_subject_probabilities
from training.train_toronto_compensation import cohort_group_folds


class ArmFeatureTest(unittest.TestCase):
    def test_action_quality_features_are_shared_and_side_specific(self):
        base = TorontoCompensationTrainingTest._points()
        raised = dict(base)
        raised["left_wrist"] = (-2.3, 2.1)
        frames = [compensation_frame_features(base)] * 10
        frames += [compensation_frame_features(raised)] * 10
        frames += [compensation_frame_features(base)] * 10

        features = summarize_action_quality(frames, "left")

        self.assertEqual(set(features), set(ACTION_QUALITY_FEATURES))
        self.assertGreater(features["wrist_vertical_range"], 0.0)
        self.assertLess(features["start_end_wrist_distance"], 1e-9)

    def test_sequence_summary_detects_left_drop(self):
        samples = [
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            (0.2, 0.0, 0.2),
            (0.4, 0.0, 0.4),
            (0.4, 0.0, 0.4),
            (0.4, 0.0, 0.4),
        ]

        features = summarize_arm_samples(samples)

        self.assertAlmostEqual(features["left_drop"], 0.4)
        self.assertAlmostEqual(features["right_drop"], 0.0)
        self.assertAlmostEqual(features["drift_difference"], 0.4)
        self.assertEqual(set(features), set(ARM_MODEL_FEATURES))

    def test_v2_summary_adds_elbow_and_torso_change(self):
        base = {
            "left_shoulder": (-1.0, 2.0), "right_shoulder": (1.0, 2.0),
            "left_elbow": (-1.5, 1.2), "right_elbow": (1.5, 1.2),
            "left_wrist": (-1.8, 0.4), "right_wrist": (1.8, 0.4),
            "left_hip": (-0.7, 0.0), "right_hip": (0.7, 0.0),
        }
        changed = dict(base)
        changed["left_wrist"] = (-0.9, 1.1)
        frames = [compensation_frame_features(base)] * 10
        frames += [compensation_frame_features(changed)] * 10

        features = summarize_arm_samples(
            [(0.0, 0.0, 0.0)] * 20,
            rich_frames=frames,
        )

        self.assertEqual(len(features), 75)
        self.assertNotEqual(features["left_elbow_angle_delta"], 0.0)

    def test_runtime_loader_and_probability_match_export(self):
        matrix = np.asarray(
            [
                [-1.2, -1.0],
                [-0.8, -1.1],
                [0.9, 1.1],
                [1.3, 0.8],
            ]
        )
        labels = np.asarray([0, 0, 1, 1])
        fitted = fit_logistic(matrix, labels, steps=700)
        with TemporaryDirectory() as directory:
            path = Path(directory) / "model.json"
            export_model(
                path,
                component="A",
                model_version="test-v1",
                feature_names=("x", "y"),
                fitted=fitted,
                decision_threshold=0.5,
                training_summary={"test": True},
            )

            runtime = LinearBinaryModel.load(path, expected_component="A")
            expected = fitted.predict_probability(np.asarray([[1.0, 1.0]]))[0]

            self.assertAlmostEqual(
                runtime.predict_probability({"x": 1.0, "y": 1.0}),
                expected,
                places=10,
            )
            self.assertGreater(expected, 0.5)

    def test_legacy_arm_model_does_not_override_bilateral_hold_rules(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "arm.json"
            payload = {
                "schema_version": 1,
                "component": "A",
                "model_version": "integration-test-v1",
                "feature_names": list(ARM_MODEL_FEATURES),
                "means": [0.0] * len(ARM_MODEL_FEATURES),
                "scales": [1.0] * len(ARM_MODEL_FEATURES),
                "coefficients": [10.0] + [0.0] * (len(ARM_MODEL_FEATURES) - 1),
                "intercept": -1.0,
                "decision_threshold": 0.5,
            }
            path.write_text(json.dumps(payload), encoding="utf-8")
            screen = ArmDriftScreen(
                BefastConfig(
                    arm_model_path=str(path),
                    arm_compensation_shadow_enabled=False,
                    arm_reach_min_wrist_motion_std=0.0,
                    arm_min_valid_samples=3,
                    arm_min_valid_fraction=0.5,
                )
            )
            screen.samples = [(0.4, 0.0, 0.4)] * 3
            screen.phase_state = "complete"
            screen.hold_capture_frames = 3
            screen.hold_valid_frames = 3

            result = screen.finish()

            self.assertEqual(result.status, "positive")
            self.assertEqual(result.reason, "persistent_arm_height_asymmetry")


class GroupedTrainingTest(unittest.TestCase):
    def test_subject_folds_never_split_a_subject(self):
        labels = np.asarray([0, 0, 0, 0, 1, 1, 1, 1])
        subjects = ["n1", "n1", "n2", "n2", "p1", "p1", "p2", "p2"]

        folds = subject_folds(labels, subjects, folds=2, seed=7)

        seen = set()
        for indices in folds:
            fold_subjects = {subjects[index] for index in indices}
            self.assertFalse(seen & fold_subjects)
            seen.update(fold_subjects)
            self.assertEqual(set(labels[indices].tolist()), {0, 1})
        self.assertEqual(seen, set(subjects))

    def test_threshold_honors_sensitivity_constraint(self):
        labels = np.asarray([0, 0, 0, 1, 1, 1])
        probabilities = np.asarray([0.1, 0.2, 0.6, 0.55, 0.8, 0.9])

        threshold, metrics = select_threshold(
            labels,
            probabilities,
            target_sensitivity=1.0,
        )

        self.assertLessEqual(threshold, 0.55)
        self.assertEqual(metrics["sensitivity"], 1.0)
        self.assertAlmostEqual(metrics["specificity"], 2 / 3)

    def test_export_contains_auditable_training_summary(self):
        matrix = np.asarray([[-1.0], [-0.5], [0.5], [1.0]])
        labels = np.asarray([0, 0, 1, 1])
        fitted = fit_logistic(matrix, labels, steps=500)
        with TemporaryDirectory() as directory:
            path = Path(directory) / "model.json"
            export_model(
                path,
                component="A",
                model_version="test-v1",
                feature_names=("signal",),
                fitted=fitted,
                decision_threshold=0.4,
                training_summary={"subjects": 4},
            )

            payload = json.loads(path.read_text())

            self.assertEqual(payload["schema_version"], 1)
            self.assertEqual(payload["training_summary"]["subjects"], 4)
            self.assertEqual(payload["feature_names"], ["signal"])


class PublicKinematicTrainingTest(unittest.TestCase):
    def test_angle_features_are_finite_and_task_coded(self):
        time = np.linspace(0.0, 1.0, 126)
        angles = np.vstack(
            [np.sin(time * np.pi * (index + 1)) for index in range(19)]
        )

        features = summarize_angle_segment(
            angles,
            sample_rate=125.0,
            task_index=3,
        )

        self.assertEqual(set(features), set(KINEMATIC_FEATURES))
        self.assertTrue(all(np.isfinite(value) for value in features.values()))
        self.assertEqual(features["task_03"], 1.0)
        self.assertEqual(sum(features[f"task_{index:02d}"] for index in range(1, 7)), 1.0)

    def test_subject_aggregation_does_not_weight_repetition_count(self):
        labels = np.asarray([0, 0, 1, 1, 1])
        probabilities = np.asarray([0.1, 0.3, 0.6, 0.8, 1.0])

        result_labels, result_probabilities, subjects = (
            aggregate_subject_probabilities(
                labels,
                probabilities,
                ["healthy", "healthy", "stroke", "stroke", "stroke"],
            )
        )

        self.assertEqual(subjects, ["healthy", "stroke"])
        np.testing.assert_array_equal(result_labels, [0, 1])
        np.testing.assert_allclose(result_probabilities, [0.2, 0.8])


class TorontoCompensationTrainingTest(unittest.TestCase):
    @staticmethod
    def _points():
        return {
            "left_shoulder": (-1.0, 2.0),
            "right_shoulder": (1.0, 2.0),
            "left_elbow": (-1.5, 1.0),
            "right_elbow": (1.5, 1.0),
            "left_wrist": (-1.8, 0.2),
            "right_wrist": (1.8, 0.2),
            "left_hip": (-0.7, 0.0),
            "right_hip": (0.7, 0.0),
        }

    def test_common_features_ignore_translation_scale_and_mirroring(self):
        points = self._points()
        transformed = {
            name: (-3.0 * point[0] + 7.0, 3.0 * point[1] - 4.0)
            for name, point in points.items()
        }

        original = compensation_frame_features(points)
        changed = compensation_frame_features(transformed)

        self.assertEqual(set(original), set(changed))
        np.testing.assert_allclose(
            [original[name] for name in original],
            [changed[name] for name in original],
            atol=1e-10,
        )

    def test_cohort_folds_keep_subjects_intact(self):
        subjects = [
            "H01", "H01", "H02", "H02", "P01", "P01", "P02", "P02"
        ]
        cohorts = [
            "healthy", "healthy", "healthy", "healthy",
            "stroke", "stroke", "stroke", "stroke",
        ]

        folds = cohort_group_folds(subjects, cohorts, folds=2, seed=3)

        assigned = set()
        for indices in folds:
            fold_subjects = {subjects[index] for index in indices}
            self.assertFalse(assigned & fold_subjects)
            assigned.update(fold_subjects)
            self.assertEqual(
                {cohorts[index] for index in indices}, {"healthy", "stroke"}
            )
        self.assertEqual(assigned, {"H01", "H02", "P01", "P02"})

    def test_toronto_models_are_shadow_only(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            for target in COMPENSATION_TARGETS:
                payload = {
                    "schema_version": 1,
                    "component": f"A_COMPENSATION_SHADOW_{target.upper()}",
                    "model_version": f"shadow-{target}",
                    "feature_names": list(COMPENSATION_MODEL_FEATURES),
                    "means": [0.0] * len(COMPENSATION_MODEL_FEATURES),
                    "scales": [1.0] * len(COMPENSATION_MODEL_FEATURES),
                    "coefficients": [0.0] * len(COMPENSATION_MODEL_FEATURES),
                    "intercept": 10.0,
                    "decision_threshold": 0.5,
                }
                (root / f"toronto_{target}_v1.json").write_text(
                    json.dumps(payload), encoding="utf-8"
                )
            screen = ArmDriftScreen(
                BefastConfig(
                    arm_model_enabled=False,
                    arm_compensation_shadow_model_dir=str(root),
                    arm_reach_min_wrist_motion_std=0.0,
                    arm_min_valid_samples=3,
                    arm_min_valid_fraction=0.5,
                )
            )
            screen.phase_state = "complete"
            screen.hold_capture_frames = 30
            screen.hold_valid_frames = 30
            screen.samples = [(0.0, 0.0, 0.0)] * 30
            frame = compensation_frame_features(self._points())
            screen.compensation_frames = [frame] * 30

            result = screen.finish()

            self.assertEqual(result.status, "negative")
            self.assertEqual(result.reason, "no_clear_arm_asymmetry")
            self.assertEqual(
                result.details["compensation_shadow_mode"],
                "research_only_no_decision",
            )
            self.assertGreater(result.metrics["shadow_trunk_rotation_probability_max"], 0.99)
