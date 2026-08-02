"""Small NumPy-only binary training utilities.

This intentionally avoids a scikit-learn runtime dependency.  The exported
model is a standardized logistic regression that the Raspberry Pi can execute
with the Python standard library.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class FittedLogisticModel:
    means: np.ndarray
    scales: np.ndarray
    coefficients: np.ndarray
    intercept: float

    def predict_probability(self, matrix: np.ndarray) -> np.ndarray:
        standardized = (matrix - self.means) / self.scales
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            logits = standardized @ self.coefficients + self.intercept
        if not np.isfinite(logits).all():
            raise ValueError("model produced non-finite probabilities")
        positive = logits >= 0.0
        output = np.empty_like(logits, dtype=float)
        output[positive] = 1.0 / (1.0 + np.exp(-logits[positive]))
        exp_logits = np.exp(logits[~positive])
        output[~positive] = exp_logits / (1.0 + exp_logits)
        return output


def fit_logistic(
    matrix: np.ndarray,
    labels: np.ndarray,
    *,
    l2: float = 0.1,
    learning_rate: float = 0.03,
    steps: int = 2500,
) -> FittedLogisticModel:
    """Fit balanced logistic regression with Adam optimization."""

    matrix = np.asarray(matrix, dtype=float)
    labels = np.asarray(labels, dtype=float)
    if matrix.ndim != 2 or labels.ndim != 1 or len(matrix) != len(labels):
        raise ValueError("matrix and labels must be aligned 2-D/1-D arrays")
    unique = set(labels.tolist())
    if unique != {0.0, 1.0}:
        raise ValueError("training labels must contain both zero and one")
    if not np.isfinite(matrix).all():
        raise ValueError("training features must be finite")
    means = matrix.mean(axis=0)
    scales = matrix.std(axis=0)
    scales = np.where(scales < 1e-8, 1.0, scales)
    standardized = (matrix - means) / scales
    positive_count = float(labels.sum())
    negative_count = float(len(labels) - positive_count)
    weights = np.where(
        labels == 1.0,
        len(labels) / (2.0 * positive_count),
        len(labels) / (2.0 * negative_count),
    )

    coefficients = np.zeros(matrix.shape[1], dtype=float)
    intercept = 0.0
    first_moment = np.zeros_like(coefficients)
    second_moment = np.zeros_like(coefficients)
    intercept_first = 0.0
    intercept_second = 0.0
    beta1, beta2, epsilon = 0.9, 0.999, 1e-8

    for step in range(1, int(steps) + 1):
        # Some Apple Accelerate/NumPy builds emit spurious floating-point
        # matmul warnings even when the resulting finite arrays are valid.
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            logits = standardized @ coefficients + intercept
        probabilities = np.where(
            logits >= 0.0,
            1.0 / (1.0 + np.exp(-np.clip(logits, -700.0, 700.0))),
            np.exp(np.clip(logits, -700.0, 700.0))
            / (1.0 + np.exp(np.clip(logits, -700.0, 700.0))),
        )
        errors = (probabilities - labels) * weights
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            coefficient_gradient = standardized.T @ errors / len(labels)
        coefficient_gradient += float(l2) * coefficients / len(labels)
        intercept_gradient = float(errors.mean())

        first_moment = beta1 * first_moment + (1.0 - beta1) * coefficient_gradient
        second_moment = beta2 * second_moment + (1.0 - beta2) * coefficient_gradient**2
        intercept_first = beta1 * intercept_first + (1.0 - beta1) * intercept_gradient
        intercept_second = beta2 * intercept_second + (1.0 - beta2) * intercept_gradient**2
        correction1 = 1.0 - beta1**step
        correction2 = 1.0 - beta2**step
        coefficients -= float(learning_rate) * (
            first_moment / correction1
        ) / (np.sqrt(second_moment / correction2) + epsilon)
        intercept -= float(learning_rate) * (
            intercept_first / correction1
        ) / ((intercept_second / correction2) ** 0.5 + epsilon)

    if not np.isfinite(coefficients).all() or not np.isfinite(intercept):
        raise ValueError("logistic optimization did not converge to finite values")
    return FittedLogisticModel(means, scales, coefficients, float(intercept))


def subject_folds(
    labels: np.ndarray,
    subject_ids: Sequence[str],
    *,
    folds: int,
    seed: int,
) -> list[np.ndarray]:
    """Assign whole subjects to approximately stratified validation folds."""

    if len(labels) != len(subject_ids):
        raise ValueError("labels and subject IDs must be aligned")
    grouped: dict[str, list[int]] = {}
    for index, subject_id in enumerate(subject_ids):
        grouped.setdefault(str(subject_id), []).append(index)
    subject_labels = {
        subject: int(max(float(labels[index]) for index in indices) >= 0.5)
        for subject, indices in grouped.items()
    }
    by_class = {
        label: [subject for subject, value in subject_labels.items() if value == label]
        for label in (0, 1)
    }
    if min(len(subjects) for subjects in by_class.values()) < folds:
        raise ValueError(
            "each class must contain at least one distinct subject per fold"
        )
    rng = np.random.default_rng(seed)
    fold_subjects: list[set[str]] = [set() for _ in range(folds)]
    for label in (0, 1):
        subjects = list(by_class[label])
        rng.shuffle(subjects)
        subjects.sort(key=lambda subject: len(grouped[subject]), reverse=True)
        for subject in subjects:
            destination = min(
                range(folds),
                key=lambda index: sum(
                    len(grouped[item]) for item in fold_subjects[index]
                ),
            )
            fold_subjects[destination].add(subject)
    return [
        np.asarray(
            [
                index
                for index, subject in enumerate(subject_ids)
                if str(subject) in selected
            ],
            dtype=int,
        )
        for selected in fold_subjects
    ]


def select_threshold(
    labels: np.ndarray,
    probabilities: np.ndarray,
    *,
    target_sensitivity: float,
) -> tuple[float, dict[str, float]]:
    """Choose the most specific threshold meeting the requested sensitivity."""

    candidates = sorted(set(float(value) for value in probabilities), reverse=True)
    candidates.extend([0.999999, 0.000001])
    scored = []
    for threshold in candidates:
        metrics = binary_metrics(labels, probabilities, threshold)
        if metrics["sensitivity"] >= target_sensitivity:
            scored.append((metrics["specificity"], threshold, metrics))
    if not scored:
        raise ValueError("no threshold meets the requested sensitivity")
    _, threshold, metrics = max(scored, key=lambda item: (item[0], item[1]))
    threshold = min(0.999999, max(0.000001, float(threshold)))
    return threshold, metrics


def binary_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    labels = np.asarray(labels, dtype=int)
    predicted = np.asarray(probabilities >= threshold, dtype=int)
    tp = int(np.sum((labels == 1) & (predicted == 1)))
    tn = int(np.sum((labels == 0) & (predicted == 0)))
    fp = int(np.sum((labels == 0) & (predicted == 1)))
    fn = int(np.sum((labels == 1) & (predicted == 0)))
    sensitivity = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    precision = tp / max(tp + fp, 1)
    f1 = 2.0 * precision * sensitivity / max(precision + sensitivity, 1e-12)
    return {
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "f1": f1,
        "balanced_accuracy": (sensitivity + specificity) / 2.0,
        "true_positive": float(tp),
        "true_negative": float(tn),
        "false_positive": float(fp),
        "false_negative": float(fn),
    }


def export_model(
    path: str | Path,
    *,
    component: str,
    model_version: str,
    feature_names: Sequence[str],
    fitted: FittedLogisticModel,
    decision_threshold: float,
    training_summary: dict[str, object],
) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "component": component,
        "model_version": model_version,
        "algorithm": "standardized_logistic_regression",
        "feature_names": list(feature_names),
        "means": fitted.means.tolist(),
        "scales": fitted.scales.tolist(),
        "coefficients": fitted.coefficients.tolist(),
        "intercept": fitted.intercept,
        "decision_threshold": float(decision_threshold),
        "training_summary": training_summary,
    }
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
