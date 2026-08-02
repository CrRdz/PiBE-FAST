#!/usr/bin/env python3
"""Train a research-only stroke/healthy functional-kinematics classifier."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.kinematic_features import KINEMATIC_FEATURES
from training.modeling import (
    binary_metrics,
    export_model,
    fit_logistic,
    select_threshold,
    subject_folds,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--features",
        type=Path,
        default=Path("training/processed/kinematic_stroke.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models/research/kinematic_stroke_v1.json"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("training/reports/kinematic_stroke_v1.json"),
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-sensitivity", type=float, default=0.90)
    parser.add_argument("--l2", type=float, default=1.0)
    parser.add_argument("--steps", type=int, default=3000)
    return parser.parse_args()


def load_features(path: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    matrix, labels, subjects = [], [], []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"subject_id", "label", *KINEMATIC_FEATURES}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"feature CSV is missing: {', '.join(sorted(missing))}")
        for row in reader:
            subjects.append(str(row["subject_id"]))
            labels.append(int(row["label"]))
            matrix.append([float(row[name]) for name in KINEMATIC_FEATURES])
    return np.asarray(matrix, dtype=float), np.asarray(labels, dtype=int), subjects


def aggregate_subject_probabilities(
    labels: np.ndarray,
    probabilities: np.ndarray,
    subjects: list[str],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    ordered = list(dict.fromkeys(subjects))
    subject_labels, subject_probabilities = [], []
    for subject in ordered:
        indices = np.asarray([value == subject for value in subjects])
        unique_labels = np.unique(labels[indices])
        if len(unique_labels) != 1:
            raise ValueError(f"subject {subject} has inconsistent labels")
        subject_labels.append(int(unique_labels[0]))
        subject_probabilities.append(float(np.mean(probabilities[indices])))
    return (
        np.asarray(subject_labels, dtype=int),
        np.asarray(subject_probabilities, dtype=float),
        ordered,
    )


def main() -> int:
    args = parse_args()
    matrix, labels, subjects = load_features(args.features)
    if not len(matrix):
        raise ValueError("feature CSV is empty")
    folds = subject_folds(labels, subjects, folds=args.folds, seed=args.seed)
    all_indices = np.arange(len(labels))
    out_of_fold = np.zeros(len(labels), dtype=float)
    fold_details = []
    for fold_number, validation_indices in enumerate(folds, 1):
        training_indices = np.setdiff1d(all_indices, validation_indices)
        fitted = fit_logistic(
            matrix[training_indices],
            labels[training_indices],
            l2=args.l2,
            steps=args.steps,
        )
        out_of_fold[validation_indices] = fitted.predict_probability(
            matrix[validation_indices]
        )
        fold_details.append(
            {
                "fold": fold_number,
                "training_subject_ids": sorted(
                    {subjects[index] for index in training_indices}
                ),
                "validation_subject_ids": sorted(
                    {subjects[index] for index in validation_indices}
                ),
                "validation_repetitions": int(len(validation_indices)),
            }
        )

    subject_labels, subject_probabilities, ordered_subjects = (
        aggregate_subject_probabilities(labels, out_of_fold, subjects)
    )
    threshold, _ = select_threshold(
        subject_labels,
        subject_probabilities,
        target_sensitivity=args.target_sensitivity,
    )
    subject_metrics = binary_metrics(subject_labels, subject_probabilities, threshold)
    repetition_metrics = binary_metrics(labels, out_of_fold, threshold)
    fitted_all = fit_logistic(matrix, labels, l2=args.l2, steps=args.steps)
    for detail in fold_details:
        validation_subjects = set(detail["validation_subject_ids"])
        selected = np.asarray(
            [subject in validation_subjects for subject in ordered_subjects]
        )
        detail["subject_metrics_at_global_threshold"] = binary_metrics(
            subject_labels[selected],
            subject_probabilities[selected],
            threshold,
        )
    strongest_features = sorted(
        (
            {
                "feature": name,
                "standardized_coefficient": float(coefficient),
            }
            for name, coefficient in zip(
                KINEMATIC_FEATURES, fitted_all.coefficients
            )
        ),
        key=lambda item: abs(item["standardized_coefficient"]),
        reverse=True,
    )[:12]
    subject_predictions = [
        {
            "subject_id": subject,
            "label": int(label),
            "out_of_fold_probability": float(probability),
        }
        for subject, label, probability in zip(
            ordered_subjects, subject_labels, subject_probabilities
        )
    ]
    summary = {
        "created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "dataset": "Lucchetti-Bailo-Lencioni kinematic and EMG dataset (2025)",
        "dataset_doi": "10.6084/m9.figshare.c.7720187.v1",
        "source_features": str(args.features),
        "target": "post-stroke plegic arm versus healthy dominant arm during functional tasks",
        "repetitions": int(len(labels)),
        "subjects": len(set(subjects)),
        "stroke_subjects": len({s for s, y in zip(subjects, labels) if y == 1}),
        "healthy_subjects": len({s for s, y in zip(subjects, labels) if y == 0}),
        "folds": args.folds,
        "seed": args.seed,
        "target_sensitivity": args.target_sensitivity,
        "threshold_selected_on": "mean out-of-fold probability per subject",
        "subject_out_of_fold_metrics": subject_metrics,
        "repetition_out_of_fold_metrics": repetition_metrics,
        "subject_out_of_fold_predictions": subject_predictions,
        "strongest_standardized_features": strongest_features,
        "fold_details": fold_details,
        "limitations": [
            "Research-only internal validation on 20 subjects; no external test cohort.",
            "Healthy and stroke groups differ in age distribution, creating confounding risk.",
            "Functional-task optical motion capture is not the BE-FAST bilateral arm-drift protocol.",
            "Do not use this artifact for diagnosis, triage, or automatic runtime deployment.",
        ],
    }
    export_model(
        args.output,
        component="A_REHAB_KINEMATICS_RESEARCH",
        model_version="kinematic-stroke-v1",
        feature_names=KINEMATIC_FEATURES,
        fitted=fitted_all,
        decision_threshold=threshold,
        training_summary=summary,
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "model": str(args.output),
                "threshold": threshold,
                "subject_metrics": subject_metrics,
                "repetition_metrics": repetition_metrics,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
