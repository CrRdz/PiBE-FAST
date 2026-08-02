#!/usr/bin/env python3
"""Train three subject-grouped Toronto compensation shadow models."""

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

from app.compensation_model import COMPENSATION_MODEL_FEATURES, COMPENSATION_TARGETS
from training.modeling import binary_metrics, export_model, fit_logistic, select_threshold


TARGET_LABELS = {"lean_forward": 2, "shoulder_elevation": 3, "trunk_rotation": 4}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--features",
        type=Path,
        default=Path("training/processed/toronto_compensation.csv"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("models/research")
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("training/reports/toronto_compensation_v1.json"),
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-sensitivity", type=float, default=0.80)
    parser.add_argument("--l2", type=float, default=1.0)
    parser.add_argument("--steps", type=int, default=1800)
    return parser.parse_args()


def load_features(
    path: Path,
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    matrix, source_labels, subjects, cohorts = [], [], [], []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"subject_id", "cohort", "source_label", *COMPENSATION_MODEL_FEATURES}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"feature CSV is missing: {', '.join(sorted(missing))}")
        for row in reader:
            subjects.append(str(row["subject_id"]))
            cohorts.append(str(row["cohort"]))
            source_labels.append(int(row["source_label"]))
            matrix.append([float(row[name]) for name in COMPENSATION_MODEL_FEATURES])
    if not matrix:
        raise ValueError("feature CSV is empty")
    return (
        np.asarray(matrix, dtype=float),
        np.asarray(source_labels, dtype=int),
        subjects,
        cohorts,
    )


def cohort_group_folds(
    subjects: list[str], cohorts: list[str], *, folds: int, seed: int
) -> list[np.ndarray]:
    """Keep people intact while distributing both cohorts across folds."""

    grouped: dict[str, list[int]] = {}
    subject_cohort: dict[str, str] = {}
    for index, (subject, cohort) in enumerate(zip(subjects, cohorts)):
        grouped.setdefault(subject, []).append(index)
        previous = subject_cohort.setdefault(subject, cohort)
        if previous != cohort:
            raise ValueError(f"subject {subject} occurs in multiple cohorts")
    cohort_subjects: dict[str, list[str]] = {}
    for subject, cohort in subject_cohort.items():
        cohort_subjects.setdefault(cohort, []).append(subject)
    if any(len(values) < folds for values in cohort_subjects.values()):
        raise ValueError("each cohort needs at least one subject per fold")
    rng = np.random.default_rng(seed)
    selected: list[set[str]] = [set() for _ in range(folds)]
    fold_sizes = [0] * folds
    for cohort in sorted(cohort_subjects):
        values = cohort_subjects[cohort][:]
        rng.shuffle(values)
        values.sort(key=lambda subject: len(grouped[subject]), reverse=True)
        cohort_counts = [0] * folds
        for subject in values:
            destination = min(
                range(folds),
                key=lambda index: (cohort_counts[index], fold_sizes[index]),
            )
            selected[destination].add(subject)
            cohort_counts[destination] += 1
            fold_sizes[destination] += len(grouped[subject])
    return [
        np.asarray(
            [index for index, subject in enumerate(subjects) if subject in fold_subjects],
            dtype=int,
        )
        for fold_subjects in selected
    ]


def _cohort_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    cohorts: list[str],
    threshold: float,
) -> dict[str, dict[str, float]]:
    output = {"all": binary_metrics(labels, probabilities, threshold)}
    cohort_array = np.asarray(cohorts)
    for cohort in sorted(set(cohorts)):
        selected = cohort_array == cohort
        output[cohort] = binary_metrics(
            labels[selected], probabilities[selected], threshold
        )
    return output


def main() -> int:
    args = parse_args()
    matrix, source_labels, subjects, cohorts = load_features(args.features)
    folds = cohort_group_folds(subjects, cohorts, folds=args.folds, seed=args.seed)
    all_indices = np.arange(len(source_labels))
    created_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    report: dict[str, object] = {
        "created_at": created_at,
        "dataset": "Toronto Rehab Stroke Pose Dataset",
        "source_features": str(args.features),
        "subjects": len(set(subjects)),
        "healthy_subjects": len({s for s in subjects if s.startswith("H")}),
        "stroke_subjects": len({s for s in subjects if s.startswith("P")}),
        "windows": int(len(source_labels)),
        "folds": args.folds,
        "seed": args.seed,
        "feature_policy": (
            "2-D shoulders, elbows, wrists and hips only; Kinect depth excluded"
        ),
        "deployment_mode": "shadow_only",
        "targets": {},
        "limitations": [
            "Labels describe rehabilitation compensation, not acute stroke diagnosis.",
            "Healthy positives are scripted simulations; patient positives are sparse and imbalanced.",
            "Kinect-to-MoveNet camera-domain shift has not been externally validated.",
            "Shadow probabilities must not change BE-FAST results or user guidance.",
        ],
    }
    for target in COMPENSATION_TARGETS:
        labels = np.asarray(source_labels == TARGET_LABELS[target], dtype=int)
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
                    "validation_subject_ids": sorted(
                        {subjects[index] for index in validation_indices}
                    ),
                    "validation_windows": int(len(validation_indices)),
                }
            )
        threshold, _ = select_threshold(
            labels, out_of_fold, target_sensitivity=args.target_sensitivity
        )
        metrics = _cohort_metrics(labels, out_of_fold, cohorts, threshold)
        for detail, validation_indices in zip(fold_details, folds):
            detail["metrics_at_global_threshold"] = binary_metrics(
                labels[validation_indices],
                out_of_fold[validation_indices],
                threshold,
            )
        fitted_all = fit_logistic(matrix, labels, l2=args.l2, steps=args.steps)
        strongest = sorted(
            (
                {
                    "feature": name,
                    "standardized_coefficient": float(coefficient),
                }
                for name, coefficient in zip(
                    COMPENSATION_MODEL_FEATURES, fitted_all.coefficients
                )
            ),
            key=lambda item: abs(item["standardized_coefficient"]),
            reverse=True,
        )[:12]
        target_summary = {
            "source_positive_label": TARGET_LABELS[target],
            "positive_windows": int(labels.sum()),
            "negative_windows": int(len(labels) - labels.sum()),
            "threshold": threshold,
            "out_of_fold_metrics": metrics,
            "strongest_standardized_features": strongest,
            "fold_details": fold_details,
        }
        report["targets"][target] = target_summary  # type: ignore[index]
        export_model(
            args.output_dir / f"toronto_{target}_v1.json",
            component=f"A_COMPENSATION_SHADOW_{target.upper()}",
            model_version=f"toronto-{target}-v1",
            feature_names=COMPENSATION_MODEL_FEATURES,
            fitted=fitted_all,
            decision_threshold=threshold,
            training_summary={
                "created_at": created_at,
                "dataset": report["dataset"],
                "target": target,
                "deployment_mode": "shadow_only",
                **target_summary,
            },
        )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                target: {
                    "threshold": values["threshold"],
                    "all": values["out_of_fold_metrics"]["all"],
                    "stroke_patient": values["out_of_fold_metrics"]["stroke_patient"],
                }
                for target, values in report["targets"].items()  # type: ignore[union-attr]
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
