#!/usr/bin/env python3
"""Train, group-validate, calibrate, and export the A screening model."""

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

from app.arm_model import ARM_MODEL_FEATURES
from training.modeling import (
    binary_metrics,
    export_model,
    fit_logistic,
    select_threshold,
    subject_folds,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True, type=Path)
    parser.add_argument("--output", default=Path("models/arm_screen_v2.json"), type=Path)
    parser.add_argument("--report", default=Path("training/reports/arm_screen_v2.json"), type=Path)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-sensitivity", type=float, default=0.90)
    parser.add_argument("--l2", type=float, default=0.1)
    parser.add_argument("--steps", type=int, default=2500)
    return parser.parse_args()


def load_features(path: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    matrix, labels, subjects = [], [], []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"subject_id", "label", *ARM_MODEL_FEATURES}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"feature CSV is missing: {', '.join(sorted(missing))}")
        for row in reader:
            subjects.append(str(row["subject_id"]))
            labels.append(int(row["label"]))
            matrix.append([float(row[name]) for name in ARM_MODEL_FEATURES])
    if not matrix:
        raise ValueError("feature CSV is empty")
    return np.asarray(matrix, dtype=float), np.asarray(labels, dtype=int), subjects


def main() -> int:
    args = parse_args()
    if not 0.0 < args.target_sensitivity <= 1.0:
        raise ValueError("target sensitivity must be in (0, 1]")
    matrix, labels, subjects = load_features(args.features)
    validation_folds = subject_folds(
        labels,
        subjects,
        folds=args.folds,
        seed=args.seed,
    )
    out_of_fold = np.zeros(len(labels), dtype=float)
    fold_reports = []
    all_indices = np.arange(len(labels))
    for fold_number, validation_indices in enumerate(validation_folds, 1):
        training_indices = np.setdiff1d(all_indices, validation_indices)
        fitted = fit_logistic(
            matrix[training_indices],
            labels[training_indices],
            l2=args.l2,
            steps=args.steps,
        )
        probabilities = fitted.predict_probability(matrix[validation_indices])
        out_of_fold[validation_indices] = probabilities
        fold_reports.append(
            {
                "fold": fold_number,
                "training_subjects": len({subjects[index] for index in training_indices}),
                "validation_subjects": len({subjects[index] for index in validation_indices}),
                "validation_samples": int(len(validation_indices)),
            }
        )
    threshold, metrics = select_threshold(
        labels,
        out_of_fold,
        target_sensitivity=args.target_sensitivity,
    )
    metrics = binary_metrics(labels, out_of_fold, threshold)
    fitted_all = fit_logistic(
        matrix,
        labels,
        l2=args.l2,
        steps=args.steps,
    )
    timestamp = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    summary = {
        "created_at": timestamp,
        "source_features": str(args.features),
        "samples": int(len(labels)),
        "subjects": len(set(subjects)),
        "positive_samples": int(labels.sum()),
        "negative_samples": int(len(labels) - labels.sum()),
        "folds": args.folds,
        "seed": args.seed,
        "target_sensitivity": args.target_sensitivity,
        "out_of_fold_metrics": metrics,
        "fold_details": fold_reports,
        "warning": "Research model only; metrics are subject-grouped internal validation, not clinical validation.",
    }
    export_model(
        args.output,
        component="A",
        model_version="arm-screen-v2",
        feature_names=ARM_MODEL_FEATURES,
        fitted=fitted_all,
        decision_threshold=threshold,
        training_summary=summary,
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"model": str(args.output), "threshold": threshold, **metrics}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
