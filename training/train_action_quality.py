#!/usr/bin/env python3
"""Train the IntelliRehabDS incorrect-action research model."""

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

from app.action_quality_model import ACTION_QUALITY_FEATURES
from training.modeling import binary_metrics, export_model, fit_logistic, select_threshold, subject_folds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, default=Path("training/processed/intellirehab_action_quality.csv"))
    parser.add_argument("--output", type=Path, default=Path("models/research/intellirehab_action_quality_v1.json"))
    parser.add_argument("--report", type=Path, default=Path("training/reports/intellirehab_action_quality_v1.json"))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-sensitivity", type=float, default=0.0,
        help="Optional minimum incorrect-action sensitivity; 0 selects maximum balanced accuracy")
    parser.add_argument("--steps", type=int, default=2500)
    return parser.parse_args()


def load(path: Path):
    matrix, labels, subjects = [], [], []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            subjects.append(row["subject_id"])
            labels.append(int(row["label"]))
            matrix.append([float(row[name]) for name in ACTION_QUALITY_FEATURES])
    return np.asarray(matrix), np.asarray(labels), subjects


def main() -> int:
    args = parse_args()
    matrix, labels, subjects = load(args.features)
    folds = subject_folds(labels, subjects, folds=args.folds, seed=args.seed)
    oof = np.zeros(len(labels))
    all_indices = np.arange(len(labels))
    fold_details = []
    for number, validation in enumerate(folds, 1):
        training = np.setdiff1d(all_indices, validation)
        fitted = fit_logistic(matrix[training], labels[training], steps=args.steps)
        oof[validation] = fitted.predict_probability(matrix[validation])
        fold_details.append({
            "fold": number,
            "training_subjects": len({subjects[i] for i in training}),
            "validation_subjects": len({subjects[i] for i in validation}),
            "validation_samples": len(validation),
        })
    if args.target_sensitivity > 0.0:
        threshold, _ = select_threshold(
            labels, oof, target_sensitivity=args.target_sensitivity
        )
    else:
        candidates = sorted(set(float(value) for value in oof))
        threshold = max(
            candidates,
            key=lambda value: binary_metrics(labels, oof, value)["balanced_accuracy"],
        )
    metrics = binary_metrics(labels, oof, threshold)
    fitted = fit_logistic(matrix, labels, steps=args.steps)
    summary = {
        "created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "dataset": "IntelliRehabDS 2.0.1",
        "dataset_doi": "10.5281/zenodo.4610859",
        "license": "CC BY 4.0",
        "gestures": {"4": "left shoulder abduction", "5": "right shoulder abduction"},
        "target": "incorrect_or_incomplete_action",
        "samples": len(labels),
        "subjects": len(set(subjects)),
        "incorrect_samples": int(labels.sum()),
        "correct_samples": int(len(labels) - labels.sum()),
        "folds": args.folds,
        "out_of_fold_metrics": metrics,
        "threshold_selection": (
            f"minimum sensitivity {args.target_sensitivity}"
            if args.target_sensitivity > 0.0
            else "maximum subject-grouped balanced accuracy"
        ),
        "fold_details": fold_details,
        "deployment": "shadow_only_until_validated_on_webcam_bilateral_protocol",
        "warning": "Internal subject-grouped validation; not a stroke diagnostic model.",
    }
    export_model(
        args.output,
        component="A_ACTION_QUALITY_SHADOW",
        model_version="intellirehab-action-quality-v1",
        feature_names=ACTION_QUALITY_FEATURES,
        fitted=fitted,
        decision_threshold=threshold,
        training_summary=summary,
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"model": str(args.output), "threshold": threshold, **metrics}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
