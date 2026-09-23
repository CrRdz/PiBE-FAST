#!/usr/bin/env python3
"""Group-validate, calibrate and export an MDSC dysarthria recognizer."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.speech_representation import SPEECH_REPRESENTATION_FEATURES, SPEECH_REPRESENTATION_VERSION
from training.modeling import binary_metrics, fit_logistic, select_threshold, subject_folds


def _auc(labels: np.ndarray, probabilities: np.ndarray) -> float:
    positives = probabilities[labels == 1]
    negatives = probabilities[labels == 0]
    if not len(positives) or not len(negatives):
        return 0.0
    comparisons = (positives[:, None] > negatives[None, :]).sum()
    ties = (positives[:, None] == negatives[None, :]).sum()
    return float((comparisons + 0.5 * ties) / (len(positives) * len(negatives)))


def _metrics(labels: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, float]:
    result = binary_metrics(labels, probabilities, threshold)
    result.update(
        {
            "roc_auc": _auc(labels, probabilities),
            "brier_score": float(np.mean((probabilities - labels) ** 2)),
            "samples": float(len(labels)),
        }
    )
    return result


def _speaker_average(labels: np.ndarray, probabilities: np.ndarray, subjects: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    unique = sorted(set(str(value) for value in subjects))
    speaker_labels, speaker_probabilities = [], []
    for subject in unique:
        indices = np.flatnonzero(subjects == subject)
        values = set(int(value) for value in labels[indices])
        if len(values) != 1:
            raise ValueError(f"speaker {subject} has mixed labels")
        speaker_labels.append(values.pop())
        speaker_probabilities.append(float(np.mean(probabilities[indices])))
    return np.asarray(speaker_labels), np.asarray(speaker_probabilities), np.asarray(unique)


def _bootstrap_intervals(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    *,
    seed: int,
    replicates: int,
) -> dict[str, list[float]]:
    """Speaker-level percentile intervals; invalid one-class draws are skipped."""

    rng = np.random.default_rng(seed)
    collected = {name: [] for name in ("sensitivity", "specificity", "balanced_accuracy", "roc_auc", "brier_score")}
    for _ in range(max(0, replicates)):
        indices = rng.integers(0, len(labels), size=len(labels))
        sampled_labels = labels[indices]
        if len(set(sampled_labels.tolist())) < 2:
            continue
        values = _metrics(sampled_labels, probabilities[indices], threshold)
        for name in collected:
            collected[name].append(values[name])
    return {
        name: [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]
        for name, values in collected.items()
        if values
    }


def _cluster_bootstrap_intervals(
    labels: np.ndarray,
    probabilities: np.ndarray,
    subjects: np.ndarray,
    threshold: float,
    *,
    seed: int,
    replicates: int,
) -> dict[str, list[float]]:
    """Recording-level intervals with speakers as the resampling unit."""

    rng = np.random.default_rng(seed)
    unique_subjects = np.asarray(sorted(set(str(value) for value in subjects)))
    subject_indices = {
        subject: np.flatnonzero(subjects == subject) for subject in unique_subjects
    }
    collected = {
        name: []
        for name in (
            "sensitivity",
            "specificity",
            "balanced_accuracy",
            "roc_auc",
            "brier_score",
        )
    }
    for _ in range(max(0, replicates)):
        sampled_subjects = rng.choice(
            unique_subjects, size=len(unique_subjects), replace=True
        )
        sampled_indices = np.concatenate(
            [subject_indices[str(subject)] for subject in sampled_subjects]
        )
        sampled_labels = labels[sampled_indices]
        if len(set(sampled_labels.tolist())) < 2:
            continue
        values = _metrics(
            sampled_labels, probabilities[sampled_indices], threshold
        )
        for name in collected:
            collected[name].append(values[name])
    return {
        name: [
            float(np.percentile(values, 2.5)),
            float(np.percentile(values, 97.5)),
        ]
        for name, values in collected.items()
        if values
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True, type=Path)
    parser.add_argument("--output", default=Path("models/mdsc_dysarthria_v1.json"), type=Path)
    parser.add_argument("--report", default=Path("training/reports/mdsc_dysarthria_v1.json"), type=Path)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-sensitivity", type=float, default=0.90)
    parser.add_argument("--l2", type=float, default=1.0)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    args = parser.parse_args()

    data = np.load(args.features, allow_pickle=False)
    matrix = np.asarray(data["matrix"], dtype=np.float64)
    labels = np.asarray(data["labels"], dtype=int)
    subjects = np.asarray(data["subjects"], dtype=str)
    splits = np.asarray(data["splits"], dtype=str)
    names = tuple(str(value) for value in data["feature_names"])
    if names != SPEECH_REPRESENTATION_FEATURES:
        raise ValueError("feature archive is incompatible with runtime representation")
    if matrix.shape != (len(labels), len(names)) or not np.isfinite(matrix).all():
        raise ValueError("invalid feature matrix")
    training = np.flatnonzero(splits == "train")
    validation = np.flatnonzero(splits == "validation")
    development = np.flatnonzero(np.isin(splits, ("train", "validation")))
    test = np.flatnonzero(splits == "test")
    if not len(training) or not len(validation) or not len(test):
        raise ValueError("features must contain train, validation and held-out test speakers")
    split_subjects = [set(subjects[indices]) for indices in (training, validation, test)]
    if any(split_subjects[left] & split_subjects[right] for left, right in ((0, 1), (0, 2), (1, 2))):
        raise ValueError("speaker leakage between train, validation and test")

    folds = subject_folds(labels[development], subjects[development], folds=args.folds, seed=args.seed)
    out_of_fold = np.zeros(len(development), dtype=np.float64)
    fold_details = []
    all_development = np.arange(len(development))
    for number, validation_local in enumerate(folds, 1):
        training_local = np.setdiff1d(all_development, validation_local)
        fitted = fit_logistic(
            matrix[development[training_local]],
            labels[development[training_local]],
            l2=args.l2,
            steps=args.steps,
        )
        out_of_fold[validation_local] = fitted.predict_probability(matrix[development[validation_local]])
        fold_details.append(
            {
                "fold": number,
                "training_speakers": len(set(subjects[development[training_local]])),
                "validation_speakers": len(set(subjects[development[validation_local]])),
                "validation_samples": len(validation_local),
            }
        )
    oof_speaker_labels, oof_speaker_probabilities, _ = _speaker_average(
        labels[development], out_of_fold, subjects[development]
    )
    oof_threshold, _ = select_threshold(
        labels[development],
        out_of_fold,
        target_sensitivity=args.target_sensitivity,
    )
    # Freeze the deployable model on train speakers, calibrate its threshold on
    # separate validation speakers, and touch test speakers only once.
    fitted_all = fit_logistic(matrix[training], labels[training], l2=args.l2, steps=args.steps)
    validation_probabilities = fitted_all.predict_probability(matrix[validation])
    validation_speaker_labels, validation_speaker_probabilities, _ = _speaker_average(
        labels[validation], validation_probabilities, subjects[validation]
    )
    threshold, threshold_selection_metrics = select_threshold(
        labels[validation],
        validation_probabilities,
        target_sensitivity=args.target_sensitivity,
    )
    test_probabilities = fitted_all.predict_probability(matrix[test])
    test_speaker_labels, test_speaker_probabilities, test_subjects = _speaker_average(
        labels[test], test_probabilities, subjects[test]
    )
    report = {
        "created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "dataset": "AISHELL-6B/MDSC",
        "license": "CC BY-NC 4.0",
        "task": "Mandarin dysarthria representation/recognition",
        "representation_version": SPEECH_REPRESENTATION_VERSION,
        "feature_archive": str(args.features),
        "feature_archive_sha256": hashlib.sha256(args.features.read_bytes()).hexdigest(),
        "development_samples": len(development),
        "development_speakers": len(set(subjects[development])),
        "training_samples": len(training),
        "training_speakers": len(set(subjects[training])),
        "validation_samples": len(validation),
        "validation_speakers": len(set(subjects[validation])),
        "test_samples": len(test),
        "test_speakers": len(set(subjects[test])),
        "folds": args.folds,
        "seed": args.seed,
        "target_sensitivity": args.target_sensitivity,
        "threshold_selection_unit": "recording",
        "threshold_selection_metrics": threshold_selection_metrics,
        "oof_decision_threshold": oof_threshold,
        "decision_threshold": threshold,
        "oof_sample_metrics": _metrics(labels[development], out_of_fold, oof_threshold),
        "oof_sample_cluster_95pct_intervals": _cluster_bootstrap_intervals(
            labels[development],
            out_of_fold,
            subjects[development],
            oof_threshold,
            seed=args.seed + 50,
            replicates=args.bootstrap_replicates,
        ),
        "oof_speaker_metrics": _metrics(oof_speaker_labels, oof_speaker_probabilities, oof_threshold),
        "oof_speaker_95pct_intervals": _bootstrap_intervals(
            oof_speaker_labels,
            oof_speaker_probabilities,
            oof_threshold,
            seed=args.seed + 100,
            replicates=args.bootstrap_replicates,
        ),
        "validation_sample_metrics": _metrics(labels[validation], validation_probabilities, threshold),
        "validation_sample_cluster_95pct_intervals": _cluster_bootstrap_intervals(
            labels[validation],
            validation_probabilities,
            subjects[validation],
            threshold,
            seed=args.seed + 150,
            replicates=args.bootstrap_replicates,
        ),
        "validation_speaker_metrics": _metrics(
            validation_speaker_labels, validation_speaker_probabilities, threshold
        ),
        "held_out_test_sample_metrics": _metrics(labels[test], test_probabilities, threshold),
        "held_out_test_sample_cluster_95pct_intervals": _cluster_bootstrap_intervals(
            labels[test],
            test_probabilities,
            subjects[test],
            threshold,
            seed=args.seed + 250,
            replicates=args.bootstrap_replicates,
        ),
        "held_out_test_speaker_metrics": _metrics(test_speaker_labels, test_speaker_probabilities, threshold),
        "held_out_test_speaker_95pct_intervals": _bootstrap_intervals(
            test_speaker_labels,
            test_speaker_probabilities,
            threshold,
            seed=args.seed + 200,
            replicates=args.bootstrap_replicates,
        ),
        "held_out_speakers": test_subjects.tolist(),
        "fold_details": fold_details,
        "warnings": [
            "Research-only internal MDSC validation; not clinical validation.",
            "The positive label represents chronic dysarthria in MDSC, not acute stroke or acute change.",
            "The deployable threshold is selected and evaluated at the recording level; uncertainty intervals resample speakers.",
            "Runtime inference contributes to the Speech screening status but is not an acute-stroke diagnosis.",
        ],
    }
    payload = {
        "schema_version": 1,
        "component": "speech_dysarthria_representation",
        "model_version": "mdsc-dysarthria-v1",
        "representation_version": SPEECH_REPRESENTATION_VERSION,
        "algorithm": "standardized_logistic_regression",
        "feature_names": list(names),
        "means": fitted_all.means.tolist(),
        "scales": fitted_all.scales.tolist(),
        "coefficients": fitted_all.coefficients.tolist(),
        "intercept": fitted_all.intercept,
        "decision_threshold": threshold,
        "medical_role": "dysarthria_speech_screening_component",
        "clinical_validation": False,
        "training_summary": report,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"model": str(args.output), **report["held_out_test_speaker_metrics"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
