#!/usr/bin/env python3
"""Generate reproducible calibration and repeatability figures for the paper."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.clip(values, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-values))


def _fit_logistic_calibration(
    labels: np.ndarray, logits: np.ndarray, *, fit_slope: bool
) -> tuple[float, float]:
    """Fit y ~ intercept + slope * logit(p) with Newton iterations."""

    design = (
        np.column_stack((np.ones(len(logits)), logits))
        if fit_slope
        else np.ones((len(logits), 1))
    )
    offset = np.zeros(len(logits)) if fit_slope else logits
    beta = np.asarray([0.0, 1.0] if fit_slope else [0.0], dtype=float)
    for _ in range(100):
        eta = offset + design @ beta
        probability = _sigmoid(eta)
        weight = np.maximum(probability * (1.0 - probability), 1e-8)
        gradient = design.T @ (labels - probability)
        information = (design.T * weight) @ design
        step = np.linalg.solve(information + np.eye(len(beta)) * 1e-9, gradient)
        beta += step
        if float(np.max(np.abs(step))) < 1e-10:
            break
    return (float(beta[0]), float(beta[1])) if fit_slope else (float(beta[0]), 1.0)


def _quantile_bins(
    labels: np.ndarray, probabilities: np.ndarray, bins: int
) -> list[dict[str, float | int]]:
    groups = np.array_split(np.argsort(probabilities), bins)
    return [
        {
            "count": int(len(group)),
            "mean_predicted": float(np.mean(probabilities[group])),
            "observed_fraction": float(np.mean(labels[group])),
        }
        for group in groups
        if len(group)
    ]


def _calibration_statistics(
    labels: np.ndarray, probabilities: np.ndarray
) -> dict[str, float]:
    clipped = np.clip(probabilities, 1e-6, 1.0 - 1e-6)
    logits = np.log(clipped / (1.0 - clipped))
    intercept, _ = _fit_logistic_calibration(labels, logits, fit_slope=False)
    joint_intercept, slope = _fit_logistic_calibration(labels, logits, fit_slope=True)
    prevalence = float(np.mean(labels))
    brier = float(np.mean((probabilities - labels) ** 2))
    reference = prevalence * (1.0 - prevalence)
    return {
        "calibration_in_the_large": intercept,
        "calibration_model_intercept": joint_intercept,
        "calibration_slope": slope,
        "mean_predicted_probability": float(np.mean(probabilities)),
        "observed_prevalence": prevalence,
        "brier_score": brier,
        "scaled_brier_score": float(1.0 - brier / reference) if reference else 0.0,
    }


def _cluster_intervals(
    labels: np.ndarray,
    probabilities: np.ndarray,
    subjects: np.ndarray,
    *,
    seed: int,
    replicates: int,
) -> dict[str, list[float]]:
    rng = np.random.default_rng(seed)
    unique = np.asarray(sorted(set(subjects.tolist())))
    indices = {subject: np.flatnonzero(subjects == subject) for subject in unique}
    values = {"calibration_in_the_large": [], "brier_score": []}
    for _ in range(replicates):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        rows = np.concatenate([indices[subject] for subject in sampled])
        if len(set(labels[rows].tolist())) < 2:
            continue
        try:
            estimate = _calibration_statistics(labels[rows], probabilities[rows])
        except np.linalg.LinAlgError:
            continue
        if all(math.isfinite(estimate[name]) for name in values):
            for name in values:
                values[name].append(estimate[name])
    return {
        name: [float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))]
        for name, samples in values.items()
        if samples
    }


def generate_calibration(
    features_path: Path,
    model_path: Path,
    figure_path: Path,
    summary_path: Path,
) -> None:
    data = np.load(features_path, allow_pickle=False)
    model = json.loads(model_path.read_text(encoding="utf-8"))
    matrix = np.asarray(data["matrix"], dtype=float)
    labels = np.asarray(data["labels"], dtype=int)
    subjects = np.asarray(data["subjects"], dtype=str)
    splits = np.asarray(data["splits"], dtype=str)
    names = [str(value) for value in data["feature_names"]]
    if names != model["feature_names"]:
        raise ValueError("feature archive and model feature names differ")
    test = np.flatnonzero(splits == "test")
    standardized = (matrix[test] - np.asarray(model["means"])) / np.asarray(model["scales"])
    probabilities = _sigmoid(
        standardized @ np.asarray(model["coefficients"]) + float(model["intercept"])
    )
    test_labels = labels[test]
    test_subjects = subjects[test]
    bins = _quantile_bins(test_labels, probabilities, 10)
    summary = {
        "analysis": "held_out_mdsc_recording_level_calibration",
        "model_version": model["model_version"],
        "decision_threshold": float(model["decision_threshold"]),
        "feature_archive": str(features_path),
        "interpretation": (
            "Descriptive internal-corpus calibration only; the effective independent "
            "test sample is six speakers and the label is chronic dysarthria, not acute stroke."
        ),
        "recordings": int(len(test)),
        "speakers": int(len(set(test_subjects.tolist()))),
        **_calibration_statistics(test_labels, probabilities),
        "speaker_cluster_bootstrap_95pct": _cluster_intervals(
            test_labels,
            probabilities,
            test_subjects,
            seed=20260903,
            replicates=5000,
        ),
        "calibration_parameter_interval_note": (
            "Speaker-clustered intervals for the joint calibration intercept and "
            "slope are not reported because six test speakers produce frequent "
            "separation and non-identifiable bootstrap fits."
        ),
        "equal_frequency_bins": bins,
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    plt.rcParams.update({"font.size": 8, "font.family": "DejaVu Sans"})
    fig, (ax_curve, ax_hist) = plt.subplots(
        2, 1, figsize=(3.45, 3.25), gridspec_kw={"height_ratios": [2.2, 0.8]}
    )
    ax_curve.plot([0, 1], [0, 1], "--", color="#777777", linewidth=1, label="Ideal")
    ax_curve.plot(
        [row["mean_predicted"] for row in bins],
        [row["observed_fraction"] for row in bins],
        "o-",
        color="#1565C0",
        linewidth=1.4,
        markersize=3.5,
        label="10 equal-frequency bins",
    )
    ax_curve.set(xlim=(0, 1), ylim=(0, 1), ylabel="Observed fraction")
    ax_curve.grid(alpha=0.22)
    ax_curve.legend(loc="upper left", frameon=False, fontsize=7)
    ax_curve.text(
        0.98,
        0.04,
        f"CITL={summary['calibration_in_the_large']:.2f}\n"
        f"Slope={summary['calibration_slope']:.2f}\n"
        f"Brier={summary['brier_score']:.3f}",
        ha="right",
        va="bottom",
        fontsize=7,
    )
    ax_hist.hist(probabilities[test_labels == 0], bins=np.linspace(0, 1, 16), alpha=0.65, label="Control", color="#4C78A8")
    ax_hist.hist(probabilities[test_labels == 1], bins=np.linspace(0, 1, 16), alpha=0.55, label="Dysarthric", color="#E45756")
    ax_hist.set(xlim=(0, 1), xlabel="Predicted probability", ylabel="Count")
    ax_hist.legend(frameon=False, fontsize=7, ncol=2)
    fig.tight_layout(pad=0.55)
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path, bbox_inches="tight")
    plt.close(fig)


def generate_repeatability(report_path: Path, figure_path: Path) -> None:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    modules = ["B", "E", "F", "A"]
    estimates = [report["modules"][module]["icc_1_1"] for module in modules]
    intervals = [
        report["modules"][module]["icc_1_1_participant_bootstrap_95pct"]
        for module in modules
    ]
    errors = np.asarray(
        [[value - interval[0] for value, interval in zip(estimates, intervals)],
         [interval[1] - value for value, interval in zip(estimates, intervals)]]
    )
    fig, ax = plt.subplots(figsize=(3.45, 1.95))
    positions = np.arange(len(modules))[::-1]
    ax.errorbar(
        estimates,
        positions,
        xerr=errors,
        fmt="o",
        color="#1565C0",
        ecolor="#4C78A8",
        capsize=3,
        linewidth=1.3,
    )
    ax.axvspan(0.0, 0.5, color="#E45756", alpha=0.08)
    ax.axvline(0.5, color="#777777", linestyle="--", linewidth=0.8)
    ax.set(yticks=positions, yticklabels=modules, xlim=(0, 1), xlabel="ICC(1,1), participant-bootstrap 95% interval")
    ax.grid(axis="x", alpha=0.22)
    fig.tight_layout(pad=0.55)
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, default=Path("training/processed/mdsc_features.npz"))
    parser.add_argument("--model", type=Path, default=Path("models/mdsc_dysarthria_v1.json"))
    parser.add_argument("--repeatability-report", type=Path, default=Path("research_data/human_repeatability/analysis_report.json"))
    parser.add_argument("--figure-dir", type=Path, default=Path("docs/pibefast-paper/figures"))
    parser.add_argument("--summary", type=Path, default=Path("experiments/mdsc_calibration/summary.json"))
    args = parser.parse_args()
    generate_calibration(args.features, args.model, args.figure_dir / "mdsc-calibration.pdf", args.summary)
    generate_repeatability(args.repeatability_report, args.figure_dir / "repeatability-icc.pdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
