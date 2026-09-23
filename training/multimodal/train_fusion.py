"""Train an exploratory 48-feature baseline; not a clinical validation pipeline."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.optimize import brentq, minimize

from app.befast.fusion import FUSION_MODEL_FEATURE_NAMES


def _fit(x: np.ndarray, y: np.ndarray, penalty: float) -> np.ndarray:
    design = np.column_stack((np.ones(len(x)), x))

    def objective(beta: np.ndarray) -> tuple[float, np.ndarray]:
        logits = design @ beta
        loss = np.logaddexp(0.0, logits).sum() - y @ logits
        loss += 0.5 * penalty * np.dot(beta[1:], beta[1:])
        probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits, -40, 40)))
        gradient = design.T @ (probabilities - y)
        gradient[1:] += penalty * beta[1:]
        return float(loss), gradient

    result = minimize(
        objective, np.zeros(design.shape[1]), jac=True, method="L-BFGS-B"
    )
    if not result.success:
        raise RuntimeError(f"fusion optimization failed: {result.message}")
    return np.asarray(result.x)


def _auc(y: np.ndarray, p: np.ndarray) -> float:
    positive = p[y == 1]
    negative = p[y == 0]
    if not len(positive) or not len(negative):
        raise ValueError("evaluation split must contain both labels")
    return float(
        np.mean(positive[:, None] > negative[None, :])
        + 0.5 * np.mean(positive[:, None] == negative[None, :])
    )


def _group_folds(groups: np.ndarray, folds: int = 5) -> list[np.ndarray]:
    unique = np.unique(groups)
    if len(unique) < 2:
        raise ValueError("development data require at least two participants")
    return [np.isin(groups, part) for part in np.array_split(unique, min(folds, len(unique)))]


def _standardize(train: np.ndarray, other: np.ndarray):
    means, scales = train.mean(axis=0), train.std(axis=0)
    scales[scales < 1e-8] = 1.0
    return (train-means)/scales, (other-means)/scales, means, scales


def _participant_predictions(labels, probabilities, groups):
    y, p = [], []
    for group in np.unique(groups):
        selected = groups == group
        if len(np.unique(labels[selected])) != 1:
            raise ValueError("participant aggregation requires a single adjudicated outcome per participant")
        y.append(labels[selected][0]); p.append(probabilities[selected].mean())
    return np.asarray(y), np.asarray(p)


def _calibration(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    """Return descriptive external-test calibration intercept and slope."""

    logits = np.log(np.clip(p, 1e-8, 1 - 1e-8) / np.clip(1 - p, 1e-8, 1))

    def loss(beta: np.ndarray) -> tuple[float, np.ndarray]:
        fitted = beta[0] + beta[1] * logits
        probabilities = 1.0 / (1.0 + np.exp(-np.clip(fitted, -40, 40)))
        value = np.logaddexp(0.0, fitted).sum() - y @ fitted
        gradient = np.asarray(
            [np.sum(probabilities - y), np.dot(probabilities - y, logits)]
        )
        return float(value), gradient

    fitted = minimize(loss, np.asarray([0.0, 1.0]), jac=True, method="L-BFGS-B")
    intercept = brentq(
        lambda value: float(np.mean(1.0 / (1.0 + np.exp(-np.clip(logits + value, -40, 40)))) - np.mean(y)),
        -40.0,
        40.0,
    )
    return {
        "intercept": float(fitted.x[0]),
        "slope": float(fitted.x[1]),
        "calibration_in_the_large": float(intercept),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("--external-site", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    with args.csv_path.open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    required = {"session_id", "participant_id", "site_id", "label", *FUSION_MODEL_FEATURE_NAMES}
    missing = required.difference(rows[0].keys() if rows else ())
    if missing:
        raise ValueError(f"missing CSV columns: {sorted(missing)}")
    if any(not row[k].strip() for row in rows for k in ("session_id", "participant_id", "site_id")):
        raise ValueError("session, participant and site identifiers must be nonempty")
    if len({row['session_id'] for row in rows}) != len(rows):
        raise ValueError("duplicate session_id")
    participants = np.asarray([row["participant_id"] for row in rows])
    sites = np.asarray([row["site_id"] for row in rows])
    labels = np.asarray([int(row["label"]) for row in rows], dtype=float)
    if not set(np.unique(labels)).issubset({0.0, 1.0}):
        raise ValueError("label must contain only 0 and 1")
    for participant in np.unique(participants):
        if len(np.unique(sites[participants == participant])) != 1:
            raise ValueError(f"participant {participant} occurs at multiple sites")
    x = np.asarray(
        [[float(row[name]) for name in FUSION_MODEL_FEATURE_NAMES] for row in rows]
    )
    if not np.isfinite(x).all():
        raise ValueError("features must be finite; use explicit missing indicators")

    external = sites == args.external_site
    if not external.any() or external.all():
        raise ValueError("external site must leave nonempty development and test sets")
    x_dev, x_test = x[~external], x[external]
    y_dev, y_test = labels[~external], labels[external]
    groups = participants[~external]
    for y in (y_dev, y_test):
        if len(np.unique(y)) != 2:
            raise ValueError("development and external test each require both labels")
    _participant_predictions(labels, np.zeros(len(labels)), participants)
    xd, xt, means, scales = _standardize(x_dev, x_test)

    penalties = (0.01, 0.1, 1.0, 10.0)
    folds = _group_folds(groups)
    losses: dict[float, float] = {}
    for penalty in penalties:
        fold_losses = []
        for validation in folds:
            if validation.all() or not validation.any():
                continue
            if len(np.unique(y_dev[~validation])) != 2:
                raise ValueError("each training fold requires both labels; revise the development split")
            fold_train, fold_val, _, _ = _standardize(x_dev[~validation], x_dev[validation])
            beta = _fit(fold_train, y_dev[~validation], penalty)
            logits = np.column_stack((np.ones(validation.sum()), fold_val)) @ beta
            probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits, -40, 40)))
            group_y, group_p = _participant_predictions(y_dev[validation], probabilities, groups[validation])
            fold_losses.extend(((group_p - group_y) ** 2).tolist())
        losses[penalty] = float(np.mean(fold_losses))
    selected = min(losses, key=losses.get)
    beta = _fit(xd, y_dev, selected)
    probabilities = 1.0 / (
        1.0 + np.exp(-np.clip(np.column_stack((np.ones(len(xt)), xt)) @ beta, -40, 40))
    )
    participant_y, participant_p = _participant_predictions(y_test, probabilities, participants[external])
    payload = {
        "model_role": "research_only_multimodal_fusion_not_safety_authority",
        "feature_names": list(FUSION_MODEL_FEATURE_NAMES),
        "external_site": args.external_site,
        "development_participants": int(len(np.unique(groups))),
        "test_participants": int(len(np.unique(participants[external]))),
        "selected_l2_penalty": selected,
        "development_cv_brier": losses,
        "evaluation_unit": "participant_mean_probability_single_outcome",
        "protocol_limits": "single grouped tuning loop; no nested assessment, probability recalibration, confidence intervals, or automatic clinical-label verification",
        "test_auc": _auc(participant_y, participant_p),
        "test_brier": float(np.mean((participant_p - participant_y) ** 2)),
        "test_calibration": _calibration(participant_y, participant_p),
        "means": means.tolist(),
        "scales": scales.tolist(),
        "intercept": float(beta[0]),
        "coefficients": beta[1:].tolist(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
