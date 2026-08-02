"""Shared arm-sequence features and lightweight linear-model inference.

The training pipeline and Raspberry Pi runtime intentionally import this same
module.  Keeping feature names and aggregation in one place prevents a trained
model from silently receiving different features after deployment.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from math import exp
from pathlib import Path
from statistics import median, pstdev
from typing import Mapping, Sequence

from app.compensation_model import (
    COMPENSATION_MODEL_FEATURES,
    summarize_compensation_frames,
)


ArmSample = tuple[float, float, float]


ARM_MODEL_FEATURES_V1 = (
    "level_difference_median",
    "initial_left_wrist_relative_y",
    "initial_right_wrist_relative_y",
    "final_left_wrist_relative_y",
    "final_right_wrist_relative_y",
    "left_drop",
    "right_drop",
    "drift_difference",
    "left_variability",
    "right_variability",
    "level_difference_variability",
    "max_abs_level_difference",
)
ARM_MODEL_FEATURES_V2 = (*ARM_MODEL_FEATURES_V1, *COMPENSATION_MODEL_FEATURES)
# Public alias used by current training/runtime code. V1 remains available for
# loading and auditing older exported models.
ARM_MODEL_FEATURES = ARM_MODEL_FEATURES_V2


def summarize_arm_samples(
    samples: Sequence[ArmSample],
    *,
    rich_frames: Sequence[Mapping[str, float]] | None = None,
) -> dict[str, float]:
    """Aggregate a valid arm-hold sequence into stable, deployable features."""

    if not samples:
        raise ValueError("arm samples must not be empty")
    segment = max(1, len(samples) // 3)
    first = samples[:segment]
    last = samples[-segment:]
    left_values = [float(sample[0]) for sample in samples]
    right_values = [float(sample[1]) for sample in samples]
    level_values = [float(sample[2]) for sample in samples]
    initial_left = median(float(sample[0]) for sample in first)
    initial_right = median(float(sample[1]) for sample in first)
    final_left = median(float(sample[0]) for sample in last)
    final_right = median(float(sample[1]) for sample in last)
    left_drop = final_left - initial_left
    right_drop = final_right - initial_right
    output = {
        "level_difference_median": median(level_values),
        "initial_left_wrist_relative_y": initial_left,
        "initial_right_wrist_relative_y": initial_right,
        "final_left_wrist_relative_y": final_left,
        "final_right_wrist_relative_y": final_right,
        "left_drop": left_drop,
        "right_drop": right_drop,
        "drift_difference": left_drop - right_drop,
        "left_variability": pstdev(left_values),
        "right_variability": pstdev(right_values),
        "level_difference_variability": pstdev(level_values),
        "max_abs_level_difference": max(abs(value) for value in level_values),
    }
    # V2 retains every V1 feature and appends the same body-centred geometry
    # used by Toronto training. Zeros only preserve legacy unit-test/caller
    # compatibility; runtime V2 inference is gated on real rich frames.
    if rich_frames:
        output.update(summarize_compensation_frames(rich_frames))
    else:
        output.update({name: 0.0 for name in COMPENSATION_MODEL_FEATURES})
    return output


@dataclass(frozen=True)
class LinearBinaryModel:
    """A standardized logistic model stored as a small, auditable JSON file."""

    component: str
    version: str
    feature_names: tuple[str, ...]
    means: tuple[float, ...]
    scales: tuple[float, ...]
    coefficients: tuple[float, ...]
    intercept: float
    decision_threshold: float

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        expected_component: str | None = None,
    ) -> "LinearBinaryModel":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if int(payload.get("schema_version", 0)) != 1:
            raise ValueError("unsupported learned-model schema version")
        component = str(payload.get("component", ""))
        if expected_component is not None and component != expected_component:
            raise ValueError(
                f"expected component {expected_component!r}, got {component!r}"
            )
        feature_names = tuple(str(name) for name in payload["feature_names"])
        means = tuple(float(value) for value in payload["means"])
        scales = tuple(float(value) for value in payload["scales"])
        coefficients = tuple(float(value) for value in payload["coefficients"])
        size = len(feature_names)
        if not size or len(means) != size or len(scales) != size or len(coefficients) != size:
            raise ValueError("learned-model vectors must have matching non-zero lengths")
        if any(value <= 0.0 for value in scales):
            raise ValueError("learned-model scales must be positive")
        threshold = float(payload["decision_threshold"])
        if not 0.0 < threshold < 1.0:
            raise ValueError("decision threshold must be between zero and one")
        return cls(
            component=component,
            version=str(payload.get("model_version", "unknown")),
            feature_names=feature_names,
            means=means,
            scales=scales,
            coefficients=coefficients,
            intercept=float(payload["intercept"]),
            decision_threshold=threshold,
        )

    def predict_probability(self, features: Mapping[str, float]) -> float:
        missing = [name for name in self.feature_names if name not in features]
        if missing:
            raise ValueError(f"missing model features: {', '.join(missing)}")
        logit = self.intercept
        for name, mean, scale, coefficient in zip(
            self.feature_names,
            self.means,
            self.scales,
            self.coefficients,
        ):
            standardized = (float(features[name]) - mean) / scale
            logit += standardized * coefficient
        # Numerically stable sigmoid for unusually large feature values.
        if logit >= 0.0:
            return 1.0 / (1.0 + exp(-logit))
        exp_logit = exp(logit)
        return exp_logit / (1.0 + exp_logit)


def load_optional_arm_model(path: str | Path) -> LinearBinaryModel | None:
    """Load a deployed A model, returning ``None`` for an absent model file."""

    model_path = Path(path)
    if not model_path.is_file():
        return None
    return LinearBinaryModel.load(model_path, expected_component="A")
