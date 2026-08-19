"""Quality-gated, feature-level BE-FAST fusion for research observation.

This module deliberately produces an inspectable feature vector rather than a
diagnostic prediction.  It preserves the established component decisions and
the T safety rule: callers may log or train on this vector, but it must not
replace ``screening_decision`` until it has been validated and calibrated with
an appropriate subject-level dataset.
"""

from __future__ import annotations

from math import isfinite
from typing import Any, Mapping

from .config import BefastConfig


FEATURE_FUSION_VERSION = "befast-feature-fusion-v1"
COMPONENTS = ("B", "E", "F", "A", "S")

# The vector is deliberately fixed-length.  Missing modality values are zeroed
# only after their matching missing mask and quality values are appended.
FUSION_FEATURE_NAMES = (
    "B_trunk_orientation_change_score",
    "B_mediolateral_sway_change_score",
    "E_left_gaze_range",
    "E_right_gaze_range",
    "E_inter_eye_range_asymmetry",
    "E_binocular_directional_asymmetry",
    "E_max_conjugacy_error",
    "E_conjugate_rest_gaze_deviation_degrees",
    "F_mouth_corner_delta",
    "F_smile_score_difference",
    "F_smile_strength",
    "A_level_difference",
    "A_left_drop",
    "A_right_drop",
    "A_drift_difference",
    "S_character_error_rate",
    "S_characters_per_second",
    "S_pause_fraction",
)
FUSION_MODEL_FEATURE_NAMES = (
    *FUSION_FEATURE_NAMES,
    *(f"quality_{component}" for component in COMPONENTS),
    *(f"missing_{component}" for component in COMPONENTS),
)


def build_feature_fusion(
    items: Mapping[str, Mapping[str, Any]],
    config: BefastConfig,
) -> dict[str, Any]:
    """Build a fixed, quality-gated feature vector from completed report items.

    ``raw_features`` retains continuous metrics as emitted by each component.
    ``gated_features`` multiplies every domain by its quality value.  Consumers
    must retain ``quality`` and ``missing`` alongside the vector so zero remains
    distinguishable from an unavailable observation.
    """

    raw_features = _raw_features(items)
    available = {
        component: _has_measurement(items.get(component, {}))
        for component in COMPONENTS
    }
    quality = {
        component: _quality(component, items.get(component, {}), config)
        if available[component]
        else 0.0
        for component in COMPONENTS
    }
    missing = {component: not available[component] for component in COMPONENTS}
    gated_features = {
        name: _round(value * quality[name[0]])
        for name, value in raw_features.items()
    }
    severities = _domain_severities(raw_features, quality, available, config)
    present_severities = [
        severity
        for component, severity in severities.items()
        if available[component]
    ]
    ranked = sorted(present_severities, reverse=True)
    top_two = ranked[:2]
    lateral = _arm_face_laterality(raw_features, severities)

    return {
        "version": FEATURE_FUSION_VERSION,
        "mode": "research_only_no_decision",
        "status": (
            "ready"
            if present_severities
            else "awaiting_completed_measurement"
        ),
        "component_order": list(COMPONENTS),
        "feature_names": list(FUSION_FEATURE_NAMES),
        "model_feature_names": list(FUSION_MODEL_FEATURE_NAMES),
        "raw_features": raw_features,
        "gated_features": gated_features,
        "raw_vector": [raw_features[name] for name in FUSION_FEATURE_NAMES],
        "gated_vector": [gated_features[name] for name in FUSION_FEATURE_NAMES],
        "quality": quality,
        "missing": missing,
        # ``model_vector`` is the ready-to-train early-fusion vector.  A zero
        # feature remains distinguishable from unavailable data by its tail.
        "model_vector": [
            *[gated_features[name] for name in FUSION_FEATURE_NAMES],
            *[quality[component] for component in COMPONENTS],
            *[1.0 if missing[component] else 0.0 for component in COMPONENTS],
        ],
        "domain_severity": severities,
        "aggregate": {
            "reliable_domain_count": sum(
                quality[component] > 0.0 for component in COMPONENTS
            ),
            "max_domain_severity": _round(max(present_severities, default=0.0)),
            "mean_top_two_domain_severity": _round(
                sum(top_two) / len(top_two) if top_two else 0.0
            ),
            "arm_face_laterality_agreement": lateral,
        },
    }


def _raw_features(items: Mapping[str, Mapping[str, Any]]) -> dict[str, float]:
    """Return all fixed vector entries, defaulting absent metrics to zero."""

    metrics = {
        component: _metrics(items.get(component, {})) for component in COMPONENTS
    }
    return {
        "B_trunk_orientation_change_score": _metric(
            metrics["B"], "trunk_orientation_change_score"
        ),
        "B_mediolateral_sway_change_score": _metric(
            metrics["B"], "mediolateral_sway_change_score"
        ),
        "E_left_gaze_range": _metric(metrics["E"], "left_gaze_range"),
        "E_right_gaze_range": _metric(metrics["E"], "right_gaze_range"),
        "E_inter_eye_range_asymmetry": _metric(
            metrics["E"], "inter_eye_range_asymmetry"
        ),
        "E_binocular_directional_asymmetry": _metric(
            metrics["E"], "binocular_directional_asymmetry"
        ),
        "E_max_conjugacy_error": _metric(metrics["E"], "max_conjugacy_error"),
        "E_conjugate_rest_gaze_deviation_degrees": _metric(
            metrics["E"], "conjugate_rest_gaze_deviation_degrees"
        ),
        "F_mouth_corner_delta": _metric(metrics["F"], "mouth_corner_delta"),
        "F_smile_score_difference": _metric(
            metrics["F"], "smile_score_difference"
        ),
        "F_smile_strength": _metric(metrics["F"], "smile_strength"),
        "A_level_difference": _metric(metrics["A"], "level_difference"),
        "A_left_drop": _metric(metrics["A"], "left_drop"),
        "A_right_drop": _metric(metrics["A"], "right_drop"),
        "A_drift_difference": _metric(metrics["A"], "drift_difference"),
        "S_character_error_rate": _metric(metrics["S"], "character_error_rate"),
        "S_characters_per_second": _metric(
            metrics["S"], "characters_per_second"
        ),
        "S_pause_fraction": _metric(metrics["S"], "pause_fraction"),
    }


def _domain_severities(
    features: Mapping[str, float],
    quality: Mapping[str, float],
    available: Mapping[str, bool],
    config: BefastConfig,
) -> dict[str, float]:
    """Compute threshold-normalized research severities for explainability."""

    raw = {
        "B": max(
            _scale(features["B_trunk_orientation_change_score"], config.balance_robust_z_threshold),
            _scale(features["B_mediolateral_sway_change_score"], config.balance_robust_z_threshold),
        ),
        "E": max(
            _scale(
                features["E_binocular_directional_asymmetry"],
                config.eye_directional_asymmetry_threshold,
            ),
            _scale(
                features["E_max_conjugacy_error"],
                config.eye_conjugacy_relative_error_threshold,
            ),
            _scale(
                features["E_conjugate_rest_gaze_deviation_degrees"],
                config.eye_rest_gaze_deviation_degrees_threshold,
            ),
        ),
        "F": max(
            _scale(
                abs(features["F_mouth_corner_delta"]),
                config.face_corner_delta_threshold,
            ),
            _scale(
                abs(features["F_smile_score_difference"]),
                config.face_smile_score_difference_threshold,
            ),
        ),
        "A": max(
            _scale(
                abs(features["A_level_difference"]),
                config.arm_level_difference_threshold,
            ),
            _scale(
                abs(features["A_drift_difference"]),
                config.arm_drift_difference_threshold,
            ),
        ),
        "S": max(
            _scale(
                features["S_character_error_rate"],
                config.fusion_speech_character_error_rate_reference,
            ),
            _scale(
                features["S_pause_fraction"],
                config.fusion_speech_pause_fraction_reference,
            ),
            _speech_rate_severity(features["S_characters_per_second"], config),
        ),
    }
    return {
        component: _round(
            min(config.feature_fusion_severity_clip, raw[component])
            * quality[component]
            if available[component]
            else 0.0
        )
        for component in COMPONENTS
    }


def _quality(
    component: str,
    item: Mapping[str, Any],
    config: BefastConfig,
) -> float:
    """Return a conservative component quality in [0, 1]."""

    result_quality = _bounded(item.get("quality"), default=0.0)
    metrics = _metrics(item)
    if component == "E":
        valid = _bounded(metrics.get("minimum_trial_valid_fraction"), result_quality)
        snr = _bounded(
            _metric(metrics, "minimum_response_snr")
            / max(config.eye_response_snr_threshold, 1e-9)
        )
        repeatability = _bounded(
            1.0
            - _metric(metrics, "max_repeat_relative_error")
            / max(config.eye_max_repeat_relative_error, 1e-9)
        )
        return _round(min(result_quality, valid, snr, repeatability))
    if component == "F":
        strength = _bounded(
            _metric(metrics, "smile_strength")
            / max(config.face_min_smile_score, 1e-9)
        )
        return _round(min(result_quality, strength))
    if component == "A":
        completion = _bounded(metrics.get("action_completion_score"), result_quality)
        return _round(min(result_quality, completion))
    return _round(result_quality)


def _arm_face_laterality(
    features: Mapping[str, float], severities: Mapping[str, float]
) -> float:
    """Expose signed A/F agreement without inferring a side from B or E."""

    arm = features["A_level_difference"]
    face = features["F_mouth_corner_delta"]
    if severities["A"] == 0.0 or severities["F"] == 0.0 or arm == 0.0 or face == 0.0:
        return 0.0
    return 1.0 if arm * face > 0.0 else -1.0


def _has_measurement(item: Mapping[str, Any]) -> bool:
    """Manual-only items intentionally do not masquerade as sensor features."""

    return (
        str(item.get("status", "")) in {"positive", "negative"}
        and bool(_metrics(item))
        and _bounded(item.get("quality"), default=0.0) > 0.0
    )


def _metrics(item: Mapping[str, Any]) -> Mapping[str, Any]:
    value = item.get("metrics", {})
    return value if isinstance(value, Mapping) else {}


def _metric(metrics: Mapping[str, Any], name: str) -> float:
    value = metrics.get(name, 0.0)
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return _round(number) if isfinite(number) else 0.0


def _bounded(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = default
    if not isfinite(number):
        number = default
    return min(1.0, max(0.0, number))


def _scale(value: float, reference: float) -> float:
    return abs(float(value)) / max(float(reference), 1e-9)


def _speech_rate_severity(rate: float, config: BefastConfig) -> float:
    if rate <= 0.0:
        return config.feature_fusion_severity_clip
    return max(
        0.0,
        config.fusion_speech_min_characters_per_second / rate - 1.0,
        rate / config.fusion_speech_max_characters_per_second - 1.0,
    )


def _round(value: float) -> float:
    return round(float(value), 5)
