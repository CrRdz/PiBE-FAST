#!/usr/bin/env python3
"""Analyze de-identified healthy-participant Eyes technical-validation logs.

The manifest maps one completed Eyes attempt to a JSONL research log and its
acquisition conditions. Results are descriptive technical measurements, not
stroke sensitivity/specificity estimates.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, median, stdev
from typing import Any, Iterable

import numpy as np


TERMINAL_STATUSES = {"negative", "positive", "insufficient", "skipped"}
EYES = ("left", "right")
DIRECTIONS = ("left", "right")
LEFT_TRIALS = (3, 9, 11)
RIGHT_TRIALS = (5, 7, 13)
CENTER_TRIALS = (2, 4, 6, 8, 10, 12)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path)
    parser.add_argument("--output-markdown", type=Path)
    return parser.parse_args()


def _float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _rounded(value: float | None) -> float | None:
    return round(float(value), 6) if value is not None and np.isfinite(value) else None


def _median(values: Iterable[float | None]) -> float | None:
    clean = [float(value) for value in values if value is not None and np.isfinite(value)]
    return median(clean) if clean else None


def _extract_eye_results(path: Path) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    previous_signature: tuple[Any, ...] | None = None
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            item = (((record.get("befast") or {}).get("items") or {}).get("E") or {})
            status = item.get("status")
            if status not in TERMINAL_STATUSES:
                previous_signature = None
                continue
            metrics = item.get("metrics") if isinstance(item.get("metrics"), dict) else {}
            signature = (
                status,
                item.get("reason"),
                len(metrics),
                metrics.get("total_valid_samples"),
                metrics.get("coordinate_orientation"),
            )
            if signature != previous_signature:
                results.append({**item, "_line_number": line_number})
            previous_signature = signature
    return results


def _resolve_path(manifest: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else (manifest.parent / path).resolve()


def _endpoint(metrics: dict[str, Any], eye: str, trials: tuple[int, ...]) -> float | None:
    return _median(metrics.get(f"trial_{trial}_{eye}_iris_position") for trial in trials)


def _enrich(row: dict[str, str], result: dict[str, Any]) -> dict[str, Any]:
    metrics = result.get("metrics") if isinstance(result.get("metrics"), dict) else {}
    orientation = _float(metrics.get("coordinate_orientation")) or 1.0
    enriched: dict[str, Any] = {
        **row,
        "attempt": int(row["attempt"]),
        "result_index": int(row.get("result_index") or 1),
        "status": result.get("status"),
        "reason": result.get("reason"),
        "quality": _float(result.get("quality")),
        "line_number": result.get("_line_number"),
    }
    for eye in EYES:
        left = _endpoint(metrics, eye, LEFT_TRIALS)
        center = _endpoint(metrics, eye, CENTER_TRIALS)
        right = _endpoint(metrics, eye, RIGHT_TRIALS)
        enriched[f"{eye}_left_endpoint"] = left
        enriched[f"{eye}_center_endpoint"] = center
        enriched[f"{eye}_right_endpoint"] = right
        enriched[f"{eye}_left_separation"] = (
            orientation * (center - left) if left is not None and center is not None else None
        )
        enriched[f"{eye}_right_separation"] = (
            orientation * (right - center) if right is not None and center is not None else None
        )
        for direction in DIRECTIONS:
            enriched[f"triple_{eye}_{direction}_response"] = _float(
                metrics.get(f"{eye}_{direction}_response")
            )
            enriched[f"single_{eye}_{direction}_response"] = _float(
                metrics.get(f"{eye}_{direction}_repeat_1_response")
            )
    for direction in DIRECTIONS:
        first_values = [
            enriched.get(f"single_{eye}_{direction}_response") for eye in EYES
        ]
        enriched[f"single_binocular_{direction}_response"] = (
            mean(float(value) for value in first_values)
            if all(value is not None for value in first_values)
            else None
        )
    for key in (
        "binocular_left_response",
        "binocular_right_response",
        "binocular_directional_asymmetry",
        "max_conjugacy_error",
        "conjugate_rest_gaze_deviation_degrees",
        "minimum_trial_valid_fraction",
        "max_head_rotation_degrees",
        "max_trial_gaze_mad",
        "max_repeat_relative_error",
    ):
        enriched[key] = _float(metrics.get(key))
    # This comparison deliberately uses only the expected response sign.  It
    # does not introduce an unvalidated SNR or amplitude cutoff.
    enriched["single_all_directions_correct"] = all(
        (enriched.get(f"single_{eye}_{direction}_response") or 0.0) > 0.0
        for eye in EYES
        for direction in DIRECTIONS
    )
    enriched["triple_all_directions_correct"] = all(
        (enriched.get(f"triple_{eye}_{direction}_response") or 0.0) > 0.0
        for eye in EYES
        for direction in DIRECTIONS
    )
    enriched["endpoint_order_correct"] = all(
        (enriched.get(f"{eye}_{direction}_separation") or 0.0) > 0.0
        for eye in EYES
        for direction in DIRECTIONS
    )
    enriched["measurement_available"] = all(
        enriched.get(key) is not None
        for key in (
            "binocular_left_response",
            "binocular_right_response",
            "binocular_directional_asymmetry",
            "max_conjugacy_error",
            "conjugate_rest_gaze_deviation_degrees",
        )
    )
    return enriched


def load_attempts(manifest: Path) -> list[dict[str, Any]]:
    attempts: list[dict[str, Any]] = []
    with manifest.open(newline="", encoding="utf-8-sig") as source:
        reader = csv.DictReader(source)
        required = {
            "participant_id", "session_id", "attempt", "condition", "distance_cm",
            "light_lux", "glasses", "head_condition", "result_index", "log_path",
        }
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"manifest missing columns: {', '.join(sorted(missing))}")
        for row_number, row in enumerate(reader, start=2):
            path = _resolve_path(manifest, row["log_path"])
            results = _extract_eye_results(path)
            result_index = int(row.get("result_index") or 1)
            if result_index < 1 or result_index > len(results):
                raise ValueError(
                    f"manifest row {row_number}: result_index {result_index} not found in {path}"
                )
            row["log_path"] = str(path)
            attempts.append(_enrich(row, results[result_index - 1]))
    return attempts


def _condition_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    interpretable = sum(bool(row.get("measurement_available")) for row in rows)
    insufficient = total - interpretable
    return {
        "attempts": total,
        "participants": len({row["participant_id"] for row in rows}),
        "interpretable_rate": _rounded(_rate(interpretable, total)),
        "insufficient_rate": _rounded(_rate(insufficient, total)),
        "endpoint_order_correct_rate": _rounded(
            _rate(sum(bool(row["endpoint_order_correct"]) for row in rows), total)
        ),
        "median_minimum_trial_valid_fraction": _rounded(
            _median(row.get("minimum_trial_valid_fraction") for row in rows)
        ),
        "median_max_head_rotation_degrees": _rounded(
            _median(row.get("max_head_rotation_degrees") for row in rows)
        ),
        "reasons": dict(sorted(_counts(row.get("reason") for row in rows).items())),
    }


def _counts(values: Iterable[Any]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for value in values:
        counts[str(value)] += 1
    return dict(counts)


def _retry_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["participant_id"], row["session_id"], row["condition"])].append(row)
    eligible = 0
    recovered = 0
    for attempts in groups.values():
        attempts.sort(key=lambda row: row["attempt"])
        if attempts[0].get("measurement_available"):
            continue
        eligible += 1
        recovered += int(
            any(bool(row.get("measurement_available")) for row in attempts[1:])
        )
    return {
        "first_attempt_insufficient_groups": eligible,
        "successful_retries": recovered,
        "retry_success_rate": _rounded(_rate(recovered, eligible)),
    }


def _paired_session_matrix(rows: list[dict[str, Any]], feature: str) -> tuple[np.ndarray, list[str]]:
    by_participant: dict[str, dict[str, float]] = defaultdict(dict)
    for row in rows:
        if (
            row["condition"] != "baseline"
            or row["attempt"] != 1
            or not row.get("measurement_available")
        ):
            continue
        value = _float(row.get(feature))
        if value is not None:
            by_participant[row["participant_id"]][row["session_id"]] = value
    participant_ids = [
        participant
        for participant, values in sorted(by_participant.items())
        if len(values) >= 2
    ]
    matrix = (
        np.asarray(
            [
                [
                    by_participant[participant][session]
                    for session in sorted(by_participant[participant])[:2]
                ]
                for participant in participant_ids
            ],
            dtype=float,
        )
        if participant_ids
        else np.empty((0, 2), dtype=float)
    )
    return matrix, ["first", "second"] if participant_ids else []


def _icc_2_1(matrix: np.ndarray) -> float | None:
    if matrix.ndim != 2 or matrix.shape[0] < 2 or matrix.shape[1] < 2:
        return None
    n, k = matrix.shape
    grand = float(matrix.mean())
    row_means = matrix.mean(axis=1)
    column_means = matrix.mean(axis=0)
    ms_rows = k * float(np.sum((row_means - grand) ** 2)) / (n - 1)
    ms_columns = n * float(np.sum((column_means - grand) ** 2)) / (k - 1)
    residual = matrix - row_means[:, None] - column_means[None, :] + grand
    ms_error = float(np.sum(residual**2)) / ((n - 1) * (k - 1))
    denominator = ms_rows + (k - 1) * ms_error + k * (ms_columns - ms_error) / n
    return (ms_rows - ms_error) / denominator if abs(denominator) > 1e-12 else None


def _repeatability(rows: list[dict[str, Any]], feature: str) -> dict[str, Any]:
    matrix, sessions = _paired_session_matrix(rows, feature)
    if matrix.shape[0] < 2 or matrix.shape[1] != 2:
        return {"paired_participants": int(matrix.shape[0]), "sessions": sessions, "icc_2_1": None}
    differences = matrix[:, 1] - matrix[:, 0]
    bias = float(differences.mean())
    difference_sd = float(differences.std(ddof=1))
    within_sd = float(np.sqrt(np.mean(differences**2) / 2.0))
    grand_mean = abs(float(matrix.mean()))
    return {
        "paired_participants": int(matrix.shape[0]),
        "sessions": sessions,
        "icc_2_1": _rounded(_icc_2_1(matrix)),
        "mean_absolute_difference": _rounded(float(np.mean(np.abs(differences)))),
        "within_subject_cv": _rounded(within_sd / grand_mean if grand_mean > 1e-12 else None),
        "bland_altman_bias": _rounded(bias),
        "bland_altman_lower_loa": _rounded(bias - 1.96 * difference_sd),
        "bland_altman_upper_loa": _rounded(bias + 1.96 * difference_sd),
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    first_attempts = [row for row in rows if row["attempt"] == 1]
    by_condition: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in first_attempts:
        by_condition[row["condition"]].append(row)
    head_turn_rows = [row for row in first_attempts if row["head_condition"] == "deliberate_turn"]
    still_rows = [row for row in first_attempts if row["head_condition"] == "still"]
    endpoint_rows = [row for row in first_attempts if row.get("measurement_available")]
    return {
        "analysis_role": "healthy-participant technical validation; not stroke diagnostic validation",
        "attempts": len(rows),
        "participants": len({row["participant_id"] for row in rows}),
        "sessions": len({(row["participant_id"], row["session_id"]) for row in rows}),
        "first_attempt_failure_rate": _rounded(
            _rate(
                sum(not row.get("measurement_available") for row in first_attempts),
                len(first_attempts),
            )
        ),
        "retry": _retry_summary(rows),
        "conditions": {key: _condition_summary(value) for key, value in sorted(by_condition.items())},
        "endpoint_separation": {
            "interpretable_attempts": len(endpoint_rows),
            "correct_left_center_right_order_rate": _rounded(
                _rate(sum(bool(row["endpoint_order_correct"]) for row in endpoint_rows), len(endpoint_rows))
            ),
            **{
                f"median_{eye}_{direction}_separation": _rounded(
                    _median(row.get(f"{eye}_{direction}_separation") for row in endpoint_rows)
                )
                for eye in EYES
                for direction in DIRECTIONS
            },
        },
        "head_motion_control": {
            "deliberate_turn_attempts": len(head_turn_rows),
            "deliberate_turn_measurement_available_rate": _rounded(
                _rate(sum(bool(row.get("measurement_available")) for row in head_turn_rows), len(head_turn_rows))
            ),
            "deliberate_turn_median_head_rotation_degrees": _rounded(
                _median(row.get("max_head_rotation_degrees") for row in head_turn_rows)
            ),
            "still_attempts": len(still_rows),
            "still_measurement_available_rate": _rounded(
                _rate(
                    sum(bool(row.get("measurement_available")) for row in still_rows),
                    len(still_rows),
                )
            ),
            "still_median_head_rotation_degrees": _rounded(
                _median(row.get("max_head_rotation_degrees") for row in still_rows)
            ),
        },
        "single_vs_triple": {
            "single_all_directions_correct_rate": _rounded(
                _rate(sum(bool(row["single_all_directions_correct"]) for row in first_attempts), len(first_attempts))
            ),
            "triple_all_directions_correct_rate": _rounded(
                _rate(sum(bool(row["triple_all_directions_correct"]) for row in first_attempts), len(first_attempts))
            ),
            "single_left_repeatability": _repeatability(rows, "single_binocular_left_response"),
            "triple_left_repeatability": _repeatability(rows, "binocular_left_response"),
            "single_right_repeatability": _repeatability(rows, "single_binocular_right_response"),
            "triple_right_repeatability": _repeatability(rows, "binocular_right_response"),
        },
        "cross_session_repeatability": {
            feature: _repeatability(rows, feature)
            for feature in (
                "binocular_left_response",
                "binocular_right_response",
                "binocular_directional_asymmetry",
                "max_conjugacy_error",
                "conjugate_rest_gaze_deviation_degrees",
            )
        },
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = sorted({key for row in rows for key in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Healthy Eyes technical validation",
        "",
        "> Technical measurement study only; these results do not estimate stroke diagnostic performance.",
        "",
        f"Participants: {summary['participants']}; attempts: {summary['attempts']}; "
        f"first-attempt failure rate: {summary['first_attempt_failure_rate']}.",
        "",
        "| Condition | Attempts | Interpretable | Insufficient | Correct endpoint order |",
        "|---|---:|---:|---:|---:|",
    ]
    for condition, values in summary["conditions"].items():
        lines.append(
            f"| {condition} | {values['attempts']} | {values['interpretable_rate']} | "
            f"{values['insufficient_rate']} | {values['endpoint_order_correct_rate']} |"
        )
    lines.extend([
        "",
        "## Prespecified controls",
        "",
        f"- Deliberate-turn measurement-available rate: {summary['head_motion_control']['deliberate_turn_measurement_available_rate']}",
        f"- Static-head measurement-available rate: {summary['head_motion_control']['still_measurement_available_rate']}",
        f"- Single-trial all-direction correct rate: {summary['single_vs_triple']['single_all_directions_correct_rate']}",
        f"- Three-repeat all-direction correct rate: {summary['single_vs_triple']['triple_all_directions_correct_rate']}",
        "",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    attempts = load_attempts(args.manifest.resolve())
    summary = summarize(attempts)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.output_csv:
        write_csv(args.output_csv, attempts)
    if args.output_markdown:
        write_markdown(args.output_markdown, summary)
    print(json.dumps({"attempts": len(attempts), "output": str(args.output_json)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
