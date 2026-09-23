#!/usr/bin/env python3
"""Reproduce the PiBE-FAST healthy-participant repeatability analysis.

The script uses only de-identified, feature-level records.  Failed quality
checks remain in the status/quality summaries but are excluded from ICCs by the
prespecified ``analysis_eligible`` flag.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from statistics import fmean


MODULES = ("B", "E", "F", "A")


def _icc_1_1(groups: list[list[float]]) -> tuple[float, float]:
    """One-way random-effects, single-measure ICC for unequal repeats."""

    groups = [values for values in groups if values]
    subjects = len(groups)
    observations = sum(len(values) for values in groups)
    if subjects < 2 or observations <= subjects:
        raise ValueError("ICC(1,1) requires at least two subjects and repeats")
    grand = sum(sum(values) for values in groups) / observations
    ss_between = sum(
        len(values) * (fmean(values) - grand) ** 2 for values in groups
    )
    ss_within = sum(
        sum((value - fmean(values)) ** 2 for value in values)
        for values in groups
    )
    ms_between = ss_between / (subjects - 1)
    ms_within = ss_within / (observations - subjects)
    effective_repeats = (
        observations
        - sum(len(values) ** 2 for values in groups) / observations
    ) / (subjects - 1)
    denominator = ms_between + (effective_repeats - 1.0) * ms_within
    icc = (ms_between - ms_within) / denominator if denominator else 0.0
    return icc, math.sqrt(ms_within)


def _icc_2_1(complete: list[list[float]]) -> float:
    """Two-way random-effects absolute-agreement ICC on complete cases."""

    subjects = len(complete)
    visits = len(complete[0]) if complete else 0
    if subjects < 2 or visits < 2 or any(len(row) != visits for row in complete):
        raise ValueError("ICC(2,1) requires a balanced matrix")
    grand = sum(sum(row) for row in complete) / (subjects * visits)
    subject_means = [fmean(row) for row in complete]
    visit_means = [fmean(row[index] for row in complete) for index in range(visits)]
    ss_subject = visits * sum((value - grand) ** 2 for value in subject_means)
    ss_visit = subjects * sum((value - grand) ** 2 for value in visit_means)
    ss_error = sum(
        (
            complete[subject][visit]
            - subject_means[subject]
            - visit_means[visit]
            + grand
        ) ** 2
        for subject in range(subjects)
        for visit in range(visits)
    )
    ms_subject = ss_subject / (subjects - 1)
    ms_visit = ss_visit / (visits - 1)
    ms_error = ss_error / ((subjects - 1) * (visits - 1))
    denominator = (
        ms_subject
        + (visits - 1) * ms_error
        + visits * (ms_visit - ms_error) / subjects
    )
    return (ms_subject - ms_error) / denominator if denominator else 0.0


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _cluster_interval(
    by_subject: dict[str, list[float]], *, seed: int, replicates: int
) -> list[float]:
    rng = random.Random(seed)
    subjects = sorted(by_subject)
    estimates: list[float] = []
    for _ in range(replicates):
        sampled = [by_subject[rng.choice(subjects)] for _ in subjects]
        estimate, _ = _icc_1_1(sampled)
        if math.isfinite(estimate):
            estimates.append(estimate)
    return [_percentile(estimates, 0.025), _percentile(estimates, 0.975)]


def analyze(path: Path, *, seed: int, replicates: int) -> dict[str, object]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 800:
        raise ValueError(f"expected 800 module records, found {len(rows)}")

    result: dict[str, object] = {
        "analysis": "healthy_participant_repeatability",
        "source": str(path),
        "bootstrap_seed": seed,
        "bootstrap_replicates": replicates,
        "module_records": len(rows),
        "participants": len({row["participant_id"] for row in rows}),
        "modules": {},
    }

    for module_index, module in enumerate(MODULES):
        module_rows = [row for row in rows if row["module"] == module]
        eligible = [row for row in module_rows if row["analysis_eligible"] == "1"]
        by_subject: dict[str, list[float]] = defaultdict(list)
        by_subject_visit: dict[str, dict[int, float]] = defaultdict(dict)
        for row in eligible:
            value = float(row["metric_value"])
            by_subject[row["participant_id"]].append(value)
            by_subject_visit[row["participant_id"]][int(row["visit_no"])] = value

        icc_1_1, within_subject_sd = _icc_1_1(list(by_subject.values()))
        complete = [
            [visits[index] for index in range(1, 6)]
            for _, visits in sorted(by_subject_visit.items())
            if set(visits) == set(range(1, 6))
        ]
        reference = float(module_rows[0]["engineering_reference"])
        status_counts = Counter(row["status"] for row in module_rows)
        result["modules"][module] = {
            "selected_output": module_rows[0]["selected_output"],
            "total_evaluations": len(module_rows),
            "analyzable_evaluations": len(eligible),
            "analyzable_rate": len(eligible) / len(module_rows),
            "status_counts": dict(sorted(status_counts.items())),
            "mean_quality": fmean(float(row["quality"]) for row in module_rows),
            "icc_1_1": icc_1_1,
            "icc_1_1_participant_bootstrap_95pct": _cluster_interval(
                by_subject,
                seed=seed + module_index,
                replicates=replicates,
            ),
            "within_participant_residual_sd": within_subject_sd,
            "engineering_reference": reference,
            "within_participant_sd_as_reference_percent": (
                100.0 * within_subject_sd / reference
            ),
            "complete_case_participants": len(complete),
            "icc_2_1_complete_case": _icc_2_1(complete),
        }
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--measurements",
        type=Path,
        default=Path("research_data/human_repeatability/measurements.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("research_data/human_repeatability/analysis_report.json"),
    )
    parser.add_argument("--seed", type=int, default=203)
    parser.add_argument("--bootstrap-replicates", type=int, default=5000)
    args = parser.parse_args()
    report = analyze(
        args.measurements,
        seed=args.seed,
        replicates=args.bootstrap_replicates,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
