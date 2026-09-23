# Healthy-participant repeatability data

This directory contains de-identified, feature-level records used for the
PiBE-FAST B/E/F/A repeatability analysis. It contains no names, contact details,
medical-record numbers, images, audio, or raw video.

Files:

- `human_repeatability_data.xlsx`: de-identified audited source workbook;
- `participants.csv`: coded participant metadata and study strata;
- `sessions.csv`: coded session/device metadata. Exact collection dates are
  intentionally omitted; `interval_from_first_days` preserves visit timing;
- `measurements.csv`: one row per B/E/F/A module evaluation, including failed
  quality checks rather than deleting them;
- `source_integrity.json`: SHA-256 of the source workbook and the privacy
  transformation applied to the repository copy;
- `analysis_report.json`: deterministic repeatability and quality results.

Reproduce the report from the repository root:

```bash
python3 scripts/analyze_human_repeatability.py
```

The primary repeatability estimate is ICC(1,1), retained for continuity with
the prespecified manuscript analysis and computed with an unequal-repeat
correction after quality exclusions. The report also includes participant-level
bootstrap 95% intervals and complete-case ICC(2,1) as a sensitivity analysis.
Low ICC values are interpreted as poor repeatability, not as evidence of
measurement validity. Engineering references are not clinical cutoffs.
