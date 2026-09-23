# Software validation evidence

This directory contains deterministic software checks and frozen model evidence;
it does not contain a clinical or hardware-performance study.

Reproduce the checks from the repository root:

```bash
.venv/bin/python scripts/analyze_model_evidence.py
.venv/bin/python scripts/verify_workflow_contracts.py
python -m unittest discover -s tests
```

- `demo_cases.csv` contains bounded urgency/completeness scenarios.
- `state_cases.csv` and `state_summary.json` record workflow-contract checks.
- `mdsc_speakers.csv` contains speaker-level results from the frozen MDSC split.
- `summary.json` records the scope, representation checks, and source hashes.

Synthetic configurations are assertions, not independent experimental trials.
The MDSC results describe a dysarthria representation and must not be read as
acute-stroke diagnostic performance.
