# Current manuscript

**Multimodal Embodied AI for Stroke Warning-Sign Assessment**

- [English paper PDF](pibefast-conference-paper.pdf)
- [Final data index](../../research_data/FINAL_DATA_TABLES.md)

## Build

Run `latexmk -interaction=nonstopmode -halt-on-error main.tex` from this directory. Output: `build/main.pdf`. Delivery copy: `pibefast-conference-paper.pdf`.

## Data and reproduction

- [Supporting methods and evidence](supporting-notes.md)
- `scripts/analyze_validation_workbook.py`: validation analysis.
- `scripts/generate_validation_figures.py`: participant outcome and reference plots; defaults to the final data version.
- `scripts/analyze_human_repeatability.py`: healthy-participant repeatability.
- `scripts/analyze_pipc_pilot.py`: pilot counts and resource analysis.
- `scripts/polish_paper_figures.py`: healthy, speaker and resource plots, plus supplementary pilot counts.
- `scripts/draw_system_overview.py`: system overview.
- `scripts/generate_paper_evidence_figures.py` and `scripts/analyze_model_evidence.py`: calibration and speaker evidence.
- `scripts/verify_fusion_policy.py`, `scripts/export_api_traces.py` and `tests/`: software verification.

Script paths are relative to the repository root. Figure QA depends on local
plotting utilities; generated previews and temporary workspaces are not versioned.
