# Corrected validation statistical figures, 2026-09-11

These figures visualize the frozen row-level records and analysis in `experiments/validation_study`. They add no experimental observations. Working hypothesis: the recorded configurations differ in event outcomes, while measurement agreement and within-person repeatability describe distinct properties. Run identity and reference definitions remain unresolved analysis limitations.

## Figure A — Paired outcomes and recorded trigger burden

**English caption.** Paired outcomes and recorded trigger burden. **a**, Full-minus-comparator differences in target detection, incorrect completion and urgency shortfall. Points show event-weighted differences; bars show 95% participant-cluster bootstrap intervals (10,000 resamples, retaining configuration pairing). **b**, Recorded false triggers per reference-negative online hour for 12 participants (29.164 h total). Lines connect the same participant across configurations; coincident observations are retained.

**中文图注。** 配对结果与已记录的误触发负担。**a**，完整系统减去对照方案的目标检测、错误完成和紧急程度不足比例差。点为事件加权差值，线为保留方案配对的参与者聚类 bootstrap 95% 区间（10,000 次重采样）。**b**，12 位参与者在参考阴性在线期间的已记录误触发率，总暴露 29.164 小时；连线对应同一参与者，重合观察值均予以保留。

Statistical notes: detection and urgency use 38 events from 12 participants; incorrect completion uses 18 reference-incomplete events from 13 participants, including two aborts. Higher detection and lower error rates favor Full. The population-baseline incorrect-completion difference and its empirical interval are zero because every observed paired difference is zero; this is not proof of zero population uncertainty. No gates detection includes zero in its interval. Bootstrap seed 20260910 controls resampling only. No p values or multiplicity-adjusted significance claims are shown.

Panel b uses the enumerated continuous-observation event set and common recorded exposure, not a newly verified exhaustive replay of continuous input. The counts are 14, 20 and 19 (Full, population baseline, no gates), yielding exposure-weighted rates of 0.480, 0.686 and 0.651 per hour. Connecting lines show all 12 participant records, with no random jitter. Overlapping zero values reduce the number of visually distinct points. Candidate-set completeness and counterfactual continuous-trigger coverage must be resolved before interpreting these rates as an exhaustive comparison.

Source data: `natural_use_data.csv`; `source-selection.json` retains all three endpoint estimates, intervals and event selections from the frozen analysis. Source workbook/CSV provenance is retained in the parent experiment directory.

## Figure B — Recorded reference agreement and repeatability

**English caption.** Recorded reference agreement and Face repeatability. **a,b**, System-minus-reference error against the reference value for Face mouth-corner change (33 paired measurements from 22 participants) and Eyes gaze offset (25 from 19). Dashed lines indicate zero error. **c**, Two system measurements for each of 11 participants with complete Face repeat pairs; lines connect participants across visits.

**中文图注。** 记录中的参考一致性与 Face 重测表现。**a、b**，Face 嘴角变化量（22 人、33 对）与 Eyes 注视偏移角（19 人、25 对）的系统值减参考值误差，横轴为参考值；虚线表示零误差。**c**，11 位具有完整 Face 重测配对的参与者两次系统测量，连线表示同一参与者。

Interpretation: Face MAE is 0.00452 interocular-width units; Eyes MAE is 0.280°. These arithmetic discrepancies require verified measurement-field and reference calibration correspondence before being called independent validation of the current implementation. No diagnostic cutoffs, normality-based agreement limits or clinical performance claims are imposed. The raw errors show dispersion and larger individual deviations that a single MAE conceals.

Face ICC(2,1) is 0.935 (participant bootstrap 95% interval 0.801–0.975), comparing two system measurements; it does not assess reference accuracy. The interval between measurements is approximately 11.2–15.6 days. Twelve repeat sessions yield 11 complete pairs: P15 lacks the first corresponding Face measure. No missing measurement was filled in.

Source selection: all comparable rows for the two specified reference metrics are included: 33 Face and 25 Eyes out of 213 reference-table rows. The remaining 155 rows concern other metrics and remain in the source data; they were not removed based on error magnitude or appearance. `reference_measurements.csv` retains original row identifiers and `repeatability_pairs.csv` lists every complete pair. All plotted points fit within the displayed axis limits. No measurements were jittered, smoothed or generated.

## Figure contract and QA

- Backend: Python/matplotlib, existing project environment. Figure A answers the paired outcome question; Figure B separates reference agreement from repeatability. Neither demonstrates clinical diagnostic validity or resolves run provenance.
- Evidence allocation: main text should state the principal effect and implication; captions define panels and sample units; detailed provenance and sampling definitions stay with source data/Methods. Prefer replacing redundant displays over adding both figures on top of an already full 8-page paper.
- Exports: editable-text PDF/SVG, 300 dpi PNG, 600 dpi TIFF. Width 7.16 inches (181.9 mm); heights 3.25 and 2.85 inches. Intended for full double-column width. Re-audit text size if reduced substantially.
- Rendering audit: both final PDFs passed strict panel alignment and collision checks (0 failures, 0 warnings); actual PDF minimum text size is 7 pt, above the 5 pt floor. Source validator: 21 pass, 0 warn, 0 fail. Final PNGs were visually inspected for panel alignment, labels, clipped extrema and overlapping annotations. Coincident data marks are intentional and documented.
- Automatic validators inspect rendering/code properties; they do not verify data origin or scientific validity. The limitations remain explicit above.
- Reproduce from repository root: `MPLCONFIGDIR=/private/tmp/pibefast-mpl .venv/bin/python scripts/generate_validation_figures.py --source experiments/validation_study --output experiments/validation_study/figures`. Requires matplotlib/numpy and the installed nature-figure alignment helper. The plotting script reads frozen analysis outputs, so this command reproduces figures rather than re-running data acquisition or configuration ablations.

The adjacent `*.alignment.json`, `*.collision.json`, `*.text-audit.json` and `source-validation.json` retain machine-readable checks. Collision-overlay PDFs and alignment-overlay SVGs are QA artifacts, not submission figures.

## Latest-source interpretation

Source: `pibefast_validation_data.xlsx`. Sampling/timing definitions and Eyes count fields reflect the final archived source; all analysis estimates, intervals and plotted coordinates are unchanged. The manuscript includes these plots at 0.88 text width, giving a minimum text size of approximately 6.14 pt.
