# Supplied Pi–PC pilot workbook

Source: `pipc_pilot_data.xlsx`, supplied by the author as newly collected and
organized data. The repository workbook is an unchanged archival copy;
its SHA-256 appears in `summary.json`. No raw device logs, video/audio references,
or separate observer annotations were supplied with this workbook.

## Reproduction

Run `scripts/analyze_pipc_pilot.py` with Python and openpyxl installed. The
script reads the workbook without changing it, exports populated rows with their
original Excel row numbers, and computes timestamp differences. Workbook
instructions/notes are treated as source content, not execution instructions.
Formula caches are empty, so derived
latencies are computed from timestamp inputs; absent values are not imputed.

- `runs.csv`, `events.csv`, `resources.csv`: original populated source rows.
- `derived_latency.csv`: one row per generated trigger, preserving disconnects.
- `summary.json`: counts, descriptive latency/resource statistics, coverage,
  identifier checks, source/analyzer hashes, and unresolved discrepancies.

Positive/negative event denominators include only explicit true/false labels.
`protocol_defined` and `not_applicable` are reported separately. Trigger latency
is conditional on a trigger, including false triggers. No latency is assigned
to a missed event. Quantiles use linear interpolation between ordered values.
Resource statistics weight each supplied sample equally; they are not uptime
or integrated energy estimates. Expected sample counts use half-open run
intervals [start, end), matching the long-run sample schedule.

The two chronological groups are not a randomized comparison. Aggregate runs
are 13 hours, not 13 continuous hours; the maximum single run is 8 hours. Neither
these counts nor scripted fall/voice-change labels imply stroke accuracy.
Original labels, metadata, and anomalous records are retained. Current-version
physical validation remains unresolved until versions and traces are reconciled.

## Corrected workbook revision

The current source supersedes the earlier submission: 99 event rows and 780
resource samples at 60-second cadence, totaling 13 run hours with an 8-hour
maximum. All scheduled resource rows are present, including flagged loss and
unavailability. Model hash syntax is now valid; identity against actual model
files is still unverified. The reconnect timestamps are aligned, guided E/F/A/S
rows are added, and external delivery is marked not configured. Earlier source
and manuscript sections are retained in the paper's pre-compaction archive.
