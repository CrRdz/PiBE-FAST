# Supporting implementation and endpoint notes

These notes accompany the goal-focused manuscript revision (11 September 2026). Participant statistics use the unchanged `pibefast_validation_data.xlsx` records. They distinguish current source definitions from the recorded experimental configurations. No comparator settings are reconstructed by assumption.

## Endpoint definitions

Trigger detection is the event trigger label. For triggered rows, the timestamp is the first trigger during acquisition; for untriggered rows, it is the negative decision after the measurement window. System completeness refers to process completion associated with the final report and is independent of trigger status. A complete record is not a final positive-confirmation flag. These differently defined timestamps are not pooled for completed-check latency. This interpretation was supplied by the author after the initial analysis. The module-specific origin of each trigger remains to be linked to runtime records.

The reported proportions are event-weighted. Participant-cluster resampling retains all events and all configuration outputs within the selected participant. There are 10,000 paired percentile resamples, seed 20260910; intervals are pointwise. A seed specifies resampling, not measurement generation.

## Current implementation and missing comparator specification

Personal Balance enrollment uses five qualified 30-s windows, each preceded by a 1.5-s warmup. The baseline stores trunk orientation, mediolateral sway velocity, and central 90% range. The profile uses the median and max(1.4826 MAD, IQR/1.349). The threshold is 3.5; trunk orientation alone, or both sway domains together, can trigger. Zero change scores zero. Positive change with zero scale is unscorable, with positive scored evidence taking precedence. See `app/befast/balance.py` and `app/befast/config.py`.

| Analysis window / rule | Current default | Source |
|---|---|---|
| Arms hold | 5 s; 20 valid samples; valid fraction 0.55; six early and six late samples | app/befast/config.py |
| Face neutral/smile | 2/3 s; five valid samples per phase; valid fraction 0.50; smile strength 0.22 | app/befast/config.py |
| Balance standing | 30 s after warmup; 30 valid samples; valid fraction 0.75 | app/befast/config.py |
| Eyes trial endpoint | 2-s target, excluding first 0.5 s; five valid samples; fraction 0.60; 24-pixel minimum eye width | app/befast/config.py, eyes.py |
| Speech audio | minimum duration 2 s; voiced duration 1 s; RMS -42 dBFS; clipped fraction 0.02 | app/speech_audio.py |

These are source defaults, not proof of the historical comparator configuration. `data/balance-baseline.json` currently has an empty samples array. The existing analysis script reads the three schemes' recorded output labels; it does not regenerate their sensor decisions. The exact REF-POP-v1.0 values, construction sample, freeze time and independence from evaluation inputs are not available in the inspected configuration artifacts. The exact No gates list of disabled quality, action, or identity rules is also unspecified. Accordingly, between-configuration contrasts remain descriptive.

## Completeness and symptom retention

The 177 started measurement events exclude continuous-observation trigger events and include two aborted attempts. Rows are paired by event ID. Values below are counts under the engineering reference protocol.

| Reference / system completeness | Full | Population baseline | No gates |
|---|---:|---:|---:|
| Complete / complete | 159 | 159 | 159 |
| Complete / incomplete | 0 | 0 | 0 |
| Incomplete / complete | 1 | 1 | 15 |
| Incomplete / incomplete | 17 | 17 | 3 |

| Symptom counts among 38 target events | Full | Population baseline | No gates |
|---|---:|---:|---:|
| Known counts | 35 | 29 | 33 |
| Below reference among known counts | 2 | 3 | 6 |
| Unknown counts | 3 | 9 | 5 |

Full failures include EVT-105 and EVT-119 for symptom retention and EVT-009 for correction history. Available aggregate records do not distinguish failed submission, state loss, export mismatch, or persistence failure. These cannot be converted into authentic interface regression trajectories without the operation sequence and report snapshots. Missing values remain unknown, including the first Face repeat measurement for P15.

## Eyes exported fields

The 25 Eyes comparisons concern the exported gaze-offset field. The implementation exposes monocular and binocular resting-offset quantities, including a same-direction binocular aggregation. The exported field has not been uniquely mapped to one of these quantities, the target geometry and reference time window. Thus its 0.280-degree MAE is a numerical field comparison, not established runtime angular accuracy. The reference uncertainty field is recorded as ±0.12; its calibration basis and interpretation are unspecified. No calibrated disease threshold follows from this comparison.

## Research representation

The research vector and optional severity summaries do not determine the reported trigger or urgency outcomes. The vector has 19 measurements, five quality values, five component masks and 19 feature masks. Invalid inputs are zeroed and masked; raw values remain in audit records. Severity summaries normalize qualified measurements using engineering references and weight them by quality. They are inspection aids, not calibrated probabilities. Undefined values and empty aggregates remain null; research-only E does not dilute another domain's severity.

## Ancillary transcription experiment

A ten-recording transcription test yielded mean CER 3.5625, median CER 1.0 and no exact transcript matches. The sample was positive-only. These results did not support using CER as a decision feature. Its exact transcription build was not retained. This failed auxiliary experiment is retained here and does not support classifier performance or the interaction-contract claim. Current guided speech uses the frozen MDSC classifier after quality assessment; transcription descriptors do not override it.

## Outstanding author input

[AUTHOR_INPUT_NEEDED: ethics approval or exemption, consent and data-use coverage for the 30-participant study, 40-participant repeatability dataset and two-person pilot. Existing record identifiers do not establish the approval scope.]

To restore stronger comparator claims, supply the actual population-reference parameters and construction provenance, the No gates configuration diff, and traceable input/output replay. To resolve deployed report failures, supply minimal deidentified operations and snapshots for the three named events. To interpret Eyes accuracy, supply the export-to-runtime and reference mapping. These are missing evidential inputs, not new results promised by this revision.

## Supporting workflow verification

These previously conducted checks establish bounded software behavior, not stroke detection or healthy-user alert burden. The manuscript now summarizes them after participant and deployment evaluations.

The suite includes 263 bounded configurations and 27 static F/A/S combinations. An independent append-only reference agreed over 1,800 transition checks: 20 seeded sequences of 30 operations, replayed for F/A/S. Measurements entered at the internal result boundary. Snapshots remained unchanged. Separate cases cover B/E questionnaire behavior.

Five synthetic arm trajectories exercise stable hold, sudden and gradual lowering, brief occlusion and missing endpoints. Removing symptom retention, snapshot isolation or correction history broke the relevant contract in each of 15 constructed cases; Full satisfied all 15. A passive-speech regression links anomaly votes to acquisition windows. The host-only 1,000-record queue recovery check remains in the artifacts.

Complete former main-text details: [English LaTeX](supporting/workflow-details-en.tex), [Chinese LaTeX](supporting/workflow-details-zh.tex). No new experimental observations were added in this revision.

## Extended evaluation retained after main-text compression

Session categories describe acquisition settings, whereas target and completeness labels describe individual events. The six routine scripted-action sessions contain 30 events and no target signs (three incomplete inputs). Seven simulated-positive sessions contain 35 events, including 28 target signs. Five challenge sessions contain 20 events: 10 simulated targets and 10 incomplete inputs. These counts were checked against the latest exported session/event tables. The operation label “scripted action measurement” also appears within simulated-positive and challenge sessions and is not a separate cohort definition.

The full pre-compression [English participant analysis](supporting/historical-participant-details-en.tex) and [English supporting evaluations](supporting/historical-evaluation-details-en.tex) preserve the detailed statistics and protocols. Disposable working copies are not versioned.

### Healthy-check variability and speech methods

For B/E/F/A, absolute within-participant residual SDs were 0.289/0.0427/0.00736/0.0279. The main ICC(1,1) retained unequal eligible repeat counts. Complete-case ICC(2,1) remains an alternative analysis in the full audit. The healthy Face and Arms fields used shoulder-width ratios.

The MDSC model used 40-bin Log-Mel means and SDs plus ten temporal, energy and spectral descriptors. Five-fold speaker-grouped evaluation used the 40-speaker development set. The final standardized logistic regression used 33 training speakers and seven validation speakers; threshold selection maximized specificity subject to recording-level sensitivity ≥0.90. Each speaker contributed 405 recordings. Test counts were TP=817, FN=398, TN=1201, FP=14. At the speaker-mean operating point, two of three dysarthric speakers and all three controls were correctly classified.

### Pilot details and supplementary figure

[Supplementary pilot scenario figure](figures/pilot-counts.pdf). Bars compare earlier/later groups descriptively, each with 34 events over 2.25 h. Three no-component negative intervals per group had no triggers. Four sway intervals per group retain protocol-defined labels outside binary denominators. Recovery and 8-h runs remain separate.

Runs occurred from 22 August to 5 September 2026. Passive Balance operated at 16 frames/s, Speech used 10-s windows, and resources were sampled every 60 s (480 samples in the 8-h run). Mean recorded power was 3.52 W; instrumentation is unspecified, so no energy estimate follows. The logged Speech interval of 0.437 s is excluded from latency estimation. Four jointly detected fall pairs had earlier/later median timestamp differences of 1.913/0.913 s, a conditional comparison. The disconnect record spans 668.722 s from trigger to PC receipt. The 9.358-s guided Balance record concerns questionnaire completion, not the 30-s standing measurement.

### Urgency endpoint shown in the main paired-outcome figure

Full/Population baseline/No gates had 3/9/5 urgency shortfalls among 38 targets, all missed detections; among detected targets the mismatch counts were 0/35, 0/29, and 0/33. Four non-new-onset targets required warning; 34 required urgent. These are engineering-reference labels. The conditional agreement excludes missed targets.

## Current decision-fusion verification and historical evidence boundary

The historical 35/38 result is a count of saved target-response labels. All 38 target records lack a verified runtime decision-path mapping in the available package. No path is inferred from component name, acquisition duration or operation label. The [target provenance ledger](../../experiments/fusion_policy_validation/target_provenance.csv) preserves event IDs and supplied log indices with unresolved fields explicitly marked. This is not a finding that the events were false; it limits which system stage the aggregate can validate.

Population baseline and No gates remain historical scheme labels. The new experiment below does not recreate either scheme or validate their reported performance. Missing historical parameters remain null in the [audit](../../experiments/fusion_policy_validation/historical-evidence-audit.json).

The new policy experiment enumerates every combination of five states across five eligible components (3,125 cases). States are negative, insufficient, new-positive, longstanding-positive and unknown-onset-positive. B/E inputs are eligible report or questionnaire states, not automatic eye diagnoses. Independent fixture expectations specify urgency and completeness. The deployed reducer matches all cases. An experimental complete-first variant uses the same reducer but suppresses urgency while incomplete; it matches 1,055 cases and differs in 2,070 cases with coexisting positive and insufficient evidence. This is a policy contrast constructed to expose the separation mechanism, not an empirical benchmark, clinical comparator, or distribution-weighted accuracy result. No random seed is used. It does not test sensor acquisition, per-feature quality gates, delivery, or the full state machine. Existing tests separately exercise those software paths.

Reproduce from the repository root:

```sh
.venv/bin/python scripts/audit_historical_evidence.py
.venv/bin/python scripts/verify_fusion_policy.py
.venv/bin/python -m unittest discover -s tests
```

The [policy manifest](../../experiments/fusion_policy_validation/fusion-policy-summary.json) records exact states, variant rules and source hashes. [All case outputs](../../experiments/fusion_policy_validation/fusion_policy_cases.csv) are retained. Main-text claims are bounded to the current decision policy; the study still lacks empirical final-alert benefit.

## Current API evidence and historical mappings

`../../scripts/export_api_traces.py` reruns four existing API regression scenarios and exports 24 requests with responses to `../../experiments/workflow_validation/api-traces.json`. The scenarios cover late audio following a symptom report, stale audio after session reset, symptom correction with persistent history readback, and unknown-onset B/E symptoms across retry. They use the production Flask API, session, report and temporary SQLite history store, with synthetic sine-wave audio and stub recognition/model outputs. They do not validate browser interaction, network transport, camera acquisition, or real participant input. Source hashes identify the tested implementation and fixture. This is new software verification, not replay of the historical participant failures.

`../../experiments/workflow_validation/contracts.json` retains the rerun 1,800 independent-reference transitions and 15 mechanism-removal cases. All full-implementation assertions passed. The 2,070 complete-first policy differences are constructive consequences of withholding urgency until completion, not effect sizes or empirical gains.

All 33 Face reference rows identify a mean of two independent geometric annotations and list uncertainty as ±0.006. Its statistical definition and annotation aggregation protocol remain unavailable. Main-text MAE is therefore an exported-value comparison, not validated accuracy. Historical healthy E positive-label origin, Balance baseline linkage, and Face normalization mapping remain unresolved. Historical detail files are archives; the current main text controls interpretation. Their shared event IDs do not establish executed, same-input controlled ablations.

Figure 4 applies participant-ordered horizontal offsets from −0.11 to +0.11 to separate overlapping natural-use observations. Each participant has the same offset across all configurations; rates, exposures, counts and intervals are unchanged. Figure 5 labels the reference panels as exported-value comparisons. Plot source data and PDF/SVG outputs are in `../../experiments/workflow_validation/figures/`.
