# Multimodal fusion training

This directory contains the executable training path for the fixed 48-feature
PiBE-FAST representation. It does **not** contain a trained stroke model: the
repository has no jointly acquired, clinically adjudicated B/E/F/A/S cohort.

Prepare one CSV row per screening session with unique `session_id`, `participant_id`, `site_id`,
`label` (0/1), and every name in `app.befast.fusion.FUSION_MODEL_FEATURE_NAMES`.
The label must be assigned independently of the component algorithms. A
participant must occur at one site only. Use an entire site as the untouched
external test set; repeated sessions from one participant must never cross a
split.

Run:

```bash
python -m training.multimodal.train_fusion cohort.csv \
  --external-site SITE_B --output models/multimodal_fusion.json
```

The script validates the schema, tunes L2 regularization with participant-grouped
cross-validation on development sites, fits logistic regression, and reports
external-site discrimination and calibration. The output remains research-only;
the deterministic safety path stays authoritative.


Standardization is fitted within each training fold. Tuning and external metrics
use participant-mean probabilities and require a single label per participant.
This is a single-loop exploratory trainer, not the proposed nested protocol.
It does not validate clinical adjudication, implement calibration selection,
or provide confidence intervals. Legacy 29D data need original per-feature
availability records; do not infer those masks from zero values.
