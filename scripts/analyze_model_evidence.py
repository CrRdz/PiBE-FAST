"""Deterministic software demo and frozen-model audit; no new human data."""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.befast.urgency import screening_assessment
from app.befast.fusion import build_feature_fusion
from app.befast.config import BefastConfig


def item(status, **extra):
    return {"status": status, "reason": "scripted_input", **extra}


def run():
    out = ROOT / "experiments/software_validation"
    out.mkdir(parents=True, exist_ok=True)
    # Expected outputs are specified here, independently of the implementation.
    cases = [
        ("D1", "All requested checks negative", {"F": item("negative"), "A": item("negative")}, {}, "none", "complete"),
        ("D2", "Technical arm acquisition failure", {"F": item("negative"), "A": item("insufficient")}, {}, "none", "incomplete"),
        ("D3", "New face positive and arm failure", {"F": item("positive"), "A": item("insufficient")}, {"F": True}, "urgent", "incomplete"),
        ("D4", "Chronic face positive and skipped arm", {"F": item("positive"), "A": item("skipped")}, {"F": False}, "warning", "incomplete"),
        ("D5", "Reported new inability with failed arm task", {"A": item("insufficient", reported_functional_problem=True)}, {"A": True}, "urgent", "incomplete"),
        ("D6", "Reported difficulty with unknown onset", {"S": item("insufficient", reported_functional_problem=True)}, {}, "urgent", "incomplete"),
        ("D7", "Chronic face and new speech positive", {"F": item("positive"), "S": item("positive")}, {"F": False, "S": True}, "urgent", "complete"),
        ("D8", "F-only request with research E data", {"F": item("negative"), "E": item("insufficient", decision_eligible=False)}, {}, "none", "complete"),
    ]
    rows = []
    for cid, label, items, onset, eu, ec in cases:
        actual = screening_assessment(items, onset)
        assert (actual["urgency"], actual["completeness"]) == (eu, ec), cid
        # Controlled ablations, not implementations attributed to prior work.
        coerced = {k: {**v, "status": v["status"] if v["status"] in {"positive", "negative"} else "negative"} for k,v in items.items()}
        missing_normal = screening_assessment(coerced, onset)
        rows.append(dict(id=cid, scenario=label, expected_urgency=eu, expected_completeness=ec,
                         missing_normal=missing_normal["decision"], scalar=actual["decision"],
                         proposed_urgency=actual["urgency"], proposed_completeness=actual["completeness"],
                         missing_normal_hides_incomplete=int(ec == "incomplete" and missing_normal["completeness"] == "complete"),
                         scalar_hides_incomplete=int(ec == "incomplete" and actual["decision"] != "incomplete"),
                         proposed_hides_incomplete=int(ec == "incomplete" and actual["completeness"] != ec)))
    def fusion(metrics):
        return build_feature_fusion({"S": item("negative", quality=1.0, metrics=metrics)}, BefastConfig())
    a = fusion({"mdsc_dysarthria_probability": .1, "pause_fraction": .2})
    b = fusion({"mdsc_dysarthria_probability": .1, "pause_fraction": .2, "character_error_rate": 0., "characters_per_second": 0.})
    masks = {"legacy_29d_collision": a["legacy_model_vector"] == b["legacy_model_vector"],
             "current_48d_collision": a["model_vector"] == b["model_vector"]}
    assert masks == {"legacy_29d_collision": True, "current_48d_collision": False}
    model_path = ROOT / "models/mdsc_dysarthria_v1.json"
    archive_path = ROOT / "training/processed/mdsc_features.npz"
    model = json.loads(model_path.read_text())
    data = np.load(archive_path, allow_pickle=False)
    assert list(data["feature_names"]) == model["feature_names"]
    z = (data["matrix"] - np.array(model["means"])) / np.array(model["scales"])
    prob = 1 / (1 + np.exp(-np.clip(z @ np.array(model["coefficients"]) + model["intercept"], -40, 40)))
    pred = prob >= model["decision_threshold"]
    speakers = []
    for split in ("validation", "test"):
        for subject in np.unique(data["subjects"][data["splits"] == split]):
            ix = (data["subjects"] == subject) & (data["splits"] == split)
            y = data["labels"][ix]; p = pred[ix]
            assert len(np.unique(y)) == 1
            speakers.append(dict(split=split, speaker=str(subject), label=int(y[0]), recordings=int(ix.sum()),
                                 tp=int(((y==1)&p).sum()), fn=int(((y==1)&~p).sum()),
                                 tn=int(((y==0)&~p).sum()), fp=int(((y==0)&p).sum()),
                                 mean_probability=float(prob[ix].mean())))
    for name, records in (("demo_cases", rows), ("mdsc_speakers", speakers)):
        with (out / f"{name}.csv").open("w", newline="") as stream:
            writer=csv.DictWriter(stream, fieldnames=list(records[0])); writer.writeheader(); writer.writerows(records)
    summary = {"scope": "deterministic software cases and descriptive frozen-MDSC reanalysis; no target-device or clinical experiment",
               "cases": rows, "representation": masks, "speakers": speakers,
               "incomplete_cases": sum(r["expected_completeness"] == "incomplete" for r in rows),
               "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in
                   (Path(__file__), model_path, archive_path, ROOT / "app/befast/fusion.py", ROOT / "app/befast/urgency.py")}}
    (out / "summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    print(json.dumps({"demo_cases":len(rows), **masks, "test_speakers":[r for r in speakers if r['split']=='test']}, indent=2))


if __name__ == "__main__":
    run()
