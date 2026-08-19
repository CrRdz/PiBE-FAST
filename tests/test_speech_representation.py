import json
from pathlib import Path
import tempfile
import unittest
import wave

import numpy as np

from app.speech_representation import (
    DysarthriaRepresentationModel,
    SPEECH_REPRESENTATION_FEATURES,
    SPEECH_REPRESENTATION_VERSION,
    extract_speech_representation,
)
from training.speech.prepare_mdsc import build_manifest


def write_wav(path: Path, frequency: float = 220.0, seconds: float = 0.8) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    time = np.arange(int(16000 * seconds)) / 16000.0
    samples = (np.sin(2 * np.pi * frequency * time) * 0.2 * 32767).astype("<i2")
    with wave.open(str(path), "wb") as target:
        target.setnchannels(1)
        target.setsampwidth(2)
        target.setframerate(16000)
        target.writeframes(samples.tobytes())


class SpeechRepresentationTest(unittest.TestCase):
    def test_extracts_finite_runtime_feature_vector(self):
        with tempfile.TemporaryDirectory() as directory:
            wav = Path(directory) / "sample.wav"
            write_wav(wav)
            first = extract_speech_representation(wav)
            second = extract_speech_representation(wav)
        self.assertEqual(first.shape, (len(SPEECH_REPRESENTATION_FEATURES),))
        self.assertTrue(np.isfinite(first).all())
        np.testing.assert_allclose(first, second)

    def test_loads_model_and_returns_shadow_prediction(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            wav = root / "sample.wav"
            write_wav(wav)
            count = len(SPEECH_REPRESENTATION_FEATURES)
            model_path = root / "model.json"
            model_path.write_text(
                json.dumps(
                    {
                        "component": "speech_dysarthria_representation",
                        "model_version": "test-mdsc",
                        "representation_version": SPEECH_REPRESENTATION_VERSION,
                        "feature_names": list(SPEECH_REPRESENTATION_FEATURES),
                        "means": [0.0] * count,
                        "scales": [1.0] * count,
                        "coefficients": [0.0] * count,
                        "intercept": 1.0,
                        "decision_threshold": 0.5,
                    }
                ),
                encoding="utf-8",
            )
            model = DysarthriaRepresentationModel(model_path)
            prediction = model.predict_wav(wav)
        self.assertTrue(model.availability()[0])
        self.assertGreater(prediction.probability, 0.5)
        self.assertTrue(prediction.predicted_dysarthria)

    def test_mdsc_manifest_is_split_by_speaker(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "train"
            for folder, prefix in (("Control", "CM"), ("Uncontrol", "DM")):
                lines = []
                for speaker_index in range(1, 5):
                    speaker = f"{prefix}{speaker_index:04d}"
                    sample = f"{speaker}_0001"
                    write_wav(root / folder / "wav" / speaker / f"{sample}.wav")
                    lines.append(f"{sample} 测试语音")
                transcript = root / folder / "transcript" / "label.txt"
                transcript.parent.mkdir(parents=True, exist_ok=True)
                transcript.write_text("\n".join(lines), encoding="utf-8")
            rows, report = build_manifest(
                Path(directory), seed=7, train_fraction=0.5, validation_fraction=0.25
            )
        by_speaker = {}
        for row in rows:
            by_speaker.setdefault(row["speaker_id"], set()).add(row["split"])
        self.assertTrue(all(len(values) == 1 for values in by_speaker.values()))
        self.assertEqual(report["class_speakers"], {"control": 4, "dysarthria": 4})

    def test_mdsc_manifest_accepts_official_class_archive_layout(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for folder, prefix in (("Control", "CM"), ("Uncontrol", "DM")):
                for speaker_index, source_split in enumerate(("train", "train", "dev", "test"), 1):
                    speaker = f"{prefix}{speaker_index:04d}"
                    sample = f"{speaker}_0001"
                    nested = "eval" if folder == "Uncontrol" and source_split != "train" else ""
                    base = root / folder / source_split
                    if nested:
                        base /= nested
                    write_wav(base / "wav" / speaker / f"{sample}.wav")
                    transcript = base / "transcript" / speaker / "label.txt"
                    transcript.parent.mkdir(parents=True, exist_ok=True)
                    transcript.write_text(f"{sample} 测试语音\n", encoding="utf-8")
            rows, report = build_manifest(
                root, seed=7, train_fraction=0.5, validation_fraction=0.25
            )
        self.assertEqual(len(rows), 8)
        self.assertEqual(report["class_speakers"], {"control": 4, "dysarthria": 4})
        self.assertTrue(all(row["source_partition"] for row in rows))


if __name__ == "__main__":
    unittest.main()
