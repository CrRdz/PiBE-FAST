import tempfile
import threading
import time
import unittest
import wave
from pathlib import Path

import numpy as np

from app.passive_speech import (
    EVIDENCE_VERSION,
    PassiveSpeechConfig,
    PassiveSpeechMonitor,
    analyze_passive_speech_wav,
)


def write_modulated_wav(
    path: Path,
    *,
    frequency: float = 220.0,
    seconds: float = 3.0,
    amplitude: float = 0.3,
) -> None:
    sample_rate = 16_000
    timestamps = np.arange(int(sample_rate * seconds)) / sample_rate
    envelope = 0.35 + 0.65 * np.square(
        np.maximum(0.0, np.sin(2.0 * np.pi * 2.5 * timestamps))
    )
    signal = (
        np.sin(2.0 * np.pi * frequency * timestamps)
        * envelope
        * amplitude
        * 32767
    ).astype("<i2")
    with wave.open(str(path), "wb") as target:
        target.setnchannels(1)
        target.setsampwidth(2)
        target.setframerate(sample_rate)
        target.writeframes(signal.tobytes())


class _BlockingCapture:
    def __init__(self) -> None:
        self.started = threading.Event()

    def availability(self):
        return True, None

    def capture(self, output_path, config, cancel_event):
        self.started.set()
        if cancel_event.wait(2.0):
            raise RuntimeError("capture cancelled")
        write_modulated_wav(Path(output_path), seconds=config.capture_seconds)


class _ImmediateCapture:
    def availability(self):
        return True, None

    def capture(self, output_path, config, cancel_event):
        write_modulated_wav(Path(output_path), seconds=config.capture_seconds)


class PassiveSpeechTest(unittest.TestCase):
    def config(self, **overrides):
        values = {
            "window_seconds": 3.0,
            "interval_seconds": 0.01,
            "retry_seconds": 0.01,
            "min_voiced_seconds": 1.0,
            "baseline_windows": 3,
            "baseline_max_windows": 8,
            "recent_windows": 3,
            "anomaly_votes_required": 2,
            "anomaly_z_threshold": 2.0,
            "min_abnormal_domains": 2,
        }
        values.update(overrides)
        return PassiveSpeechConfig(**values)

    def test_extracts_local_natural_speech_features(self):
        with tempfile.TemporaryDirectory() as directory:
            wav_path = Path(directory) / "window.wav"
            write_modulated_wav(wav_path)

            result = analyze_passive_speech_wav(wav_path, self.config())

        self.assertTrue(result["valid"])
        self.assertEqual(result["reason"], "passive_speech_window_ready")
        self.assertEqual(result["estimator"], EVIDENCE_VERSION)
        self.assertGreater(
            result["features"]["syllable_nuclei_rate_hz"],
            0.0,
        )
        self.assertGreater(result["features"]["intensity_range_db"], 0.0)
        self.assertGreater(result["metrics"]["f0_median_hz"], 100.0)
        self.assertIn("hnr_db", result["features"])
        self.assertEqual(
            result["available_domains"],
            ["timing", "phonation"],
        )
        self.assertNotIn("spectral_centroid_hz", result["features"])
        self.assertNotIn("zero_crossing_rate", result["features"])

    def test_silence_does_not_enter_personal_baseline(self):
        with tempfile.TemporaryDirectory() as directory:
            wav_path = Path(directory) / "silence.wav"
            write_modulated_wav(wav_path, amplitude=0.0)
            monitor = PassiveSpeechMonitor(
                Path(directory) / "work",
                capture_backend=_BlockingCapture(),
                config=self.config(),
            )

            status = monitor.process_window(wav_path)

        self.assertEqual(status["baseline_windows"], 0)
        self.assertEqual(status["assessment"], "waiting_for_speech")
        self.assertFalse(status["latest_window"]["valid"])

    def test_calibrates_persists_and_reloads_personal_baseline(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "work"
            wav_path = Path(directory) / "baseline.wav"
            write_modulated_wav(wav_path)
            monitor = PassiveSpeechMonitor(
                root,
                capture_backend=_BlockingCapture(),
                config=self.config(),
            )

            for _ in range(3):
                status = monitor.process_window(wav_path)

            reloaded = PassiveSpeechMonitor(
                root,
                capture_backend=_BlockingCapture(),
                config=self.config(),
            ).snapshot()

        self.assertTrue(status["baseline_ready"])
        self.assertEqual(status["assessment"], "stable")
        self.assertEqual(reloaded["baseline_windows"], 3)
        self.assertTrue(reloaded["baseline_ready"])

    def test_one_changed_domain_does_not_count_as_anomaly(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "work"
            monitor = PassiveSpeechMonitor(
                root,
                capture_backend=_BlockingCapture(),
                config=self.config(),
            )
            for _ in range(3):
                monitor._accept_analysis(_analysis(), monitor.generation)

            monitor._accept_analysis(
                _analysis(pause_fraction=0.8),
                monitor.generation,
            )
            status = monitor.snapshot()

        self.assertEqual(status["assessment"], "stable")
        self.assertFalse(status["latest_window"]["anomaly"])
        self.assertEqual(
            status["latest_window"]["changed_domains"],
            ["timing"],
        )

    def test_two_changed_domains_and_multiple_windows_recommend_confirmation(self):
        with tempfile.TemporaryDirectory() as directory:
            monitor = PassiveSpeechMonitor(
                Path(directory) / "work",
                capture_backend=_BlockingCapture(),
                config=self.config(),
            )
            for _ in range(3):
                monitor._accept_analysis(_analysis(), monitor.generation)

            changed = _analysis(pause_fraction=0.8, hnr_db=-5.0)
            monitor._accept_analysis(changed, monitor.generation)
            first = monitor.snapshot()
            monitor._accept_analysis(changed, monitor.generation)
            second = monitor.snapshot()

        self.assertEqual(first["assessment"], "observing_change")
        self.assertFalse(first["recommend_guided_check"])
        self.assertEqual(second["assessment"], "suspected_change")
        self.assertTrue(second["recommend_guided_check"])
        self.assertEqual(
            second["latest_window"]["changed_domains"],
            ["phonation", "timing"],
        )
        self.assertFalse(second["clinical_validation"])
        self.assertEqual(second["evidence_version"], EVIDENCE_VERSION)

    def test_version_one_baseline_is_not_reused(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "passive-baseline.json").write_text(
                '{"version":1,"samples":[{"pause_fraction":0.1}]}',
                encoding="utf-8",
            )
            status = PassiveSpeechMonitor(
                root,
                capture_backend=_BlockingCapture(),
                config=self.config(),
            ).snapshot()

        self.assertEqual(status["baseline_windows"], 0)
        self.assertFalse(status["baseline_ready"])

    def test_pause_releases_an_in_progress_capture(self):
        with tempfile.TemporaryDirectory() as directory:
            capture = _BlockingCapture()
            monitor = PassiveSpeechMonitor(
                directory,
                capture_backend=capture,
                config=self.config(),
            )
            monitor.start()
            self.assertTrue(capture.started.wait(timeout=1.0))

            released = monitor.pause("guided_speech_check", timeout=1.0)
            status = monitor.snapshot()
            monitor.stop()

        self.assertTrue(released)
        self.assertEqual(status["state"], "paused")
        self.assertEqual(status["pause_reason"], "guided_speech_check")
        self.assertFalse(list(Path(directory).glob("passive-*.wav")))

    def test_background_monitor_deletes_routine_wav_after_analysis(self):
        with tempfile.TemporaryDirectory() as directory:
            monitor = PassiveSpeechMonitor(
                directory,
                capture_backend=_ImmediateCapture(),
                config=self.config(
                    baseline_windows=1,
                    interval_seconds=30.0,
                ),
            )
            monitor.start()
            deadline = time.monotonic() + 1.0
            while (
                monitor.snapshot()["baseline_windows"] < 1
                and time.monotonic() < deadline
            ):
                time.sleep(0.01)
            status = monitor.snapshot()
            monitor.stop()
            routine_wavs = list(Path(directory).glob("passive-*.wav"))

        self.assertTrue(status["baseline_ready"])
        self.assertFalse(status["raw_audio_retained"])
        self.assertEqual(routine_wavs, [])


def _analysis(**overrides):
    features = {
        "pause_fraction": 0.10,
        "mean_pause_duration_seconds": 0.18,
        "syllable_nuclei_rate_hz": 3.0,
        "intensity_range_db": 12.0,
        "f0_median_hz": 180.0,
        "f0_std_semitones": 3.0,
        "jitter_local": 0.015,
        "shimmer_local": 0.08,
        "hnr_db": 18.0,
        "formant_f1_iqr_hz": 250.0,
        "formant_f2_iqr_hz": 500.0,
    }
    features.update(overrides)
    return {
        "valid": True,
        "reason": "passive_speech_window_ready",
        "quality": 1.0,
        "metrics": dict(features),
        "features": features,
        "available_domains": [
            "timing",
            "phonation",
            "articulation_resonance",
        ],
        "estimator": EVIDENCE_VERSION,
    }


if __name__ == "__main__":
    unittest.main()
