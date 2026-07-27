"""Local-first long-running natural-speech change monitoring.

This module deliberately detects changes from a personal acoustic baseline. It
does not diagnose dysarthria or stroke, and a sustained change is only a reason
to request the existing guided fixed-phrase speech check.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import json
from math import ceil, log10
from pathlib import Path
import threading
import time
from typing import Any
from uuid import uuid4

import numpy as np

from app.speech_audio import (
    SpeechAudioConfig,
    SpeechCaptureBackend,
    _read_pcm_wav,
)


EVIDENCE_VERSION = "stroke-speech-evidence-v2"

# These feature families mirror the motor-speech subsystems repeatedly examined
# in stroke/dysarthria acoustic studies. The lightweight estimators below are
# engineering approximations for edge monitoring; they are not clinical
# implementations of eGeMAPS, vowel-space analysis, or an SLP assessment.
FEATURE_DOMAINS = {
    "timing": (
        "pause_fraction",
        "mean_pause_duration_seconds",
        "syllable_nuclei_rate_hz",
    ),
    "phonation": (
        "intensity_range_db",
        "f0_median_hz",
        "f0_std_semitones",
        "jitter_local",
        "shimmer_local",
        "hnr_db",
    ),
    "articulation_resonance": (
        "formant_f1_iqr_hz",
        "formant_f2_iqr_hz",
    ),
}
PASSIVE_FEATURES = tuple(
    feature for features in FEATURE_DOMAINS.values() for feature in features
)

# A robust baseline with little natural variance still needs a non-zero scale.
# These engineering floors also prevent tiny numerical changes from becoming
# large anomaly scores.
FEATURE_SCALE_FLOORS = {
    "pause_fraction": 0.08,
    "mean_pause_duration_seconds": 0.12,
    "syllable_nuclei_rate_hz": 0.60,
    "intensity_range_db": 3.0,
    "f0_median_hz": 15.0,
    "f0_std_semitones": 1.5,
    "jitter_local": 0.008,
    "shimmer_local": 0.04,
    "hnr_db": 3.0,
    "formant_f1_iqr_hz": 80.0,
    "formant_f2_iqr_hz": 150.0,
}


@dataclass(frozen=True)
class PassiveSpeechConfig:
    """Capture cadence and conservative change-detection thresholds."""

    sample_rate: int = 16_000
    window_seconds: float = 8.0
    interval_seconds: float = 1.0
    retry_seconds: float = 5.0
    frame_ms: int = 20
    min_duration_seconds: float = 2.0
    min_voiced_seconds: float = 1.5
    min_rms_dbfs: float = -42.0
    max_clip_fraction: float = 0.02
    baseline_windows: int = 6
    baseline_max_windows: int = 48
    recent_windows: int = 5
    anomaly_votes_required: int = 3
    anomaly_z_threshold: float = 3.5
    min_abnormal_domains: int = 2


def analyze_passive_speech_wav(
    wav_path: str | Path,
    config: PassiveSpeechConfig | None = None,
) -> dict[str, Any]:
    """Extract lightweight natural-speech features from one local WAV window."""

    cfg = config or PassiveSpeechConfig()
    samples, sample_rate = _read_pcm_wav(Path(wav_path))
    duration = len(samples) / max(sample_rate, 1)
    absolute = np.abs(samples)
    rms = (
        float(np.sqrt(np.mean(np.square(samples), dtype=np.float64)))
        if samples.size
        else 0.0
    )
    rms_dbfs = 20.0 * log10(max(rms, 1e-9))
    clip_fraction = float(np.mean(absolute >= 0.999)) if samples.size else 0.0

    frame_size = max(1, int(sample_rate * cfg.frame_ms / 1000))
    frame_count = len(samples) // frame_size
    if frame_count:
        framed = samples[: frame_count * frame_size].reshape(frame_count, frame_size)
        frame_rms = np.sqrt(
            np.mean(np.square(framed), axis=1, dtype=np.float64)
        )
    else:
        framed = np.zeros((0, frame_size), dtype=np.float32)
        frame_rms = np.zeros(0, dtype=np.float64)

    noise_floor = float(np.percentile(frame_rms, 20)) if frame_rms.size else 0.0
    absolute_floor = 10 ** (cfg.min_rms_dbfs / 20.0)
    activity_threshold = min(
        max(noise_floor * 2.5, absolute_floor),
        max(rms * 0.45, absolute_floor),
    )
    active = _smooth_activity(
        frame_rms >= activity_threshold,
        frame_ms=cfg.frame_ms,
    )
    voiced_seconds = float(np.sum(active) * cfg.frame_ms / 1000.0)
    voiced_fraction = voiced_seconds / max(duration, 1e-6)
    if np.any(active):
        indices = np.flatnonzero(active)
        active_span = active[indices[0] : indices[-1] + 1]
        pauses = _inactive_run_lengths(active_span)
        minimum_pause_frames = max(
            1,
            int(ceil(120.0 / max(cfg.frame_ms, 1))),
        )
        pause_frames = [
            run_length
            for run_length in pauses
            if run_length >= minimum_pause_frames
        ]
        pause_seconds = [
            run_length * cfg.frame_ms / 1000.0 for run_length in pause_frames
        ]
        pause_fraction = float(
            sum(pause_frames) / max(len(active_span), 1)
        )
        mean_pause_duration_seconds = (
            float(np.mean(pause_seconds)) if pause_seconds else 0.0
        )
    else:
        pause_fraction = 1.0
        mean_pause_duration_seconds = 0.0

    metrics = {
        "duration_seconds": duration,
        "rms_dbfs": rms_dbfs,
        "clip_fraction": clip_fraction,
        "voiced_seconds": voiced_seconds,
        "sample_rate": float(sample_rate),
    }
    quality = min(
        1.0,
        max(0.0, duration / max(cfg.min_duration_seconds, 1e-6)),
        max(0.0, voiced_seconds / max(cfg.min_voiced_seconds, 1e-6)),
    )

    reason = "passive_speech_window_ready"
    if duration < cfg.min_duration_seconds:
        reason = "passive_speech_window_too_short"
    elif rms_dbfs < cfg.min_rms_dbfs:
        reason = "passive_speech_audio_too_quiet"
    elif clip_fraction > cfg.max_clip_fraction:
        reason = "passive_speech_audio_clipped"
    elif voiced_seconds < cfg.min_voiced_seconds:
        reason = "passive_speech_not_detected"
    if reason != "passive_speech_window_ready":
        return {
            "valid": False,
            "reason": reason,
            "quality": round(quality, 4),
            "metrics": _rounded(metrics),
            "features": {},
        }

    active_rms = np.maximum(frame_rms[active], 1e-9)
    active_db = 20.0 * np.log10(active_rms)
    intensity_range_db = float(
        np.percentile(active_db, 95) - np.percentile(active_db, 5)
    )

    syllable_nuclei_rate_hz = _energy_peak_rate(
        frame_rms,
        active,
        cfg.frame_ms,
        voiced_seconds,
        activity_threshold,
    )
    phonation = _estimate_phonation(
        samples,
        sample_rate,
        activity_threshold,
    )
    formants = _estimate_formant_distribution(
        samples,
        sample_rate,
        activity_threshold,
    )
    features: dict[str, float] = {
        "pause_fraction": pause_fraction,
        "mean_pause_duration_seconds": mean_pause_duration_seconds,
        "syllable_nuclei_rate_hz": syllable_nuclei_rate_hz,
        "intensity_range_db": intensity_range_db,
    }
    features.update(phonation["features"])
    features.update(formants)
    metrics.update(features)
    metrics.update(phonation["metrics"])
    available_domains = [
        domain
        for domain, names in FEATURE_DOMAINS.items()
        if any(name in features for name in names)
    ]
    return {
        "valid": True,
        "reason": reason,
        "quality": round(quality, 4),
        "metrics": _rounded(metrics),
        "features": _rounded(features),
        "available_domains": available_domains,
        "estimator": EVIDENCE_VERSION,
    }


class PassiveSpeechMonitor:
    """Continuously capture local windows and detect sustained baseline changes."""

    BASELINE_VERSION = 2

    def __init__(
        self,
        root_dir: str | Path,
        *,
        capture_backend: SpeechCaptureBackend,
        config: PassiveSpeechConfig | None = None,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.baseline_path = self.root_dir / "passive-baseline.json"
        self.capture_backend = capture_backend
        self.config = config or PassiveSpeechConfig()
        self.lock = threading.RLock()
        self.condition = threading.Condition(self.lock)
        self.stop_event = threading.Event()
        self.capture_cancel_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.state = "stopped"
        self.assessment = "calibrating"
        self.paused = False
        self.pause_reason: str | None = None
        self.capturing = False
        self.started_at: float | None = None
        self.last_window_at: float | None = None
        self.last_error: str | None = None
        self.latest_window: dict[str, Any] | None = None
        self.baseline_samples: list[dict[str, float]] = []
        self.recent_anomalies: deque[bool] = deque(
            maxlen=max(
                1,
                int(self.config.recent_windows),
                int(self.config.anomaly_votes_required),
            )
        )
        self.recommend_guided_check = False
        self.suspected_since: float | None = None
        self.generation = 0
        self._load_baseline()

    def start(self) -> dict[str, Any]:
        with self.condition:
            if self.thread is not None and self.thread.is_alive():
                return self.snapshot()
            self.stop_event = threading.Event()
            self.capture_cancel_event = threading.Event()
            self.paused = False
            self.pause_reason = None
            self.started_at = time.time()
            self.last_error = None
            self.state = "listening"
            self.assessment = (
                "stable"
                if len(self.baseline_samples) >= self.config.baseline_windows
                else "calibrating"
            )
            self.thread = threading.Thread(
                target=self._run,
                name="passive-speech-monitor",
                daemon=True,
            )
            self.thread.start()
            return self.snapshot()

    def stop(self, timeout: float = 4.0) -> None:
        with self.condition:
            self.stop_event.set()
            self.capture_cancel_event.set()
            self.condition.notify_all()
            thread = self.thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=timeout)
        with self.condition:
            self.state = "stopped"
            self.capturing = False
            self.thread = None
            self.condition.notify_all()

    def pause(self, reason: str = "manual_pause", timeout: float = 4.0) -> bool:
        deadline = time.monotonic() + max(0.0, float(timeout))
        with self.condition:
            self.paused = True
            self.pause_reason = str(reason)
            self.capture_cancel_event.set()
            self.condition.notify_all()
            while self.capturing:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self.condition.wait(timeout=min(0.2, remaining))
            self.state = "paused"
            return True

    def resume(self) -> dict[str, Any]:
        with self.condition:
            self.paused = False
            self.pause_reason = None
            self.last_error = None
            if self.thread is not None and self.thread.is_alive():
                self.state = "listening"
            self.condition.notify_all()
            return self.snapshot()

    def reset_baseline(self) -> dict[str, Any]:
        with self.condition:
            self.generation += 1
            self.baseline_samples.clear()
            self.recent_anomalies.clear()
            self.latest_window = None
            self.recommend_guided_check = False
            self.suspected_since = None
            self.assessment = "calibrating"
            self.baseline_path.unlink(missing_ok=True)
            return self.snapshot()

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            capture_ready = True
            capture_reason: str | None = None
            availability = getattr(self.capture_backend, "availability", None)
            if callable(availability):
                try:
                    capture_ready, capture_reason = availability()
                except Exception as exc:
                    capture_ready = False
                    capture_reason = str(exc)
            profile = _baseline_profile(self.baseline_samples)
            return {
                "enabled": True,
                "state": self.state,
                "assessment": self.assessment,
                "paused": self.paused,
                "pause_reason": self.pause_reason,
                "capture_ready": bool(capture_ready),
                "capture_reason": capture_reason,
                "window_seconds": self.config.window_seconds,
                "interval_seconds": self.config.interval_seconds,
                "baseline_windows": len(self.baseline_samples),
                "baseline_target": self.config.baseline_windows,
                "baseline_ready": len(self.baseline_samples)
                >= self.config.baseline_windows,
                "baseline_profile": profile,
                "recent_anomaly_votes": int(sum(self.recent_anomalies)),
                "recent_window_count": len(self.recent_anomalies),
                "anomaly_votes_required": self.config.anomaly_votes_required,
                "anomaly_z_threshold": self.config.anomaly_z_threshold,
                "min_abnormal_domains": self.config.min_abnormal_domains,
                "recommend_guided_check": self.recommend_guided_check,
                "suspected_since": self.suspected_since,
                "last_window_at": self.last_window_at,
                "latest_window": (
                    dict(self.latest_window)
                    if self.latest_window is not None
                    else None
                ),
                "last_error": self.last_error,
                "raw_audio_retained": False,
                "speaker_verification": False,
                "medical_role": "change_detection_trigger_only",
                "clinical_validation": False,
                "evidence_version": EVIDENCE_VERSION,
                "feature_domains": {
                    name: list(features)
                    for name, features in FEATURE_DOMAINS.items()
                },
                "trigger_policy": (
                    "engineering_threshold_two_or_more_domains_then_"
                    "multi_window_vote"
                ),
            }

    def process_window(self, wav_path: str | Path) -> dict[str, Any]:
        """Analyze one WAV synchronously; useful for local files and tests."""

        with self.lock:
            generation = self.generation
        analysis = analyze_passive_speech_wav(wav_path, self.config)
        self._accept_analysis(analysis, generation)
        return self.snapshot()

    def _run(self) -> None:
        capture_config = SpeechAudioConfig(
            sample_rate=self.config.sample_rate,
            capture_seconds=max(2.0, float(self.config.window_seconds)),
            frame_ms=self.config.frame_ms,
            min_duration_seconds=self.config.min_duration_seconds,
            min_voiced_seconds=self.config.min_voiced_seconds,
            min_rms_dbfs=self.config.min_rms_dbfs,
            max_clip_fraction=self.config.max_clip_fraction,
        )
        while not self.stop_event.is_set():
            with self.condition:
                while self.paused and not self.stop_event.is_set():
                    self.state = "paused"
                    self.condition.wait(timeout=0.5)
                if self.stop_event.is_set():
                    break
                self.capture_cancel_event = threading.Event()
                cancel_event = self.capture_cancel_event
                generation = self.generation
                self.capturing = True
                self.state = "capturing"
                self.condition.notify_all()

            path = self.root_dir / f"passive-{uuid4().hex}.wav"
            failed = False
            try:
                self.capture_backend.capture(path, capture_config, cancel_event)
                if cancel_event.is_set() or self.stop_event.is_set():
                    continue
                analysis = analyze_passive_speech_wav(path, self.config)
                self._accept_analysis(analysis, generation)
            except Exception as exc:
                if not cancel_event.is_set() and not self.stop_event.is_set():
                    failed = True
                    with self.lock:
                        self.state = "error"
                        self.assessment = "unavailable"
                        self.last_error = str(exc)
            finally:
                path.unlink(missing_ok=True)
                with self.condition:
                    self.capturing = False
                    if not self.stop_event.is_set() and not self.paused and not failed:
                        self.state = "listening"
                    self.condition.notify_all()

            delay = (
                self.config.retry_seconds if failed else self.config.interval_seconds
            )
            if self.stop_event.wait(max(0.0, float(delay))):
                break
        with self.condition:
            self.capturing = False
            self.state = "stopped"
            self.condition.notify_all()

    def _accept_analysis(self, analysis: dict[str, Any], generation: int) -> None:
        now = time.time()
        save_baseline = False
        with self.lock:
            if generation != self.generation:
                return
            self.last_window_at = now
            self.last_error = None
            window = {
                **analysis,
                "captured_at": round(now, 4),
                "anomaly": False,
                "anomaly_score": 0.0,
                "changed_features": [],
                "changed_domains": [],
                "feature_scores": {},
                "domain_scores": {},
            }
            if not analysis.get("valid"):
                self.latest_window = window
                self.assessment = "waiting_for_speech"
                return

            features = {
                name: float(value)
                for name, value in analysis.get("features", {}).items()
                if name in PASSIVE_FEATURES and np.isfinite(float(value))
            }
            if len(self.baseline_samples) < self.config.baseline_windows:
                self.baseline_samples.append(features)
                self.baseline_samples = self.baseline_samples[
                    -max(
                        1,
                        int(self.config.baseline_windows),
                        int(self.config.baseline_max_windows),
                    ) :
                ]
                self.latest_window = window
                self.assessment = (
                    "stable"
                    if len(self.baseline_samples) >= self.config.baseline_windows
                    else "calibrating"
                )
                save_baseline = True
            else:
                feature_scores = _feature_scores(
                    features,
                    self.baseline_samples,
                )
                anomaly_score = max(feature_scores.values(), default=0.0)
                changed = sorted(
                    name
                    for name, score in feature_scores.items()
                    if score >= self.config.anomaly_z_threshold
                )
                domain_scores = {
                    domain: max(
                        (
                            feature_scores[name]
                            for name in names
                            if name in feature_scores
                        ),
                        default=0.0,
                    )
                    for domain, names in FEATURE_DOMAINS.items()
                }
                changed_domains = sorted(
                    domain
                    for domain, score in domain_scores.items()
                    if score >= self.config.anomaly_z_threshold
                )
                anomaly = (
                    len(changed_domains)
                    >= max(1, int(self.config.min_abnormal_domains))
                )
                self.recent_anomalies.append(anomaly)
                votes = int(sum(self.recent_anomalies))
                sustained = (
                    len(self.recent_anomalies)
                    >= self.config.anomaly_votes_required
                    and votes >= self.config.anomaly_votes_required
                )
                self.recommend_guided_check = sustained
                if sustained:
                    if self.suspected_since is None:
                        self.suspected_since = now
                    self.assessment = "suspected_change"
                elif anomaly:
                    self.suspected_since = None
                    self.assessment = "observing_change"
                else:
                    self.suspected_since = None
                    self.assessment = "stable"
                window.update(
                    {
                        "anomaly": anomaly,
                        "anomaly_score": round(anomaly_score, 4),
                        "changed_features": changed,
                        "changed_domains": changed_domains,
                        "feature_scores": _rounded(feature_scores),
                        "domain_scores": _rounded(domain_scores),
                    }
                )
                self.latest_window = window
        if save_baseline:
            self._save_baseline(generation)

    def _load_baseline(self) -> None:
        try:
            payload = json.loads(self.baseline_path.read_text(encoding="utf-8"))
            if int(payload.get("version", 0)) != self.BASELINE_VERSION:
                return
            samples = payload.get("samples")
            if not isinstance(samples, list):
                return
            loaded: list[dict[str, float]] = []
            for sample in samples:
                if not isinstance(sample, dict):
                    continue
                values = {
                    name: float(sample[name])
                    for name in PASSIVE_FEATURES
                    if name in sample and np.isfinite(float(sample[name]))
                }
                if values:
                    loaded.append(values)
            self.baseline_samples = loaded[
                -max(
                    1,
                    int(self.config.baseline_windows),
                    int(self.config.baseline_max_windows),
                ) :
            ]
            if len(self.baseline_samples) >= self.config.baseline_windows:
                self.assessment = "stable"
        except (OSError, ValueError, TypeError):
            self.baseline_samples = []

    def _save_baseline(self, generation: int) -> None:
        with self.lock:
            if generation != self.generation:
                return
            payload = {
                "version": self.BASELINE_VERSION,
                "updated_at": time.time(),
                "features": list(PASSIVE_FEATURES),
                "samples": list(self.baseline_samples),
            }
            temporary = self.baseline_path.with_suffix(".tmp")
            temporary.write_text(
                json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
                encoding="utf-8",
            )
            temporary.replace(self.baseline_path)


def _energy_peak_rate(
    frame_rms: np.ndarray,
    active: np.ndarray,
    frame_ms: int,
    voiced_seconds: float,
    activity_threshold: float,
) -> float:
    if frame_rms.size < 3 or not np.any(active):
        return 0.0
    smooth_frames = max(1, int(round(80.0 / max(frame_ms, 1))))
    kernel = np.ones(smooth_frames, dtype=np.float64) / smooth_frames
    envelope = np.convolve(frame_rms, kernel, mode="same")
    active_envelope = envelope[active]
    peak_threshold = max(
        activity_threshold,
        float(np.percentile(active_envelope, 55)) * 0.9,
    )
    minimum_distance = max(1, int(ceil(120.0 / max(frame_ms, 1))))
    last_peak = -minimum_distance
    peaks = 0
    for index in range(1, len(envelope) - 1):
        if not active[index] or envelope[index] < peak_threshold:
            continue
        if (
            envelope[index] < envelope[index - 1]
            or envelope[index] <= envelope[index + 1]
        ):
            continue
        if index - last_peak < minimum_distance:
            continue
        peaks += 1
        last_peak = index
    return float(peaks / max(voiced_seconds, 1e-6))


def _smooth_activity(active: np.ndarray, *, frame_ms: int) -> np.ndarray:
    """Suppress clicks and bridge brief gaps before computing speech timing."""

    smoothed = np.asarray(active, dtype=bool).copy()
    if smoothed.size == 0:
        return smoothed
    maximum_gap = max(1, int(round(100.0 / max(frame_ms, 1))))
    minimum_burst = max(1, int(round(60.0 / max(frame_ms, 1))))
    for start, end, value in _boolean_runs(smoothed):
        if (
            not value
            and start > 0
            and end < len(smoothed)
            and end - start <= maximum_gap
        ):
            smoothed[start:end] = True
    for start, end, value in _boolean_runs(smoothed):
        if value and end - start < minimum_burst:
            smoothed[start:end] = False
    return smoothed


def _boolean_runs(mask: np.ndarray) -> list[tuple[int, int, bool]]:
    if mask.size == 0:
        return []
    boundaries = np.flatnonzero(mask[1:] != mask[:-1]) + 1
    starts = np.concatenate(([0], boundaries))
    ends = np.concatenate((boundaries, [len(mask)]))
    return [
        (int(start), int(end), bool(mask[start]))
        for start, end in zip(starts, ends)
    ]


def _inactive_run_lengths(mask: np.ndarray) -> list[int]:
    return [
        end - start
        for start, end, value in _boolean_runs(mask)
        if not value
    ]


def _estimate_phonation(
    samples: np.ndarray,
    sample_rate: int,
    activity_threshold: float,
) -> dict[str, dict[str, float]]:
    """Estimate lightweight F0, perturbation, and harmonicity statistics.

    This autocorrelation estimator is intentionally dependency-free for a
    Raspberry Pi. Values are useful for within-speaker change monitoring but
    are not interchangeable with clinical or openSMILE/eGeMAPS measurements.
    """

    frame_size = max(1, int(round(sample_rate * 0.040)))
    hop_size = max(1, int(round(sample_rate * 0.020)))
    if len(samples) < frame_size or sample_rate <= 0:
        return {"features": {}, "metrics": {"pitch_frame_count": 0.0}}

    minimum_lag = max(1, int(sample_rate / 400.0))
    maximum_lag = min(frame_size - 2, int(sample_rate / 75.0))
    if maximum_lag <= minimum_lag:
        return {"features": {}, "metrics": {"pitch_frame_count": 0.0}}

    analysis_window = np.hanning(frame_size).astype(np.float64)
    fft_size = 1 << (2 * frame_size - 1).bit_length()
    periods: list[float] = []
    amplitudes: list[float] = []
    harmonicities: list[float] = []
    frame_indices: list[int] = []

    for frame_index, start in enumerate(
        range(0, len(samples) - frame_size + 1, hop_size)
    ):
        frame = np.asarray(
            samples[start : start + frame_size],
            dtype=np.float64,
        )
        amplitude = float(np.sqrt(np.mean(np.square(frame))))
        if amplitude < activity_threshold:
            continue
        centered = (frame - float(np.mean(frame))) * analysis_window
        spectrum = np.fft.rfft(centered, n=fft_size)
        autocorrelation = np.fft.irfft(
            spectrum * np.conjugate(spectrum),
            n=fft_size,
        )[:frame_size]
        energy = float(autocorrelation[0])
        if energy <= 1e-12:
            continue
        normalized = autocorrelation / energy
        search = normalized[minimum_lag : maximum_lag + 1]
        lag = int(np.argmax(search)) + minimum_lag
        peak = float(normalized[lag])
        if peak < 0.30:
            continue
        periods.append(lag / sample_rate)
        amplitudes.append(amplitude)
        harmonicities.append(
            10.0 * log10(max(peak, 1e-4) / max(1.0 - peak, 1e-4))
        )
        frame_indices.append(frame_index)

    pitch_frame_count = len(periods)
    metrics: dict[str, float] = {"pitch_frame_count": float(pitch_frame_count)}
    if pitch_frame_count < 5:
        return {"features": {}, "metrics": metrics}

    period_values = np.asarray(periods, dtype=np.float64)
    f0_values = 1.0 / np.maximum(period_values, 1e-9)
    median_f0 = float(np.median(f0_values))
    semitones = 12.0 * np.log2(f0_values / max(median_f0, 1e-9))
    metrics["f0_median_hz"] = median_f0

    adjacent = np.diff(np.asarray(frame_indices, dtype=np.int64)) == 1
    period_differences = np.abs(np.diff(period_values))[adjacent]
    amplitude_values = np.asarray(amplitudes, dtype=np.float64)
    amplitude_differences = np.abs(np.diff(amplitude_values))[adjacent]

    features: dict[str, float] = {
        "f0_median_hz": median_f0,
        "f0_std_semitones": float(np.std(semitones)),
        "hnr_db": float(np.median(harmonicities)),
    }
    if period_differences.size >= 4:
        features["jitter_local"] = float(
            np.mean(period_differences) / max(np.mean(period_values), 1e-9)
        )
    if amplitude_differences.size >= 4:
        features["shimmer_local"] = float(
            np.mean(amplitude_differences)
            / max(np.mean(amplitude_values), 1e-9)
        )
    return {"features": features, "metrics": metrics}


def _estimate_formant_distribution(
    samples: np.ndarray,
    sample_rate: int,
    activity_threshold: float,
) -> dict[str, float]:
    """Return exploratory F1/F2 variability from connected-speech LPC frames."""

    frame_size = max(1, int(round(sample_rate * 0.030)))
    base_hop = max(1, int(round(sample_rate * 0.020)))
    starts = list(range(0, len(samples) - frame_size + 1, base_hop))
    if not starts:
        return {}
    sample_every = max(1, int(ceil(len(starts) / 120.0)))
    order = min(max(10, 2 + sample_rate // 1000), frame_size - 2)
    window = np.hanning(frame_size).astype(np.float64)
    first_formants: list[float] = []
    second_formants: list[float] = []

    for start in starts[::sample_every]:
        frame = np.asarray(
            samples[start : start + frame_size],
            dtype=np.float64,
        )
        if float(np.sqrt(np.mean(np.square(frame)))) < activity_threshold:
            continue
        emphasized = np.empty_like(frame)
        emphasized[0] = frame[0]
        emphasized[1:] = frame[1:] - 0.97 * frame[:-1]
        emphasized = (emphasized - float(np.mean(emphasized))) * window
        autocorrelation = np.correlate(emphasized, emphasized, mode="full")
        autocorrelation = autocorrelation[frame_size - 1 : frame_size + order]
        if autocorrelation.size < order + 1 or autocorrelation[0] <= 1e-10:
            continue
        toeplitz = autocorrelation[
            np.abs(
                np.arange(order)[:, None] - np.arange(order)[None, :]
            )
        ]
        toeplitz += np.eye(order) * autocorrelation[0] * 1e-6
        try:
            predictor = np.linalg.solve(toeplitz, autocorrelation[1:])
        except np.linalg.LinAlgError:
            continue
        roots = np.roots(np.concatenate(([1.0], -predictor)))
        roots = roots[np.imag(roots) > 0.0]
        frequencies = np.angle(roots) * sample_rate / (2.0 * np.pi)
        bandwidths = (
            -0.5
            * sample_rate
            / np.pi
            * np.log(np.maximum(np.abs(roots), 1e-9))
        )
        candidates = sorted(
            float(frequency)
            for frequency, bandwidth in zip(frequencies, bandwidths)
            if 90.0 < frequency < 4000.0 and 20.0 < bandwidth < 800.0
        )
        f1 = next(
            (frequency for frequency in candidates if 150.0 <= frequency <= 1200.0),
            None,
        )
        f2 = next(
            (
                frequency
                for frequency in candidates
                if f1 is not None
                and max(600.0, f1 + 200.0) <= frequency <= 3500.0
            ),
            None,
        )
        if f1 is not None and f2 is not None:
            first_formants.append(f1)
            second_formants.append(f2)

    if len(first_formants) < 5:
        return {}
    return {
        "formant_f1_iqr_hz": float(
            np.percentile(first_formants, 75)
            - np.percentile(first_formants, 25)
        ),
        "formant_f2_iqr_hz": float(
            np.percentile(second_formants, 75)
            - np.percentile(second_formants, 25)
        ),
    }


def _baseline_profile(
    samples: list[dict[str, float]],
) -> dict[str, dict[str, float]]:
    if not samples:
        return {}
    profile: dict[str, dict[str, float]] = {}
    minimum_observations = min(3, len(samples))
    for name in PASSIVE_FEATURES:
        observed = [
            float(sample[name])
            for sample in samples
            if name in sample and np.isfinite(float(sample[name]))
        ]
        if len(observed) < minimum_observations:
            continue
        values = np.asarray(observed, dtype=np.float64)
        median = float(np.median(values))
        mad = float(np.median(np.abs(values - median)))
        profile[name] = {
            "median": round(median, 5),
            "scale": round(
                max(1.4826 * mad, FEATURE_SCALE_FLOORS[name]),
                5,
            ),
            "sample_count": len(observed),
        }
    return profile


def _feature_scores(
    features: dict[str, float],
    baseline_samples: list[dict[str, float]],
) -> dict[str, float]:
    profile = _baseline_profile(baseline_samples)
    return {
        name: abs(float(features[name]) - values["median"])
        / max(values["scale"], 1e-9)
        for name, values in profile.items()
        if name in features
    }


def _rounded(values: dict[str, float]) -> dict[str, float]:
    return {key: round(float(value), 5) for key, value in values.items()}
