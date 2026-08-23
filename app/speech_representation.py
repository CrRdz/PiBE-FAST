"""Lightweight Mandarin dysarthria representation and screening inference.

The model trained by ``training/speech/train_mdsc.py`` distinguishes speech
from MDSC Control and Dysarthria speakers. Its output can contribute to the
Speech screening status, but it is not an acute-stroke classifier or diagnosis.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from math import log10
from pathlib import Path
from typing import Any

import numpy as np

from app.speech_audio import _read_pcm_wav


SPEECH_REPRESENTATION_VERSION = "mdsc-logmel-v1"
NUM_MEL_BINS = 40
SAMPLE_RATE = 16_000
FRAME_LENGTH_MS = 25
FRAME_SHIFT_MS = 10


def feature_names(num_mel_bins: int = NUM_MEL_BINS) -> tuple[str, ...]:
    names = [f"logmel_mean_{index:02d}" for index in range(num_mel_bins)]
    names.extend(f"logmel_std_{index:02d}" for index in range(num_mel_bins))
    names.extend(
        (
            "duration_seconds",
            "rms_dbfs",
            "voiced_fraction",
            "pause_fraction",
            "mean_pause_seconds",
            "log_energy_std",
            "spectral_flux_mean",
            "zero_crossing_rate",
            "spectral_centroid_mean_hz",
            "spectral_centroid_std_hz",
        )
    )
    return tuple(names)


SPEECH_REPRESENTATION_FEATURES = feature_names()


@dataclass(frozen=True)
class DysarthriaPrediction:
    probability: float
    threshold: float
    model_version: str
    representation_version: str
    embedding: tuple[float, ...]

    @property
    def predicted_dysarthria(self) -> bool:
        return self.probability >= self.threshold

    def as_dict(self, *, include_embedding: bool = False) -> dict[str, Any]:
        value: dict[str, Any] = {
            "probability": round(self.probability, 6),
            "threshold": round(self.threshold, 6),
            "predicted_dysarthria": self.predicted_dysarthria,
            "model_version": self.model_version,
            "representation_version": self.representation_version,
            "medical_role": "dysarthria_speech_screening_component",
            "clinical_validation": False,
        }
        if include_embedding:
            value["embedding"] = [round(item, 6) for item in self.embedding]
        return value


class DysarthriaRepresentationModel:
    """Load and execute the compact JSON model exported from MDSC training."""

    def __init__(self, model_path: str | Path) -> None:
        self.model_path = Path(model_path)
        self.model_version = "unavailable"
        self.threshold = 0.5
        self.means = np.zeros(0, dtype=np.float64)
        self.scales = np.ones(0, dtype=np.float64)
        self.coefficients = np.zeros(0, dtype=np.float64)
        self.intercept = 0.0
        self._error: str | None = None
        self._load()

    def _load(self) -> None:
        if not self.model_path.is_file():
            self._error = f"dysarthria representation model not found: {self.model_path}"
            return
        try:
            payload = json.loads(self.model_path.read_text(encoding="utf-8"))
            if payload.get("component") != "speech_dysarthria_representation":
                raise ValueError("unexpected model component")
            if payload.get("representation_version") != SPEECH_REPRESENTATION_VERSION:
                raise ValueError("unsupported speech representation version")
            if tuple(payload.get("feature_names", ())) != SPEECH_REPRESENTATION_FEATURES:
                raise ValueError("speech feature order does not match runtime")
            means = np.asarray(payload["means"], dtype=np.float64)
            scales = np.asarray(payload["scales"], dtype=np.float64)
            coefficients = np.asarray(payload["coefficients"], dtype=np.float64)
            expected = len(SPEECH_REPRESENTATION_FEATURES)
            if any(array.shape != (expected,) for array in (means, scales, coefficients)):
                raise ValueError("invalid speech model vector dimensions")
            if not all(np.isfinite(array).all() for array in (means, scales, coefficients)):
                raise ValueError("speech model contains non-finite values")
            if np.any(scales <= 0.0):
                raise ValueError("speech model scales must be positive")
            self.means = means
            self.scales = scales
            self.coefficients = coefficients
            self.intercept = float(payload["intercept"])
            self.threshold = float(payload["decision_threshold"])
            self.model_version = str(payload["model_version"])
            if not (0.0 < self.threshold < 1.0):
                raise ValueError("speech model threshold must be in (0, 1)")
            if not np.isfinite(self.intercept):
                raise ValueError("speech model intercept must be finite")
        except Exception as exc:
            self._error = str(exc)

    def availability(self) -> tuple[bool, str | None]:
        return self._error is None, self._error

    def predict_wav(self, wav_path: str | Path) -> DysarthriaPrediction:
        ready, reason = self.availability()
        if not ready:
            raise RuntimeError(reason or "dysarthria representation model unavailable")
        vector = extract_speech_representation(wav_path)
        embedding = (vector - self.means) / self.scales
        logit = float(embedding @ self.coefficients + self.intercept)
        probability = _sigmoid(logit)
        return DysarthriaPrediction(
            probability=probability,
            threshold=self.threshold,
            model_version=self.model_version,
            representation_version=SPEECH_REPRESENTATION_VERSION,
            embedding=tuple(float(value) for value in embedding),
        )


def extract_speech_representation(wav_path: str | Path) -> np.ndarray:
    samples, sample_rate = _read_pcm_wav(Path(wav_path))
    if sample_rate != SAMPLE_RATE:
        samples = _linear_resample(samples, sample_rate, SAMPLE_RATE)
        sample_rate = SAMPLE_RATE
    if samples.size < int(sample_rate * 0.20):
        raise ValueError("speech representation requires at least 0.2 seconds")
    samples = np.asarray(samples, dtype=np.float64)
    samples = samples - float(np.mean(samples))
    peak = float(np.max(np.abs(samples)))
    if peak > 1.0:
        samples = samples / peak

    frame_length = int(sample_rate * FRAME_LENGTH_MS / 1000)
    frame_shift = int(sample_rate * FRAME_SHIFT_MS / 1000)
    frames = _frames(samples, frame_length, frame_shift)
    windowed = frames * np.hanning(frame_length)[None, :]
    spectrum = np.abs(np.fft.rfft(windowed, n=512)) ** 2
    mel_filters = _mel_filterbank(sample_rate, 512, NUM_MEL_BINS)
    # Some Apple Accelerate builds emit floating-point warnings for otherwise
    # finite matrix products; the explicit finite check below remains binding.
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        mel_energy = spectrum @ mel_filters.T
    logmel = np.log(np.maximum(mel_energy, 1e-10))

    energy = np.sqrt(np.mean(frames**2, axis=1) + 1e-12)
    log_energy = np.log(np.maximum(energy, 1e-9))
    noise_floor = float(np.percentile(energy, 20))
    threshold = min(max(noise_floor * 2.5, 10 ** (-50.0 / 20.0)), max(float(np.mean(energy)) * 0.45, 1e-5))
    active = energy >= threshold
    if not np.any(active):
        active = np.ones(len(frames), dtype=bool)
    active_logmel = logmel[active]

    pause_fraction, mean_pause = _pause_metrics(active, frame_shift / sample_rate)
    normalized_spectrum = spectrum / np.maximum(np.sum(spectrum, axis=1, keepdims=True), 1e-12)
    flux = np.sqrt(np.sum(np.diff(normalized_spectrum, axis=0) ** 2, axis=1))
    crossings = np.mean(np.abs(np.diff(np.signbit(frames), axis=1)), axis=1)
    frequencies = np.fft.rfftfreq(512, 1.0 / sample_rate)
    centroids = np.sum(normalized_spectrum * frequencies[None, :], axis=1)
    rms = float(np.sqrt(np.mean(samples**2) + 1e-12))

    values = [*np.mean(active_logmel, axis=0), *np.std(active_logmel, axis=0)]
    values.extend(
        (
            len(samples) / sample_rate,
            20.0 * log10(max(rms, 1e-9)),
            float(np.mean(active)),
            pause_fraction,
            mean_pause,
            float(np.std(log_energy[active])),
            float(np.mean(flux)) if flux.size else 0.0,
            float(np.mean(crossings[active])),
            float(np.mean(centroids[active])),
            float(np.std(centroids[active])),
        )
    )
    vector = np.asarray(values, dtype=np.float64)
    if vector.shape != (len(SPEECH_REPRESENTATION_FEATURES),) or not np.isfinite(vector).all():
        raise ValueError("speech representation produced invalid features")
    return vector


def _frames(samples: np.ndarray, length: int, shift: int) -> np.ndarray:
    if len(samples) < length:
        samples = np.pad(samples, (0, length - len(samples)))
    count = 1 + max(0, (len(samples) - length) // shift)
    shape = (count, length)
    strides = (samples.strides[0] * shift, samples.strides[0])
    return np.lib.stride_tricks.as_strided(samples, shape=shape, strides=strides).copy()


def _mel_filterbank(sample_rate: int, n_fft: int, bins: int) -> np.ndarray:
    low_mel = 2595.0 * np.log10(1.0)
    high_mel = 2595.0 * np.log10(1.0 + (sample_rate / 2.0) / 700.0)
    mel_points = np.linspace(low_mel, high_mel, bins + 2)
    hz = 700.0 * (10 ** (mel_points / 2595.0) - 1.0)
    indices = np.floor((n_fft + 1) * hz / sample_rate).astype(int)
    filters = np.zeros((bins, n_fft // 2 + 1), dtype=np.float64)
    for index in range(bins):
        left, center, right = indices[index : index + 3]
        center = max(center, left + 1)
        right = max(right, center + 1)
        for item in range(left, min(center, filters.shape[1])):
            filters[index, item] = (item - left) / max(center - left, 1)
        for item in range(center, min(right, filters.shape[1])):
            filters[index, item] = (right - item) / max(right - center, 1)
    return filters


def _pause_metrics(active: np.ndarray, seconds_per_frame: float) -> tuple[float, float]:
    indices = np.flatnonzero(active)
    if not indices.size:
        return 1.0, 0.0
    span = active[indices[0] : indices[-1] + 1]
    runs: list[int] = []
    current = 0
    for value in span:
        if not value:
            current += 1
        elif current:
            runs.append(current)
            current = 0
    if current:
        runs.append(current)
    minimum = max(1, int(round(0.12 / seconds_per_frame)))
    pauses = [run for run in runs if run >= minimum]
    return float(sum(pauses) / max(len(span), 1)), float(np.mean(pauses) * seconds_per_frame) if pauses else 0.0


def _linear_resample(samples: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate <= 0:
        raise ValueError("invalid source sample rate")
    target_count = max(1, int(round(len(samples) * target_rate / source_rate)))
    source_x = np.linspace(0.0, 1.0, len(samples), endpoint=False)
    target_x = np.linspace(0.0, 1.0, target_count, endpoint=False)
    return np.interp(target_x, source_x, samples).astype(np.float32)


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        return 1.0 / (1.0 + float(np.exp(-value)))
    exponential = float(np.exp(value))
    return exponential / (1.0 + exponential)
