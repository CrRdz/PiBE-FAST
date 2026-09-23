"""Local-computer microphone capture and offline guided speech assessment."""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import ceil, log10
from pathlib import Path
import platform
import re
import shutil
import subprocess
import threading
import time
from typing import Any, Protocol
from uuid import uuid4
import wave

import numpy as np

from app.befast.result import MotionResult


class SpeechBackendUnavailable(RuntimeError):
    """Raised when a configured microphone or recognizer is not installed."""


@dataclass(frozen=True)
class SpeechAudioConfig:
    """Capture and engineering thresholds for the guided speech check."""

    sample_rate: int = 16_000
    capture_seconds: float = 7.0
    frame_ms: int = 20
    min_duration_seconds: float = 2.0
    min_voiced_seconds: float = 1.0
    min_rms_dbfs: float = -42.0
    max_clip_fraction: float = 0.02
    max_pause_fraction: float = 0.55
    max_character_error_rate: float = 0.35
    min_characters_per_second: float = 1.0
    max_characters_per_second: float = 8.0
    prompt_zh: str = "今天天气很好，我们一起去公园散步"
    prompt_en: str = "The sky is blue and we are walking in the park"

    def prompt(self, language: str) -> str:
        return self.prompt_en if language == "en" else self.prompt_zh


@dataclass(frozen=True)
class Transcript:
    text: str
    engine: str


class SpeechCaptureBackend(Protocol):
    def capture(
        self,
        output_path: Path,
        config: SpeechAudioConfig,
        cancel_event: threading.Event,
    ) -> None:
        """Capture one mono PCM WAV file."""


class SpeechRecognizer(Protocol):
    @property
    def name(self) -> str:
        """Return a stable engine name for reports."""

    def availability(self) -> tuple[bool, str | None]:
        """Return whether the engine is ready and an optional reason."""

    def recognize(self, wav_path: Path, language: str) -> Transcript:
        """Transcribe one WAV file locally."""


class SpeechRepresentationModel(Protocol):
    """MDSC dysarthria model whose prediction can control the S decision."""

    model_version: str

    def availability(self) -> tuple[bool, str | None]: ...

    def predict_wav(self, wav_path: str | Path) -> Any: ...


class AlsaMicrophoneCapture:
    """Capture a fixed-duration WAV with ALSA's arecord on Raspberry Pi OS."""

    def __init__(self, device: str = "default", executable: str = "arecord") -> None:
        self.device = str(device)
        self.executable = str(executable)

    def availability(self) -> tuple[bool, str | None]:
        if shutil.which(self.executable) is None:
            return False, "ALSA arecord is unavailable; install alsa-utils"
        return True, None

    def capture(
        self,
        output_path: Path,
        config: SpeechAudioConfig,
        cancel_event: threading.Event,
    ) -> None:
        available, reason = self.availability()
        executable = shutil.which(self.executable)
        if not available or executable is None:
            raise SpeechBackendUnavailable(reason or "ALSA arecord is unavailable")
        command = [
            executable,
            "--quiet",
            "--device",
            self.device,
            "--format",
            "S16_LE",
            "--rate",
            str(config.sample_rate),
            "--channels",
            "1",
            "--duration",
            str(max(1, int(ceil(config.capture_seconds)))),
            "--file-type",
            "wav",
            str(output_path),
        ]
        process = subprocess.Popen(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
        deadline = time.monotonic() + max(
            10.0,
            float(config.capture_seconds) + 10.0,
        )
        while process.poll() is None:
            if cancel_event.wait(0.1):
                process.terminate()
                try:
                    process.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=2.0)
                raise RuntimeError("speech capture cancelled")
            if time.monotonic() >= deadline:
                process.terminate()
                try:
                    process.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=2.0)
                detail = (
                    process.stderr.read() if process.stderr is not None else ""
                ).strip()
                raise RuntimeError(
                    detail
                    or (
                        "ALSA microphone capture timed out; check the arecord device"
                    )
                )
        stderr = (process.stderr.read() if process.stderr is not None else "").strip()
        if process.returncode != 0:
            raise RuntimeError(stderr or f"arecord exited with code {process.returncode}")
        if not output_path.is_file() or output_path.stat().st_size <= 44:
            raise RuntimeError("microphone capture did not produce a usable WAV file")


class AvfoundationMicrophoneCapture:
    """Capture a fixed-duration WAV from macOS through FFmpeg/AVFoundation."""

    def __init__(self, device: str = "default", executable: str = "ffmpeg") -> None:
        self.device = str(device)
        self.executable = str(executable)

    def availability(self) -> tuple[bool, str | None]:
        if platform.system() != "Darwin":
            return False, "AVFoundation microphone capture is available only on macOS"
        if shutil.which(self.executable) is None:
            return False, "FFmpeg is unavailable; install it with 'brew install ffmpeg'"
        return True, None

    def capture(
        self,
        output_path: Path,
        config: SpeechAudioConfig,
        cancel_event: threading.Event,
    ) -> None:
        available, reason = self.availability()
        executable = shutil.which(self.executable)
        if not available or executable is None:
            raise SpeechBackendUnavailable(
                reason or "FFmpeg AVFoundation capture is unavailable"
            )
        command = [
            executable,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "avfoundation",
            "-i",
            f":{'0' if self.device in {'', 'default'} else self.device}",
            "-t",
            str(max(1.0, float(config.capture_seconds))),
            "-ac",
            "1",
            "-ar",
            str(config.sample_rate),
            "-c:a",
            "pcm_s16le",
            str(output_path),
        ]
        process = subprocess.Popen(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
        deadline = time.monotonic() + max(
            10.0,
            float(config.capture_seconds) + 10.0,
        )
        while process.poll() is None:
            if cancel_event.wait(0.1):
                process.terminate()
                try:
                    process.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=2.0)
                raise RuntimeError("speech capture cancelled")
            if time.monotonic() >= deadline:
                process.terminate()
                try:
                    process.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=2.0)
                detail = (
                    process.stderr.read() if process.stderr is not None else ""
                ).strip()
                raise RuntimeError(
                    detail
                    or (
                        "FFmpeg microphone capture timed out; check the AVFoundation "
                        "device and macOS microphone permission"
                    )
                )
        stderr = (process.stderr.read() if process.stderr is not None else "").strip()
        if process.returncode != 0:
            raise RuntimeError(stderr or f"ffmpeg exited with code {process.returncode}")
        if not output_path.is_file() or output_path.stat().st_size <= 44:
            raise RuntimeError("microphone capture did not produce a usable WAV file")


def default_microphone_capture(device: str = "default") -> SpeechCaptureBackend:
    """Select the native microphone command for the service host."""

    if platform.system() == "Darwin":
        return AvfoundationMicrophoneCapture(device=device)
    return AlsaMicrophoneCapture(device=device)


class WhisperCppRecognizer:
    """Run a local whisper.cpp CLI without adding a Python ML dependency."""

    def __init__(
        self,
        model_path: str | Path,
        executable: str = "whisper-cli",
        timeout_seconds: float = 90.0,
    ) -> None:
        self.model_path = Path(model_path)
        self.executable = str(executable)
        self.timeout_seconds = float(timeout_seconds)

    @property
    def name(self) -> str:
        return "whisper_cpp"

    def availability(self) -> tuple[bool, str | None]:
        if shutil.which(self.executable) is None:
            return False, f"{self.executable} is not on PATH"
        if not self.model_path.is_file():
            return False, f"speech model not found: {self.model_path}"
        return True, None

    def recognize(self, wav_path: Path, language: str) -> Transcript:
        available, reason = self.availability()
        if not available:
            raise SpeechBackendUnavailable(reason or "whisper.cpp is unavailable")
        executable = shutil.which(self.executable)
        assert executable is not None
        command = [executable]
        # Homebrew's whisper.cpp enables Metal by default. A child process launched
        # by a sandboxed desktop host can fail GPU buffer allocation, while the CPU
        # backend is stable and still fast enough for this short guided check.
        if platform.system() == "Darwin":
            command.append("--no-gpu")
        command.extend(
            [
                "-m",
                str(self.model_path),
                "-f",
                str(wav_path),
                "-l",
                "en" if language == "en" else "zh",
                "-nt",
            ]
        )
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=self.timeout_seconds,
            check=False,
        )
        if completed.returncode != 0:
            detail = completed.stderr.strip() or completed.stdout.strip()
            raise RuntimeError(
                detail or f"whisper.cpp exited with code {completed.returncode}"
            )
        lines: list[str] = []
        for raw_line in completed.stdout.splitlines():
            line = re.sub(r"^\[[^\]]+\]\s*", "", raw_line.strip())
            if not line or line.startswith(("whisper_", "system_info", "main:")):
                continue
            lines.append(line)
        return Transcript(text=" ".join(lines).strip(), engine=self.name)


def analyze_speech_wav(
    wav_path: str | Path,
    *,
    transcript: Transcript | None,
    expected_text: str,
    config: SpeechAudioConfig | None = None,
) -> MotionResult:
    """Assess recording quality, fixed-phrase accuracy, rate, and pauses."""

    cfg = config or SpeechAudioConfig()
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
        frame_rms = np.sqrt(np.mean(np.square(framed), axis=1, dtype=np.float64))
    else:
        frame_rms = np.zeros(0, dtype=np.float64)
    noise_floor = float(np.percentile(frame_rms, 20)) if frame_rms.size else 0.0
    absolute_floor = 10 ** (cfg.min_rms_dbfs / 20.0)
    # Cap the adaptive threshold below the recording RMS so continuous speech
    # is not mistaken for a constant noise floor.
    activity_threshold = min(
        max(noise_floor * 2.5, absolute_floor),
        max(rms * 0.45, absolute_floor),
    )
    active = frame_rms >= activity_threshold
    voiced_seconds = float(np.sum(active) * cfg.frame_ms / 1000.0)
    if np.any(active):
        indices = np.flatnonzero(active)
        span = active[indices[0] : indices[-1] + 1]
        pause_fraction = float(1.0 - np.mean(span))
    else:
        pause_fraction = 1.0

    metrics = {
        "duration_seconds": duration,
        "rms_dbfs": rms_dbfs,
        "clip_fraction": clip_fraction,
        "voiced_seconds": voiced_seconds,
        "pause_fraction": pause_fraction,
        "sample_rate": float(sample_rate),
    }
    quality = min(
        1.0,
        max(0.0, duration / max(cfg.min_duration_seconds, 1e-6)),
        max(0.0, voiced_seconds / max(cfg.min_voiced_seconds, 1e-6)),
    )

    if duration < cfg.min_duration_seconds:
        return _speech_result(
            "insufficient", "speech_recording_too_short", quality, metrics
        )
    if rms_dbfs < cfg.min_rms_dbfs:
        return _speech_result(
            "insufficient", "speech_audio_too_quiet", quality, metrics
        )
    if clip_fraction > cfg.max_clip_fraction:
        return _speech_result(
            "insufficient", "speech_audio_clipped", quality, metrics
        )
    if voiced_seconds < cfg.min_voiced_seconds:
        return _speech_result(
            "insufficient", "speech_not_detected", quality, metrics
        )
    if transcript is None:
        return _speech_result(
            "insufficient", "speech_recognizer_unavailable", quality, metrics
        )

    expected = _normalize_transcript(expected_text)
    recognized = _normalize_transcript(transcript.text)
    details = {
        "expected_text": expected_text,
        "transcript": transcript.text,
        "recognizer": transcript.engine,
    }
    if not recognized:
        return _speech_result(
            "insufficient",
            "speech_transcription_empty",
            quality,
            metrics,
            details,
        )

    character_error_rate = _levenshtein(expected, recognized) / max(len(expected), 1)
    characters_per_second = len(recognized) / max(voiced_seconds, 1e-6)
    metrics.update(
        {
            "character_error_rate": character_error_rate,
            "characters_per_second": characters_per_second,
            "recognized_characters": float(len(recognized)),
            "expected_characters": float(len(expected)),
        }
    )
    if character_error_rate >= cfg.max_character_error_rate:
        return _speech_result(
            "positive", "speech_content_mismatch", quality, metrics, details
        )
    if (
        pause_fraction >= cfg.max_pause_fraction
        or characters_per_second < cfg.min_characters_per_second
        or characters_per_second > cfg.max_characters_per_second
    ):
        return _speech_result(
            "positive", "speech_delivery_irregular", quality, metrics, details
        )
    return _speech_result(
        "negative", "no_clear_speech_abnormality", quality, metrics, details
    )


_MDSC_AUDIO_QUALITY_FAILURE_REASONS = frozenset(
    {
        "speech_recording_too_short",
        "speech_audio_too_quiet",
        "speech_audio_clipped",
        "speech_not_detected",
        "microphone_unavailable",
        "speech_processing_failed",
    }
)


def _mdsc_audio_eligible(result: MotionResult) -> bool:
    """Return whether the recording passed the audio gates needed by MDSC."""

    return result.reason not in _MDSC_AUDIO_QUALITY_FAILURE_REASONS


def _apply_mdsc_prediction(result: MotionResult, prediction: Any) -> MotionResult:
    """Use qualified chronic-dysarthria screening evidence for guided S.

    This classifies the observable speech component, not stroke etiology.  The
    separately collected onset answer controls urgency downstream.
    """

    predicted = bool(prediction.predicted_dysarthria)
    trigger_recommended = bool(predicted and _mdsc_audio_eligible(result))
    return MotionResult(
        status="positive" if trigger_recommended else "negative",
        reason=(
            "mdsc_dysarthria_detected"
            if trigger_recommended
            else "mdsc_dysarthria_not_detected"
        ),
        affected_side=result.affected_side,
        quality=result.quality,
        metrics={
            **result.metrics,
            "mdsc_dysarthria_probability": prediction.probability,
            "mdsc_dysarthria_threshold": prediction.threshold,
        },
        details={
            **result.details,
            "mdsc_dysarthria_model": prediction.model_version,
            "mdsc_dysarthria_prediction": str(predicted).lower(),
            "mdsc_guided_check_recommended": str(trigger_recommended).lower(),
            "mdsc_medical_role": "guided_S_screening_evidence_not_stroke_specific",
        },
    )


class SpeechCaptureService:
    """Coordinate one asynchronous microphone capture and local assessment."""

    def __init__(
        self,
        root_dir: str | Path,
        *,
        capture_backend: SpeechCaptureBackend | None = None,
        recognizer: SpeechRecognizer | None = None,
        representation_model: SpeechRepresentationModel | None = None,
        config: SpeechAudioConfig | None = None,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.capture_backend = capture_backend or AlsaMicrophoneCapture()
        self.recognizer = recognizer
        self.representation_model = representation_model
        self.config = config or SpeechAudioConfig()
        self.lock = threading.RLock()
        self.cancel_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.state = "idle"
        self.started_at: float | None = None
        self.language = "zh"
        self.expected_text = self.config.prompt("zh")
        self.speech_attempt_id: str | None = None
        self.new_or_sudden: bool | None = None
        self.onset_time: str | None = None
        self.result: MotionResult | None = None
        self.audio_path: Path | None = None

    def start(
        self,
        *,
        language: str = "zh",
        new_or_sudden: bool | None = None,
        onset_time: str | None = None,
        speech_attempt_id: str | None = None,
    ) -> dict[str, Any]:
        normalized_language = "en" if language == "en" else "zh"
        with self.lock:
            if self.state in {"recording", "analyzing"}:
                raise ValueError("a speech recording is already in progress")
            self._discard_audio_locked()
            self.cancel_event = threading.Event()
            self.state = "recording"
            self.started_at = time.time()
            self.language = normalized_language
            self.expected_text = self.config.prompt(normalized_language)
            self.speech_attempt_id = speech_attempt_id
            self.new_or_sudden = new_or_sudden
            self.onset_time = str(onset_time).strip() if onset_time else None
            self.result = None
            self.audio_path = self.root_dir / f"speech-{uuid4().hex}.wav"
            cancel_event = self.cancel_event
            audio_path = self.audio_path
            expected_text = self.expected_text
            self.thread = threading.Thread(
                target=self._run,
                args=(
                    cancel_event,
                    audio_path,
                    normalized_language,
                    expected_text,
                ),
                daemon=True,
            )
            self.thread.start()
            return self.snapshot()

    def cancel(self) -> None:
        with self.lock:
            self.cancel_event.set()
            thread = self.thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=2.5)
        with self.lock:
            self.state = "idle"
            self.result = None
            self.thread = None
            self._discard_audio_locked()

    def wait(self, timeout: float = 15.0) -> bool:
        with self.lock:
            thread = self.thread
        if thread is not None:
            thread.join(timeout=timeout)
        with self.lock:
            return self.state == "ready"

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            elapsed = (
                max(0.0, time.time() - self.started_at)
                if self.started_at is not None
                else 0.0
            )
            recognizer_ready = False
            recognizer_reason: str | None = None
            recognizer_name: str | None = None
            capture_ready = True
            capture_reason: str | None = None
            capture_availability = getattr(
                self.capture_backend, "availability", None
            )
            if callable(capture_availability):
                try:
                    capture_ready, capture_reason = capture_availability()
                except Exception as exc:
                    capture_ready = False
                    capture_reason = str(exc)
            if self.recognizer is not None:
                recognizer_name = self.recognizer.name
                try:
                    recognizer_ready, recognizer_reason = (
                        self.recognizer.availability()
                    )
                except Exception as exc:
                    recognizer_reason = str(exc)
            representation_ready = False
            representation_reason: str | None = None
            representation_version: str | None = None
            if self.representation_model is not None:
                representation_version = self.representation_model.model_version
                try:
                    representation_ready, representation_reason = (
                        self.representation_model.availability()
                    )
                except Exception as exc:
                    representation_reason = str(exc)
            return {
                "state": self.state,
                "language": self.language,
                "expected_text": self.expected_text,
                "capture_seconds": self.config.capture_seconds,
                "elapsed_seconds": round(elapsed, 2),
                "progress": round(
                    min(1.0, elapsed / max(self.config.capture_seconds, 1e-6)), 4
                ),
                "capture_ready": capture_ready,
                "capture_reason": capture_reason,
                "recognizer": recognizer_name,
                "recognizer_ready": recognizer_ready,
                "recognizer_reason": recognizer_reason,
                "representation_model": representation_version,
                "representation_ready": representation_ready,
                "representation_reason": representation_reason,
                "representation_mode": "guided_S_dysarthria_screening_evidence",
                "result": self.result.as_dict() if self.result is not None else None,
            }

    def consume_result(
        self,
    ) -> tuple[MotionResult, Path | None, bool | None, str | None]:
        with self.lock:
            if self.state != "ready" or self.result is None:
                raise ValueError("speech assessment is not ready")
            result = self.result
            if self.speech_attempt_id is not None:
                result = replace(result, details={**result.details, "speech_attempt_id": self.speech_attempt_id})
            audio_path = self.audio_path
            sudden = self.new_or_sudden
            onset_time = self.onset_time
            self.state = "idle"
            self.started_at = None
            self.result = None
            self.audio_path = None
            self.thread = None
            return result, audio_path, sudden, onset_time

    @staticmethod
    def discard_consumed_audio(path: Path | None) -> None:
        if path is not None:
            path.unlink(missing_ok=True)

    def _run(
        self,
        cancel_event: threading.Event,
        path: Path,
        language: str,
        expected_text: str,
    ) -> None:
        try:
            self.capture_backend.capture(path, self.config, cancel_event)
            if cancel_event.is_set():
                return
            with self.lock:
                if self.cancel_event is not cancel_event:
                    return
                self.state = "analyzing"
            transcript: Transcript | None = None
            if self.recognizer is not None:
                available, _ = self.recognizer.availability()
                if available:
                    transcript = self.recognizer.recognize(path, language)
            result = analyze_speech_wav(
                path,
                transcript=transcript,
                expected_text=expected_text,
                config=self.config,
            )
            # A quality-qualified MDSC prediction is guided-S screening evidence.
            # It is not evidence of stroke etiology; transcript and timing values
            # remain audit features, and onset is handled by the session layer.
            if _mdsc_audio_eligible(result):
                model_ready = False
                model_reason = "dysarthria model not configured"
                if self.representation_model is not None:
                    model_ready, model_reason = self.representation_model.availability()
                if model_ready and self.representation_model is not None:
                    try:
                        prediction = self.representation_model.predict_wav(path)
                    except Exception as exc:
                        result = MotionResult(
                            status="insufficient",
                            reason="dysarthria_model_failed",
                            affected_side=result.affected_side,
                            quality=result.quality,
                            metrics=result.metrics,
                            details={
                                **result.details,
                                "mdsc_dysarthria_error": str(exc),
                            },
                        )
                    else:
                        result = _apply_mdsc_prediction(result, prediction)
                else:
                    result = MotionResult(
                        status="insufficient",
                        reason="dysarthria_model_unavailable",
                        quality=result.quality,
                        metrics=result.metrics,
                        details={
                            **result.details,
                            "mdsc_dysarthria_error": model_reason,
                            "asr_role": "audit_feature_only",
                        },
                    )
        except SpeechBackendUnavailable as exc:
            result = MotionResult(
                status="insufficient",
                reason="microphone_unavailable",
                details={"error": str(exc)},
            )
        except Exception as exc:
            if cancel_event.is_set():
                return
            result = MotionResult(
                status="insufficient",
                reason="speech_processing_failed",
                details={"error": str(exc)},
            )
        with self.lock:
            if self.cancel_event is cancel_event and not cancel_event.is_set():
                self.result = result
                self.state = "ready"

    def _discard_audio_locked(self) -> None:
        if self.audio_path is not None:
            self.audio_path.unlink(missing_ok=True)
        self.audio_path = None


def _read_pcm_wav(path: Path) -> tuple[np.ndarray, int]:
    try:
        with wave.open(str(path), "rb") as source:
            channels = source.getnchannels()
            sample_width = source.getsampwidth()
            sample_rate = source.getframerate()
            frame_count = source.getnframes()
            payload = source.readframes(frame_count)
    except (OSError, wave.Error) as exc:
        raise ValueError(f"unable to read speech WAV: {exc}") from exc
    if sample_width != 2:
        raise ValueError("speech WAV must use 16-bit PCM")
    if channels < 1:
        raise ValueError("speech WAV must contain at least one channel")
    raw = np.frombuffer(payload, dtype="<i2").astype(np.float32)
    if channels > 1:
        raw = raw[: (len(raw) // channels) * channels].reshape(-1, channels).mean(axis=1)
    return np.clip(raw / 32768.0, -1.0, 1.0), int(sample_rate)


def _normalize_transcript(value: str) -> str:
    return "".join(character.lower() for character in str(value) if character.isalnum())


def _levenshtein(left: str, right: str) -> int:
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for left_index, left_character in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_character in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1]
                    + int(left_character != right_character),
                )
            )
        previous = current
    return previous[-1]


def _speech_result(
    status: str,
    reason: str,
    quality: float,
    metrics: dict[str, float],
    details: dict[str, str] | None = None,
) -> MotionResult:
    return MotionResult(
        status=status,
        reason=reason,
        quality=max(0.0, min(1.0, quality)),
        metrics=metrics,
        details=details or {},
    )
