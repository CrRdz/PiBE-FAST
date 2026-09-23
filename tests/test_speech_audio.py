import io
import tempfile
import threading
import unittest
import wave
from pathlib import Path
from subprocess import CompletedProcess
from unittest.mock import Mock, patch

import numpy as np

from app.speech_audio import (
    AlsaMicrophoneCapture,
    AvfoundationMicrophoneCapture,
    SpeechAudioConfig,
    SpeechCaptureService,
    Transcript,
    WhisperCppRecognizer,
    analyze_speech_wav,
    default_microphone_capture,
)
from app.speech_representation import DysarthriaPrediction


PROMPT = "今天天气很好，我们一起去公园散步"


def write_wav(path: Path, *, seconds: float = 3.0, amplitude: float = 0.25):
    sample_rate = 16000
    count = int(sample_rate * seconds)
    samples = (
        np.sin(2.0 * np.pi * 220.0 * np.arange(count) / sample_rate)
        * amplitude
        * 32767
    ).astype("<i2")
    with wave.open(str(path), "wb") as target:
        target.setnchannels(1)
        target.setsampwidth(2)
        target.setframerate(sample_rate)
        target.writeframes(samples.tobytes())


class _Capture:
    def capture(self, output_path, config, cancel_event):
        write_wav(Path(output_path), seconds=3.0)


class _Recognizer:
    name = "test_offline_asr"

    def __init__(self, text=PROMPT, available=True):
        self.text = text
        self.available = available

    def availability(self):
        return self.available, None if self.available else "model missing"

    def recognize(self, wav_path, language):
        return Transcript(self.text, self.name)


class _RepresentationModel:
    model_version = "test-mdsc"

    def availability(self):
        return True, None

    def predict_wav(self, wav_path):
        return DysarthriaPrediction(
            0.8, 0.7, self.model_version, "mdsc-logmel-v1", ()
        )


class SpeechAudioTest(unittest.TestCase):
    def test_default_capture_backend_uses_avfoundation_on_macos(self):
        with patch("app.speech_audio.platform.system", return_value="Darwin"):
            backend = default_microphone_capture("default")

        self.assertIsInstance(backend, AvfoundationMicrophoneCapture)
        self.assertEqual(backend.device, "default")

    def test_default_capture_backend_keeps_alsa_on_linux(self):
        with patch("app.speech_audio.platform.system", return_value="Linux"):
            backend = default_microphone_capture("plughw:1,0")

        self.assertIsInstance(backend, AlsaMicrophoneCapture)
        self.assertEqual(backend.device, "plughw:1,0")

    def test_avfoundation_default_uses_first_audio_device(self):
        with tempfile.TemporaryDirectory() as directory:
            wav_path = Path(directory) / "speech.wav"
            write_wav(wav_path)
            process = Mock()
            process.poll.return_value = 0
            process.returncode = 0
            process.stderr = io.StringIO("")
            backend = AvfoundationMicrophoneCapture("default")

            with (
                patch("app.speech_audio.shutil.which", return_value="/bin/ffmpeg"),
                patch(
                    "app.speech_audio.subprocess.Popen",
                    return_value=process,
                ) as popen,
            ):
                backend.capture(
                    wav_path,
                    SpeechAudioConfig(capture_seconds=3.0),
                    threading.Event(),
                )

        command = popen.call_args.args[0]
        self.assertEqual(command[command.index("-i") + 1], ":0")

    def test_whisper_cpp_disables_gpu_on_macos(self):
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "ggml-base.bin"
            wav_path = Path(directory) / "speech.wav"
            model_path.write_bytes(b"model")
            write_wav(wav_path)
            recognizer = WhisperCppRecognizer(model_path)

            with (
                patch("app.speech_audio.platform.system", return_value="Darwin"),
                patch("app.speech_audio.shutil.which", return_value="/bin/whisper-cli"),
                patch(
                    "app.speech_audio.subprocess.run",
                    return_value=CompletedProcess([], 0, "测试文本\n", ""),
                ) as run,
            ):
                transcript = recognizer.recognize(wav_path, "zh")

        self.assertIn("--no-gpu", run.call_args.args[0])
        self.assertEqual(transcript.text, "测试文本")

    def test_whisper_cpp_keeps_default_backend_on_linux(self):
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "ggml-base.bin"
            wav_path = Path(directory) / "speech.wav"
            model_path.write_bytes(b"model")
            write_wav(wav_path)
            recognizer = WhisperCppRecognizer(model_path)

            with (
                patch("app.speech_audio.platform.system", return_value="Linux"),
                patch("app.speech_audio.shutil.which", return_value="/bin/whisper-cli"),
                patch(
                    "app.speech_audio.subprocess.run",
                    return_value=CompletedProcess([], 0, "test text\n", ""),
                ) as run,
            ):
                recognizer.recognize(wav_path, "en")

        self.assertNotIn("--no-gpu", run.call_args.args[0])

    def test_matching_prompt_is_negative(self):
        with tempfile.TemporaryDirectory() as directory:
            wav_path = Path(directory) / "speech.wav"
            write_wav(wav_path)

            result = analyze_speech_wav(
                wav_path,
                transcript=Transcript(PROMPT, "test_offline_asr"),
                expected_text=PROMPT,
            )

            self.assertEqual(result.status, "negative")
            self.assertEqual(result.reason, "no_clear_speech_abnormality")
            self.assertEqual(result.details["transcript"], PROMPT)
            self.assertEqual(result.metrics["character_error_rate"], 0.0)

    def test_large_repetition_mismatch_is_positive(self):
        with tempfile.TemporaryDirectory() as directory:
            wav_path = Path(directory) / "speech.wav"
            write_wav(wav_path)

            result = analyze_speech_wav(
                wav_path,
                transcript=Transcript("今天", "test_offline_asr"),
                expected_text=PROMPT,
            )

            self.assertEqual(result.status, "positive")
            self.assertEqual(result.reason, "speech_content_mismatch")
            self.assertGreater(result.metrics["character_error_rate"], 0.35)

    def test_silence_is_insufficient_not_negative(self):
        with tempfile.TemporaryDirectory() as directory:
            wav_path = Path(directory) / "silence.wav"
            write_wav(wav_path, amplitude=0.0)

            result = analyze_speech_wav(
                wav_path,
                transcript=Transcript(PROMPT, "test_offline_asr"),
                expected_text=PROMPT,
            )

            self.assertEqual(result.status, "insufficient")
            self.assertEqual(result.reason, "speech_audio_too_quiet")

    def test_missing_recognizer_is_insufficient_not_negative(self):
        with tempfile.TemporaryDirectory() as directory:
            wav_path = Path(directory) / "speech.wav"
            write_wav(wav_path)

            result = analyze_speech_wav(
                wav_path,
                transcript=None,
                expected_text=PROMPT,
            )

            self.assertEqual(result.status, "insufficient")
            self.assertEqual(result.reason, "speech_recognizer_unavailable")

    def test_capture_service_runs_asynchronously_and_returns_audio(self):
        with tempfile.TemporaryDirectory() as directory:
            service = SpeechCaptureService(
                directory,
                capture_backend=_Capture(),
                recognizer=_Recognizer(),
                config=SpeechAudioConfig(capture_seconds=3.0),
            )

            service.start(language="zh", new_or_sudden=True)
            self.assertTrue(service.wait(timeout=2.0))
            result, audio_path, sudden, onset_time = service.consume_result()

            self.assertEqual(result.status, "insufficient")
            self.assertEqual(result.reason, "dysarthria_model_unavailable")
            self.assertTrue(sudden)
            self.assertIsNone(onset_time)
            self.assertIsNotNone(audio_path)
            self.assertTrue(audio_path.is_file())
            service.discard_consumed_audio(audio_path)
            self.assertFalse(audio_path.exists())

    def test_mdsc_positive_is_guided_S_screening_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            service = SpeechCaptureService(
                directory,
                capture_backend=_Capture(),
                recognizer=_Recognizer(),
                representation_model=_RepresentationModel(),
                config=SpeechAudioConfig(capture_seconds=3.0),
            )
            service.start(language="zh")
            self.assertTrue(service.wait(timeout=2.0))
            result, audio_path, _, _ = service.consume_result()
            self.assertEqual(result.status, "positive")
            self.assertEqual(result.reason, "mdsc_dysarthria_detected")
            self.assertEqual(result.metrics["mdsc_dysarthria_probability"], 0.8)
            self.assertEqual(
                result.details["mdsc_medical_role"],
                "guided_S_screening_evidence_not_stroke_specific",
            )
            self.assertEqual(
                result.details["mdsc_guided_check_recommended"], "true"
            )
            self.assertEqual(
                service.snapshot()["representation_mode"],
                "guided_S_dysarthria_screening_evidence",
            )
            service.discard_consumed_audio(audio_path)


if __name__ == "__main__":
    unittest.main()
