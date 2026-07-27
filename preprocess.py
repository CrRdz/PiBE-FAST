from pathlib import Path

import librosa
import soundfile as sf


RECORDINGS_FOLDER = Path("recordings")
PROCESSED_FOLDER = Path("processed")

TARGET_SAMPLE_RATE = 16000


def get_latest_recording() -> Path:
    """
    Find the newest WAV file inside the recordings folder.
    """

    if not RECORDINGS_FOLDER.exists():
        raise FileNotFoundError(
            f"Recordings folder not found: {RECORDINGS_FOLDER.resolve()}"
        )

    wav_files = list(RECORDINGS_FOLDER.glob("*.wav"))

    if not wav_files:
        raise FileNotFoundError(
            f"No WAV files found in: {RECORDINGS_FOLDER.resolve()}"
        )

    latest_file = max(
        wav_files,
        key=lambda file_path: file_path.stat().st_mtime
    )

    return latest_file


def preprocess_audio(input_file: Path) -> Path:
    """
    Load, normalise, trim, and save an audio recording.

    Args:
        input_file: Path to the original WAV file.

    Returns:
        Path to the processed WAV file.
    """

    if not input_file.exists():
        raise FileNotFoundError(
            f"Input audio file not found: {input_file.resolve()}"
        )

    PROCESSED_FOLDER.mkdir(parents=True, exist_ok=True)

    output_file = (
        PROCESSED_FOLDER
        / f"{input_file.stem}_clean.wav"
    )

    print(f"Loading audio: {input_file}")

    audio, sample_rate = librosa.load(
        input_file,
        sr=TARGET_SAMPLE_RATE,
        mono=True
    )

    if len(audio) == 0:
        raise ValueError("The audio file is empty.")

    print("Normalising volume...")

    audio = librosa.util.normalize(audio)

    print("Removing leading and trailing silence...")

    trimmed_audio, _ = librosa.effects.trim(
        audio,
        top_db=20
    )

    if len(trimmed_audio) == 0:
        raise ValueError(
            "No usable audio remained after silence trimming."
        )

    print("Saving processed audio...")

    sf.write(
        output_file,
        trimmed_audio,
        TARGET_SAMPLE_RATE
    )

    print("Preprocessing finished.")
    print(f"Saved to: {output_file.resolve()}")

    return output_file


if __name__ == "__main__":
    try:
        latest_recording = get_latest_recording()
        preprocess_audio(latest_recording)

    except (FileNotFoundError, ValueError) as error:
        print(f"Preprocessing failed: {error}")

    except Exception as error:
        print(f"Unexpected error: {error}")