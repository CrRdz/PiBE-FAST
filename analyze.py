from pathlib import Path

import numpy as np
from scipy.io.wavfile import read

RECORDINGS_FOLDER = Path("recordings")

MIN_DURATION = 2.0
MIN_AVERAGE_VOLUME = 100
CLIPPING_LIMIT = 32000


def find_latest_recording(folder):
    wav_files = list(folder.glob("*.wav"))

    if not wav_files:
        return None

    return max(wav_files, key=lambda file: file.stat().st_mtime)


def analyze_audio(file_path):
    sample_rate, audio_data = read(file_path)

    if audio_data.ndim > 1:
        audio_data = audio_data[:, 0]

    audio_data = audio_data.astype(np.float32)

    duration = len(audio_data) / sample_rate
    average_volume = np.mean(np.abs(audio_data))
    maximum_volume = np.max(np.abs(audio_data))

    speech_detected = average_volume >= MIN_AVERAGE_VOLUME
    too_short = duration < MIN_DURATION
    clipping_detected = maximum_volume >= CLIPPING_LIMIT

    print(f"Analyzing: {file_path.name}")
    print(f"Sample rate: {sample_rate} Hz")
    print(f"Duration: {duration:.2f} seconds")
    print(f"Average volume: {average_volume:.2f}")
    print(f"Maximum volume: {maximum_volume:.2f}")
    print(f"Speech detected: {'Yes' if speech_detected else 'No'}")

    if not speech_detected:
        print("Result: Please record again. No clear speech detected.")
    elif too_short:
        print("Result: Please record again. Recording is too short.")
    elif clipping_detected:
        print("Result: Please record again. Audio is too loud or distorted.")
    else:
        print("Result: Recording quality is acceptable.")


latest_file = find_latest_recording(RECORDINGS_FOLDER)

if latest_file is None:
    print("No WAV recordings were found.")
else:
    analyze_audio(latest_file)