from datetime import datetime
from pathlib import Path

import sounddevice as sd
from scipy.io.wavfile import write


RECORDING_SECONDS = 5
SAMPLE_RATE = 16000
CHANNELS = 1

# Use None to let sounddevice use the system's default microphone.
# Change this to an integer, such as 1, only when you know the correct
# microphone device number.
INPUT_DEVICE = None

OUTPUT_DIRECTORY = Path("recordings")


def record_audio(
    duration: int = RECORDING_SECONDS,
    sample_rate: int = SAMPLE_RATE,
    input_device=INPUT_DEVICE,
) -> Path:
    """
    Record audio using a computer or USB microphone.

    Returns:
        Path: The location of the saved WAV file.

    Raises:
        ValueError: If duration or sample rate is invalid.
        RuntimeError: If recording or saving fails.
    """

    if duration <= 0:
        raise ValueError("Recording duration must be greater than zero.")

    if sample_rate <= 0:
        raise ValueError("Sample rate must be greater than zero.")

    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = OUTPUT_DIRECTORY / f"speech_{timestamp}.wav"

    number_of_samples = int(duration * sample_rate)

    try:
        print("Recording started...")
        print(f"Duration: {duration} seconds")
        print(f"Sample rate: {sample_rate} Hz")

        audio_data = sd.rec(
            frames=number_of_samples,
            samplerate=sample_rate,
            channels=CHANNELS,
            dtype="int16",
            device=input_device,
        )

        sd.wait()

        write(
            filename=output_file,
            rate=sample_rate,
            data=audio_data,
        )

    except Exception as error:
        raise RuntimeError(f"Recording failed: {error}") from error

    print("Recording finished.")
    print(f"Saved to: {output_file.resolve()}")

    return output_file


if __name__ == "__main__":
    try:
        record_audio()
    except (ValueError, RuntimeError) as error:
        print(error)