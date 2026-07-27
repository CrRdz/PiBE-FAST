from pathlib import Path
import json

import librosa
import numpy as np


PROCESSED_FOLDER = Path("processed")
FEATURE_FOLDER = Path("features")

FEATURE_FOLDER.mkdir(exist_ok=True)


def get_latest_processed_audio():

    wav_files = list(PROCESSED_FOLDER.glob("*_clean.wav"))

    if not wav_files:
        raise FileNotFoundError("No processed audio found.")

    return max(
        wav_files,
        key=lambda file: file.stat().st_mtime
    )


def extract_features(audio_file):

    print(f"Loading {audio_file.name}")

    audio, sample_rate = librosa.load(
        audio_file,
        sr=16000
    )

    duration = librosa.get_duration(
        y=audio,
        sr=sample_rate
    )

    # MFCC
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sample_rate,
        n_mfcc=13
    )

    # RMS Energy
    rms = librosa.feature.rms(
        y=audio
    )

    # Zero Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(
        audio
    )

    # Spectral Centroid
    centroid = librosa.feature.spectral_centroid(
        y=audio,
        sr=sample_rate
    )

    feature_data = {

        "sample_rate": sample_rate,

        "duration": float(duration),

        "mfcc_mean":

            np.mean(
                mfcc,
                axis=1
            ).tolist(),

        "rms_mean":

            float(
                np.mean(rms)
            ),

        "zero_crossing_rate":

            float(
                np.mean(zcr)
            ),

        "spectral_centroid":

            float(
                np.mean(centroid)
            )
    }

    return feature_data


def save_features(audio_file, feature_data):

    output_file = FEATURE_FOLDER / f"{audio_file.stem}.json"

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            feature_data,
            file,
            indent=4
        )

    return output_file


if __name__ == "__main__":

    audio_file = get_latest_processed_audio()

    features = extract_features(
        audio_file
    )

    output = save_features(
        audio_file,
        features
    )

    print()

    print("Feature Extraction Complete")

    print(f"Saved to {output}")

    print()

    print(json.dumps(
        features,
        indent=4
    ))