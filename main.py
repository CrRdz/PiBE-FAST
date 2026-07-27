from pathlib import Path
import sys

from audio import record_audio
from preprocess import preprocess_audio
from features import extract_features, save_features

from classify import (
    MODEL_FILE,
    create_feature_vector,
    load_model,
    classify_features,
    save_result,
)


def print_stage(stage_number: int, title: str) -> None:
    """
    Print a clear heading for each pipeline stage.
    """

    print()
    print("=" * 60)
    print(f"Stage {stage_number}: {title}")
    print("=" * 60)


def check_recording_quality(audio_file: Path) -> bool:
    """
    Perform a simple quality check on the recorded WAV file.

    This temporary version confirms that:
    - the recording exists;
    - the file is not empty.

    Later, this function can call analyze.py directly.
    """

    if not audio_file.exists():
        print(f"Recording not found: {audio_file}")
        return False

    file_size = audio_file.stat().st_size

    if file_size <= 44:
        # A basic WAV header is approximately 44 bytes.
        print("The recording appears to be empty.")
        return False

    print(f"Recording found: {audio_file}")
    print(f"File size: {file_size} bytes")
    print("Basic recording check passed.")

    return True


def run_pipeline() -> None:
    """
    Run the complete speech-processing pipeline.
    """

    print()
    print("BE-FAST Speech Screening Prototype")
    print("Research prototype only — not a medical diagnosis.")

    # ---------------------------------------------------------
    # Stage 1: Record audio
    # ---------------------------------------------------------

    print_stage(1, "Record audio")

    recording_file = record_audio()

    # ---------------------------------------------------------
    # Stage 2: Check recording quality
    # ---------------------------------------------------------

    print_stage(2, "Check recording quality")

    quality_passed = check_recording_quality(recording_file)

    if not quality_passed:
        print()
        print("The recording did not pass the quality check.")
        print("Please record the speech again.")
        return

    # ---------------------------------------------------------
    # Stage 3: Preprocess audio
    # ---------------------------------------------------------

    print_stage(3, "Preprocess audio")

    processed_file = preprocess_audio(recording_file)

    # ---------------------------------------------------------
    # Stage 4: Extract features
    # ---------------------------------------------------------

    print_stage(4, "Extract speech features")

    feature_data = extract_features(processed_file)

    feature_file = save_features(
        processed_file,
        feature_data,
    )

    print(f"Features saved to: {feature_file.resolve()}")

    # ---------------------------------------------------------
    # Stage 5: Prepare classifier input
    # ---------------------------------------------------------

    print_stage(5, "Prepare classification input")

    feature_vector, feature_names = create_feature_vector(
        feature_data
    )

    print(
        f"Feature vector created successfully: "
        f"{feature_vector.shape[1]} values"
    )

    # ---------------------------------------------------------
    # Stage 6: Run classifier
    # ---------------------------------------------------------

    print_stage(6, "Run speech-abnormality classifier")

    if not MODEL_FILE.exists():
        print("No trained speech model is currently available.")
        print(f"Expected model location: {MODEL_FILE.resolve()}")
        print()
        print("Recording, preprocessing, and feature extraction")
        print("were completed successfully.")
        print()
        print("Classification was skipped until your team provides")
        print("the trained Random Forest model.")
        return

    model = load_model(MODEL_FILE)

    classification_result = classify_features(
        model,
        feature_vector,
    )

    result_file = save_result(
        feature_file,
        feature_names,
        classification_result,
    )

    # ---------------------------------------------------------
    # Final output
    # ---------------------------------------------------------

    print()
    print("=" * 60)
    print("Pipeline complete")
    print("=" * 60)

    print(
        f"Prediction: "
        f"{classification_result['prediction']}"
    )

    confidence = classification_result.get("confidence")

    if confidence is not None:
        print(f"Model confidence: {confidence * 100:.2f}%")

    print(f"Result saved to: {result_file.resolve()}")

    print()
    print(
        "Important: This result is produced by a research "
        "screening prototype and is not a medical diagnosis."
    )


def main() -> None:
    """
    Start the pipeline and handle expected errors.
    """

    try:
        run_pipeline()

    except KeyboardInterrupt:
        print()
        print("Pipeline cancelled by the user.")
        sys.exit(1)

    except FileNotFoundError as error:
        print()
        print(f"Required file not found: {error}")
        sys.exit(1)

    except ValueError as error:
        print()
        print(f"Invalid data: {error}")
        sys.exit(1)

    except RuntimeError as error:
        print()
        print(f"Pipeline error: {error}")
        sys.exit(1)

    except Exception as error:
        print()
        print(f"Unexpected error: {error}")
        sys.exit(1)


if __name__ == "__main__":
    main()