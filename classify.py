from datetime import datetime
from pathlib import Path
from typing import Any
import json

import joblib
import numpy as np


FEATURES_FOLDER = Path("features")
MODELS_FOLDER = Path("models")
RESULTS_FOLDER = Path("results")

MODEL_FILE = MODELS_FOLDER / "speech_model.joblib"


def get_latest_feature_file() -> Path:
    """
    Find the newest JSON feature file in the features folder.

    Returns:
        Path to the newest feature JSON file.
    """

    if not FEATURES_FOLDER.exists():
        raise FileNotFoundError(
            f"Features folder does not exist: "
            f"{FEATURES_FOLDER.resolve()}"
        )

    json_files = list(FEATURES_FOLDER.glob("*.json"))

    if not json_files:
        raise FileNotFoundError(
            f"No feature JSON files were found in: "
            f"{FEATURES_FOLDER.resolve()}"
        )

    return max(
        json_files,
        key=lambda file_path: file_path.stat().st_mtime,
    )


def load_features(feature_file: Path) -> dict[str, Any]:
    """
    Load extracted audio features from a JSON file.

    Args:
        feature_file: Path to the feature JSON file.

    Returns:
        Dictionary containing the extracted features.
    """

    if not feature_file.exists():
        raise FileNotFoundError(
            f"Feature file does not exist: {feature_file.resolve()}"
        )

    with feature_file.open("r", encoding="utf-8") as file:
        feature_data = json.load(file)

    return feature_data


def create_feature_vector(
    feature_data: dict[str, Any],
) -> tuple[np.ndarray, list[str]]:
    """
    Convert the feature dictionary into one ordered NumPy vector.

    The feature order must be exactly the same during:
    1. model training;
    2. model prediction.

    Returns:
        A tuple containing:
        - a two-dimensional NumPy array for model.predict();
        - a list containing the feature names.
    """

    required_fields = [
        "duration",
        "mfcc_mean",
        "rms_mean",
        "zero_crossing_rate",
        "spectral_centroid",
    ]

    missing_fields = [
        field
        for field in required_fields
        if field not in feature_data
    ]

    if missing_fields:
        raise ValueError(
            "The feature file is missing these fields: "
            + ", ".join(missing_fields)
        )

    mfcc_values = feature_data["mfcc_mean"]

    if not isinstance(mfcc_values, list):
        raise ValueError("'mfcc_mean' must be stored as a list.")

    if len(mfcc_values) != 13:
        raise ValueError(
            "Expected 13 MFCC values, "
            f"but received {len(mfcc_values)}."
        )

    feature_names = [
        "duration",
        *[
            f"mfcc_{index}"
            for index in range(1, 14)
        ],
        "rms_mean",
        "zero_crossing_rate",
        "spectral_centroid",
    ]

    feature_values = [
        float(feature_data["duration"]),
        *[float(value) for value in mfcc_values],
        float(feature_data["rms_mean"]),
        float(feature_data["zero_crossing_rate"]),
        float(feature_data["spectral_centroid"]),
    ]

    feature_vector = np.asarray(
        feature_values,
        dtype=np.float64,
    ).reshape(1, -1)

    if not np.all(np.isfinite(feature_vector)):
        raise ValueError(
            "The feature vector contains NaN or infinite values."
        )

    return feature_vector, feature_names


def load_model(model_file: Path = MODEL_FILE) -> Any:
    """
    Load the team's trained machine-learning model.

    Args:
        model_file: Path to a trusted joblib model file.

    Returns:
        Loaded model object.
    """

    if not model_file.exists():
        raise FileNotFoundError(
            f"Trained model not found: {model_file.resolve()}"
        )

    model = joblib.load(model_file)

    if not hasattr(model, "predict"):
        raise TypeError(
            "The loaded object does not provide a predict() method."
        )

    return model


def classify_features(
    model: Any,
    feature_vector: np.ndarray,
) -> dict[str, Any]:
    """
    Run the trained model on one feature vector.

    Args:
        model: A fitted model with a predict() method.
        feature_vector: Shape (1, number_of_features).

    Returns:
        Dictionary containing the prediction and optional confidence.
    """

    prediction = model.predict(feature_vector)[0]

    result: dict[str, Any] = {
        "prediction": str(prediction),
        "confidence": None,
        "class_probabilities": None,
    }

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(feature_vector)[0]
        classes = model.classes_

        probability_map = {
            str(class_name): float(probability)
            for class_name, probability in zip(
                classes,
                probabilities,
            )
        }

        result["class_probabilities"] = probability_map
        result["confidence"] = float(np.max(probabilities))

    return result


def save_result(
    feature_file: Path,
    feature_names: list[str],
    classification_result: dict[str, Any],
) -> Path:
    """
    Save the classification result as JSON.
    """

    RESULTS_FOLDER.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = RESULTS_FOLDER / f"result_{timestamp}.json"

    output_data = {
        "source_feature_file": str(feature_file),
        "classification_time": datetime.now().isoformat(
            timespec="seconds"
        ),
        "feature_order": feature_names,
        **classification_result,
        "notice": (
            "Research screening output only. "
            "This is not a medical diagnosis."
        ),
    }

    with output_file.open("w", encoding="utf-8") as file:
        json.dump(
            output_data,
            file,
            indent=4,
            ensure_ascii=False,
        )

    return output_file


def main() -> None:
    """
    Run the complete classification stage.
    """

    try:
        feature_file = get_latest_feature_file()

        print(f"Loading features: {feature_file}")

        feature_data = load_features(feature_file)

        feature_vector, feature_names = create_feature_vector(
            feature_data
        )

        print(
            f"Feature vector created: "
            f"{feature_vector.shape[1]} values"
        )

        if not MODEL_FILE.exists():
            print()
            print("Classification was not performed.")
            print(
                f"No trained model was found at: "
                f"{MODEL_FILE.resolve()}"
            )
            print(
                "Feature extraction is working, but your team "
                "must train and provide the model first."
            )
            return

        model = load_model()

        print("Running trained model...")

        result = classify_features(
            model,
            feature_vector,
        )

        result_file = save_result(
            feature_file,
            feature_names,
            result,
        )

        print()
        print("Classification complete.")
        print(f"Prediction: {result['prediction']}")

        if result["confidence"] is not None:
            confidence_percent = result["confidence"] * 100

            print(
                f"Model confidence: "
                f"{confidence_percent:.2f}%"
            )

        print(f"Result saved to: {result_file.resolve()}")
        print(
            "Important: this is a research screening output, "
            "not a medical diagnosis."
        )

    except (
        FileNotFoundError,
        ValueError,
        TypeError,
        json.JSONDecodeError,
    ) as error:
        print(f"Classification failed: {error}")

    except Exception as error:
        print(f"Unexpected classification error: {error}")


if __name__ == "__main__":
    main()