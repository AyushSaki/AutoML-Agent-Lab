"""Run predictions using an exported AutoML model."""

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

HERE = Path(__file__).resolve().parent


def predict(input_csv, output_csv):
    metadata = json.loads(
        (HERE / "metadata.json").read_text(encoding="utf-8")
    )
    features = metadata["features"]
    data = pd.read_csv(input_csv)

    missing = [name for name in features if name not in data.columns]
    if missing:
        raise ValueError(
            "Input CSV is missing required feature columns: "
            + ", ".join(missing)
        )

    model = joblib.load(HERE / "model.joblib")
    predictions = model.predict(data.loc[:, features].copy())

    result = data.copy()
    # LabelEncoder in the training pipeline maps original labels to
    # integers. Restore original labels only when the mapping is verifiable.
    class_names = metadata.get("classes") or []
    class_lookup = {index: str(name) for index, name in enumerate(class_names)}

    def encoded_class_id(value):
        try:
            numeric = float(value)
            if numeric.is_integer():
                return int(numeric)
        except (TypeError, ValueError, OverflowError):
            pass
        return None

    prediction_ids = [encoded_class_id(value) for value in predictions]
    can_restore_labels = (
        metadata.get("task") == "classification"
        and bool(class_lookup)
        and all(value in class_lookup for value in prediction_ids)
    )

    if can_restore_labels:
        result["prediction"] = [
            class_lookup[value] for value in prediction_ids
        ]
    else:
        result["prediction"] = predictions

    if hasattr(model, "predict_proba"):
        try:
            probabilities = model.predict_proba(data.loc[:, features].copy())
            model_classes = list(model.classes_)

            for index, label in enumerate(model_classes):
                class_id = encoded_class_id(label)
                if can_restore_labels and class_id in class_lookup:
                    column_label = class_lookup[class_id]
                else:
                    column_label = str(label)

                result[f"probability_{column_label}"] = probabilities[:, index]
        except (ValueError, AttributeError, IndexError):
            pass

    result.to_csv(output_csv, index=False)
    print(f"Predictions saved to: {output_csv}")
    print(f"Rows predicted: {len(result)}")
    print(f"Target: {metadata.get('target', 'prediction')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Predict with an exported AutoML model."
    )
    parser.add_argument("--input", required=True, help="Input CSV file")
    parser.add_argument("--output", default="predictions.csv")
    args = parser.parse_args()
    predict(args.input, args.output)
