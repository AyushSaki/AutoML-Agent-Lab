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
    result["prediction"] = predictions

    if hasattr(model, "predict_proba"):
        try:
            probabilities = model.predict_proba(data.loc[:, features].copy())
            for index, label in enumerate(model.classes_):
                result[f"probability_{label}"] = probabilities[:, index]
        except (ValueError, AttributeError):
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
