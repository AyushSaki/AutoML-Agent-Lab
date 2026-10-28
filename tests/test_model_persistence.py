import json

import numpy as np
import pandas as pd
import pytest

import ml.pipeline as pipeline


@pytest.mark.parametrize("task", ["classification", "regression"])
def test_train_save_reload_and_predict(tmp_path, monkeypatch, task):
    monkeypatch.setattr(pipeline, "MODEL_DIR", tmp_path / "models")

    rng = np.random.default_rng(42)
    rows = 80

    df = pd.DataFrame({
        "age": rng.integers(18, 65, rows),
        "income": rng.integers(20000, 120000, rows),
        "segment": rng.choice(["A", "B", "C"], rows),
    })

    if task == "classification":
        target = "approved"
        df[target] = np.where(
            (df["income"] > 65000) & (df["age"] > 25),
            "Yes",
            "No",
        )
    else:
        target = "price"
        df[target] = (
            df["income"] * 1.5
            + df["age"] * 200
            + rng.normal(0, 1000, rows)
        )

    training_file = tmp_path / "training.csv"
    prediction_file = tmp_path / "new_data.csv"

    df.to_csv(training_file, index=False)
    df.drop(columns=[target]).head(5).to_csv(
        prediction_file, index=False
    )

    result = pipeline.train_models(
        training_file,
        target=target,
        task=task,
    )

    saved = result["saved_model"]
    model_id = saved["model_id"]
    model_dir = pipeline.MODEL_DIR

    model_file = model_dir / saved["filename"]
    metadata_file = model_dir / saved["metadata_filename"]

    assert model_file.is_file()
    assert metadata_file.is_file()

    metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
    assert metadata["model_id"] == model_id
    assert metadata["target"] == target
    assert metadata["task"] == task

    prediction = pipeline.predict_with_saved_model(
        model_id, prediction_file
    )

    assert prediction["rows"] == 5
    assert len(prediction["predictions"]) == 5
    assert prediction["target"] == target

    if task == "classification":
        assert set(prediction["predictions"]).issubset({"Yes", "No"})
    else:
        assert all(
            np.isfinite(float(value))
            for value in prediction["predictions"]
        )


def test_prediction_rejects_invalid_model_id(tmp_path):
    prediction_file = tmp_path / "new_data.csv"
    pd.DataFrame({"feature": [1, 2]}).to_csv(
        prediction_file, index=False
    )

    with pytest.raises(ValueError, match="Invalid model ID"):
        pipeline.predict_with_saved_model(
            "../invalid", prediction_file
        )


def test_prediction_rejects_missing_features(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "MODEL_DIR", tmp_path / "models")
    model_dir = pipeline.MODEL_DIR
    model_dir.mkdir()

    model_id = "a" * 32
    (model_dir / f"{model_id}.joblib").touch()
    (model_dir / f"{model_id}.json").write_text(
        json.dumps({
            "model_id": model_id,
            "model": "test",
            "target": "target",
            "task": "regression",
            "features": ["required_feature"],
            "classes": None,
        }),
        encoding="utf-8",
    )

    prediction_file = tmp_path / "new_data.csv"
    pd.DataFrame({"other_feature": [1, 2]}).to_csv(
        prediction_file, index=False
    )

    with pytest.raises(ValueError, match="missing required columns"):
        pipeline.predict_with_saved_model(
            model_id, prediction_file
        )
