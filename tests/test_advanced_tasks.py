import numpy as np
import pandas as pd
import pytest

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from ml.advanced_tasks import run_clustering, run_anomaly_detection
from ml.tuning import tune_model
from ml.explainability import explain_model


def sample_data():
    X = pd.DataFrame({
        "signal": [0, 1, 2, 3, 7, 8, 9, 10, 11, 12, 13, 14],
        "noise":  [3, 5, 2, 6, 1, 4, 3, 7, 2, 5, 1, 6],
    })
    y = pd.Series([0] * 6 + [1] * 6)
    return X, y


def test_clustering_returns_one_label_per_row():
    X, _ = sample_data()
    result = run_clustering(X, n_clusters=2)
    assert len(result["labels"]) == len(X)
    assert result["metrics"]["clusters_found"] == 2


def test_anomaly_detection_returns_one_label_per_row():
    X, _ = sample_data()
    result = run_anomaly_detection(X, contamination=0.1)
    assert len(result["labels"]) == len(X)
    assert result["metrics"]["anomaly_count"] >= 1


def test_classification_tuning():
    X, y = sample_data()
    estimator = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000),
    )
    search, result = tune_model(
        estimator,
        X,
        y,
        {"logisticregression__C": [0.1, 1.0, 10.0]},
        cv=3,
        n_jobs=1,
    )
    assert search.best_estimator_ is not None
    assert result["task"] == "classification"
    assert result["candidates_tested"] == 3


def test_regression_tuning():
    X, _ = sample_data()
    y = pd.Series(2 * X["signal"] + X["noise"])
    _, result = tune_model(
        RandomForestRegressor(random_state=42),
        X,
        y,
        {"n_estimators": [10, 20], "max_depth": [2, None]},
        cv=3,
        n_jobs=1,
    )
    assert result["task"] == "regression"
    assert result["candidates_tested"] == 4


def test_permutation_importance():
    X, y = sample_data()
    model = RandomForestClassifier(n_estimators=30, random_state=42)
    model.fit(X, y)
    result = explain_model(model, X, y, n_repeats=3, n_jobs=1)
    assert len(result["features"]) == X.shape[1]
    assert {item["feature"] for item in result["features"]} == set(X.columns)


def test_tuning_rejects_invalid_search_type():
    X, y = sample_data()
    with pytest.raises(ValueError, match="search_type"):
        tune_model(
            LogisticRegression(),
            X,
            y,
            {"C": [1.0]},
            search_type="invalid",
            n_jobs=1,
        )


def test_tuning_rejects_too_small_class():
    X = pd.DataFrame({"x": [1, 2, 3, 4]})
    y = pd.Series([0, 0, 0, 1])
    with pytest.raises(ValueError, match="at least two samples"):
        tune_model(
            LogisticRegression(),
            X,
            y,
            {"C": [1.0]},
            cv=3,
            n_jobs=1,
        )
