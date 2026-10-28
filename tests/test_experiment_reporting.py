import json

import pytest

from ml.experiment_store import ExperimentStore
from ml.reporting import generate_markdown_report


def test_experiment_store_round_trip(tmp_path):
    db = tmp_path / "experiments.db"
    store = ExperimentStore(db)

    experiment_id = store.log_experiment(
        name="Sales baseline",
        task="regression",
        algorithm="RandomForestRegressor",
        dataset_name="sales.csv",
        rows_count=100,
        features_count=5,
        metrics={"mae": 2.5, "r2": 0.91},
        parameters={"n_estimators": 100},
        notes="Initial baseline",
    )

    saved = store.get_experiment(experiment_id)

    assert saved is not None
    assert saved["name"] == "Sales baseline"
    assert saved["metrics"]["mae"] == 2.5
    assert saved["parameters"]["n_estimators"] == 100
    assert store.list_experiments()[0]["id"] == experiment_id


def test_experiment_store_filters_by_task(tmp_path):
    store = ExperimentStore(tmp_path / "experiments.db")
    store.log_experiment("Classification", "classification", {"f1": 0.9})
    store.log_experiment("Regression", "regression", {"mae": 1.2})

    rows = store.list_experiments(task="classification")
    assert len(rows) == 1
    assert rows[0]["name"] == "Classification"


def test_report_generation_and_file_output(tmp_path):
    store = ExperimentStore(tmp_path / "experiments.db")
    experiment_id = store.log_experiment(
        "Test report",
        "classification",
        {"accuracy": 0.95, "f1": 0.93},
        algorithm="LogisticRegression",
        parameters={"C": 1.0},
    )
    experiment = store.get_experiment(experiment_id)
    output = tmp_path / "reports" / "test-report.md"

    report = generate_markdown_report(experiment, output)

    assert "# AutoML Experiment Report: Test report" in report
    assert "| accuracy | 0.95 |" in report
    assert output.exists()
    assert output.read_text(encoding="utf-8") == report


def test_store_rejects_empty_name(tmp_path):
    store = ExperimentStore(tmp_path / "experiments.db")
    with pytest.raises(ValueError, match="name"):
        store.log_experiment("", "classification", {})


def test_report_rejects_invalid_metrics():
    with pytest.raises(ValueError, match="metrics"):
        generate_markdown_report({"metrics": [], "parameters": {}})
