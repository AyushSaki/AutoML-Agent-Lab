import pandas as pd

from ml.data_intelligence import analyze_dataset
from ml.autonomous_scientist import inspect_dataset


def test_data_intelligence_detects_missing_values_and_duplicates(tmp_path):
    path = tmp_path / "sample.csv"

    pd.DataFrame({
        "age": [20, 30, 30, None, 40],
        "department": ["IT", "HR", "HR", "IT", "Sales"],
        "salary": [30000, 40000, 40000, 35000, 50000],
    }).to_csv(path, index=False)

    result = analyze_dataset(path)

    assert result["rows"] == 5
    assert result["columns"] == 3
    assert result["total_missing"] == 1
    assert result["duplicate_rows"] == 1
    assert "age" in result["numeric_columns"]


def test_dataset_inspection_identifies_target_candidates(tmp_path):
    path = tmp_path / "loan_data.csv"

    pd.DataFrame({
        "applicant_id": range(1, 21),
        "income": [25000 + i * 1000 for i in range(20)],
        "Loan_Status": ["Approved", "Rejected"] * 10,
    }).to_csv(path, index=False)

    result = inspect_dataset(path)

    assert result["rows"] == 20
    assert result["columns"] == 3
    assert result["target_candidates"]
    assert any(
        item["column"] == "Loan_Status"
        for item in result["target_candidates"]
    )


def test_data_intelligence_handles_excel(tmp_path):
    path = tmp_path / "sample.xlsx"

    pd.DataFrame({
        "sales": [100, 150, 200, 250],
        "region": ["North", "South", "North", "West"],
    }).to_excel(path, index=False)

    result = analyze_dataset(path)

    assert result["rows"] == 4
    assert result["columns"] == 2
    assert "sales" in result["numeric_columns"]


def test_task_detection_matches_inspection_module():
    from ml.pipeline import _infer_task
    from ml.autonomous_scientist import _task_for_column

    classification_target = pd.Series(list(range(10)) * 10)
    regression_target = pd.Series(list(range(20)) * 5)

    assert _infer_task(classification_target) == _task_for_column(
        classification_target, len(classification_target)
    )
    assert _infer_task(regression_target) == _task_for_column(
        regression_target, len(regression_target)
    )


def test_task_detection_handles_text_target():
    from ml.pipeline import _infer_task

    target = pd.Series(["Approved", "Rejected"] * 10)

    assert _infer_task(target) == "classification"
