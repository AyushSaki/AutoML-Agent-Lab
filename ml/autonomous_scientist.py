from pathlib import Path
import json
import re

import pandas as pd


def load_dataset(path):
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".csv":
        return pd.read_csv(path, low_memory=False)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)

    raise ValueError("Only CSV, XLSX and XLS datasets are supported.")


def _normalize_name(value):
    value = str(value).strip().lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_")


def _task_for_column(series, rows):
    observed = series.dropna()
    unique_count = int(observed.nunique())

    if unique_count < 2 or observed.empty:
        return "unsuitable"

    if (
        not pd.api.types.is_numeric_dtype(series)
        or pd.api.types.is_bool_dtype(series)
    ):
        return "classification"

    threshold = min(12, max(2, int(rows * 0.1)))
    if unique_count <= threshold:
        return "classification"

    return "regression"


def _target_candidates(df):
    rows = len(df)
    candidates = []

    # These names are strong clues, not proof that a column is the target.
    target_names = {
        "target": 100,
        "label": 95,
        "class": 90,
        "outcome": 90,
        "y": 85,
        "medv": 100,
        "price": 100,
        "house_price": 110,
        "houseprice": 110,
        "sale_price": 110,
        "selling_price": 105,
        "loan_status": 110,
        "loanstatus": 110,
        "loan_approved": 105,
        "approved": 90,
        "churn": 100,
        "churned": 100,
        "default": 95,
        "defaulted": 95,
        "survived": 100,
        "diagnosis": 90,
        "species": 90,
    }

    for column in df.columns:
        series = df[column]
        observed = series.dropna()
        unique_count = int(observed.nunique())

        if len(observed) == 0 or unique_count < 2:
            continue

        normalized = _normalize_name(column)
        task = _task_for_column(series, rows)

        if task == "unsuitable":
            continue

        score = 10
        reason = (
            "Categorical target candidate"
            if task == "classification"
            else "Numeric target candidate"
        )

        if normalized in target_names:
            score += target_names[normalized]
            reason = "Recognized target-like column name"
        elif any(
            token in normalized.split("_")
            for token in ("target", "label", "outcome", "status", "price")
        ):
            score += 45
            reason = "Column name suggests a possible outcome"

        if normalized in {
            "id", "index", "row_id", "customer_id", "customerid",
            "loan_id", "employee_id", "record_id"
        } or normalized.endswith("_id"):
            score -= 100
            reason = "Identifier-like column; usually not a prediction target"

        # Small-cardinality targets are often useful for classification.
        if task == "classification":
            score += 5

        candidates.append({
            "column": str(column),
            "suggested_task": task,
            "reason": reason,
            "score": score,
        })

    candidates.sort(key=lambda item: item["score"], reverse=True)

    # Do not expose the internal ranking score in the dashboard/API.
    for candidate in candidates:
        candidate.pop("score", None)

    return candidates


def inspect_dataset(path, target=None):
    df = load_dataset(path)
    rows, columns = df.shape

    if rows == 0 or columns == 0:
        raise ValueError("The dataset is empty.")

    findings = []
    recommendations = []
    column_details = []

    missing_total = int(df.isna().sum().sum())
    duplicates = int(df.duplicated().sum())

    if missing_total:
        findings.append(f"{missing_total} missing cell values were found.")
        recommendations.append(
            "Review missingness by column. Impute or remove missing values "
            "only after considering their meaning."
        )
    else:
        findings.append("No missing cell values were detected.")

    if duplicates:
        findings.append(f"{duplicates} duplicate rows were found.")
        recommendations.append(
            "Review duplicate rows before deciding whether to remove them."
        )

    constant_columns = []
    high_missing_columns = []
    identifier_like_columns = []

    for column in df.columns:
        series = df[column]
        unique_count = int(series.nunique(dropna=True))
        missing_count = int(series.isna().sum())
        missing_pct = round(missing_count / max(rows, 1) * 100, 2)

        column_details.append({
            "column": str(column),
            "dtype": str(series.dtype),
            "missing": missing_count,
            "missing_percent": missing_pct,
            "unique_values": unique_count,
        })

        if unique_count <= 1:
            constant_columns.append(str(column))

        if missing_pct >= 40:
            high_missing_columns.append(str(column))

        normalized = _normalize_name(column)
        id_name = (
            normalized in {"id", "index", "row_id", "customer_id"}
            or normalized.endswith("_id")
            or normalized.startswith("id_")
        )

        if rows >= 5 and unique_count / rows >= 0.9:
            if id_name or "id" in normalized:
                identifier_like_columns.append(str(column))

    if constant_columns:
        findings.append(
            "Constant or effectively constant columns: "
            + ", ".join(constant_columns)
        )
        recommendations.append(
            "Consider removing constant columns from model features."
        )

    if high_missing_columns:
        findings.append(
            "Columns with at least 40% missing values: "
            + ", ".join(high_missing_columns)
        )
        recommendations.append(
            "Investigate high-missingness columns before imputing or dropping them."
        )

    if identifier_like_columns:
        findings.append(
            "Potential identifier columns: "
            + ", ".join(identifier_like_columns)
        )
        recommendations.append(
            "Check whether identifier-like columns should be excluded from "
            "predictive features to avoid misleading patterns."
        )

    candidates = _target_candidates(df)
    chosen_target = target if target in df.columns else None
    auto_selected = False

    # Only auto-select a target when the column name provides a strong clue.
    if chosen_target is None and candidates:
        top = candidates[0]
        normalized_top = _normalize_name(top["column"])
        if top["reason"] == "Recognized target-like column name":
            chosen_target = top["column"]
            auto_selected = True

    task = "Unsupervised exploration"
    target_info = None

    if chosen_target is not None:
        target_series = df[chosen_target].dropna()
        target_unique = int(target_series.nunique())
        target_task = _task_for_column(df[chosen_target], rows)

        if target_task == "unsuitable" or target_unique < 2:
            task = "Target unsuitable"
            recommendations.append(
                f"Target '{chosen_target}' has fewer than two observed values. "
                "Choose another target."
            )
        elif target_task == "classification":
            task = "Classification"
        else:
            task = "Regression"

        target_info = {
            "column": str(chosen_target),
            "dtype": str(df[chosen_target].dtype),
            "unique_values": target_unique,
            "missing_values": int(df[chosen_target].isna().sum()),
            "auto_selected": auto_selected,
        }

        if target_task == "classification" and target_unique >= 2:
            counts = target_series.value_counts()
            proportions = target_series.value_counts(normalize=True)

            imbalance_ratio = (
                float(counts.max() / counts.min())
                if len(counts) > 1 and counts.min() > 0
                else None
            )

            target_info["class_counts"] = {
                str(key): int(value)
                for key, value in counts.head(30).items()
            }
            target_info["imbalance_ratio"] = (
                round(imbalance_ratio, 2)
                if imbalance_ratio is not None
                else None
            )

            if len(proportions) > 1 and proportions.min() < 0.1:
                findings.append(
                    f"Target '{chosen_target}' appears imbalanced: "
                    "at least one class represents less than 10% of rows."
                )
                recommendations.append(
                    "Use stratified splitting where possible and inspect "
                    "precision, recall, F1 and the confusion matrix."
                )

        if target_info["missing_values"]:
            recommendations.append(
                "Rows with missing target values need explicit handling "
                "before supervised training."
            )

        if auto_selected:
            findings.append(
                f"Automatically suggested '{chosen_target}' as the likely "
                f"{task.lower()} target based on its column name. Confirm it "
                "matches your prediction goal."
            )
    else:
        recommendations.append(
            "Choose a target column for supervised learning, or use "
            "clustering/anomaly detection when there is no known target."
        )

    if rows < 50:
        findings.append(
            f"The dataset is small ({rows} rows); evaluation results may be unstable."
        )
        recommendations.append(
            "Treat scores cautiously. A larger representative dataset is "
            "recommended before drawing strong conclusions."
        )

    if not recommendations:
        recommendations.append(
            "Continue with exploratory analysis and validate the dataset's "
            "meaning before model training."
        )

    numeric_summary = {}
    numeric_df = df.select_dtypes(include="number")

    if not numeric_df.empty:
        numeric_summary = json.loads(
            numeric_df.describe().round(3).to_json()
        )

    return {
        "rows": int(rows),
        "columns": int(columns),
        "missing_values": missing_total,
        "duplicate_rows": duplicates,
        "constant_columns": constant_columns,
        "high_missing_columns": high_missing_columns,
        "identifier_like_columns": identifier_like_columns,
        "target": target_info,
        "recommended_task": task,
        "target_candidates": candidates[:15],
        "findings": findings,
        "recommendations": recommendations,
        "column_details": column_details,
        "numeric_summary": numeric_summary,
    }
