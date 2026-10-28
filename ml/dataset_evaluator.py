from pathlib import Path
import pandas as pd
import numpy as np

def evaluate_dataset(df):
    if df is None or df.empty:
        raise ValueError("The dataset is empty.")
    if len(df.columns) == 0:
        raise ValueError("The dataset has no columns.")

    rows, cols = df.shape
    missing = df.isna().sum()
    duplicated = int(df.duplicated().sum())

    column_details = []
    numeric_columns = df.select_dtypes(include="number").columns.tolist()
    categorical_columns = [
        c for c in df.columns
        if c not in numeric_columns
    ]

    for column in df.columns:
        series = df[column]
        missing_count = int(series.isna().sum())
        unique_count = int(series.nunique(dropna=True))
        item = {
            "name": str(column),
            "dtype": str(series.dtype),
            "missing_count": missing_count,
            "missing_percent": round(missing_count / rows * 100, 2),
            "unique_count": unique_count,
            "unique_percent": round(unique_count / rows * 100, 2),
            "constant": unique_count <= 1,
            "sample_values": [
                str(x)[:120] for x in series.dropna().astype(str).unique()[:5]
            ],
        }

        if column in numeric_columns:
            values = pd.to_numeric(series, errors="coerce").dropna()
            item["statistics"] = {
                "mean": safe_number(values.mean()),
                "median": safe_number(values.median()),
                "std": safe_number(values.std()),
                "min": safe_number(values.min()),
                "max": safe_number(values.max()),
            }
            if len(values) >= 4:
                q1 = values.quantile(0.25)
                q3 = values.quantile(0.75)
                iqr = q3 - q1
                if iqr == 0:
                    outliers = 0
                else:
                    outliers = int(((values < q1 - 1.5 * iqr) |
                                    (values > q3 + 1.5 * iqr)).sum())
                item["iqr_outlier_count"] = outliers
                item["iqr_outlier_percent"] = round(outliers / len(values) * 100, 2)
            else:
                item["iqr_outlier_count"] = None
                item["iqr_outlier_percent"] = None

        else:
            counts = series.dropna().astype(str).value_counts().head(10)
            item["top_categories"] = [
                {"value": str(key)[:120], "count": int(value)}
                for key, value in counts.items()
            ]

        column_details.append(item)

    correlations = []
    if len(numeric_columns) >= 2:
        corr = df[numeric_columns].corr(numeric_only=True)
        for i, col_a in enumerate(corr.columns):
            for col_b in corr.columns[i + 1:]:
                value = corr.loc[col_a, col_b]
                if pd.notna(value):
                    correlations.append({
                        "column_a": str(col_a),
                        "column_b": str(col_b),
                        "correlation": round(float(value), 4),
                        "strength": (
                            "strong" if abs(value) >= 0.7
                            else "moderate" if abs(value) >= 0.4
                            else "weak"
                        ),
                    })
        correlations.sort(key=lambda x: abs(x["correlation"]), reverse=True)
        correlations = correlations[:30]

    possible_targets = []
    for column in df.columns:
        series = df[column]
        unique_count = int(series.nunique(dropna=True))
        if unique_count < 2:
            continue
        if unique_count <= 20 or column in numeric_columns:
            possible_targets.append({
                "column": str(column),
                "dtype": str(series.dtype),
                "unique_values": unique_count,
                "suggestion": (
                    "Possible classification target"
                    if column not in numeric_columns or unique_count <= 10
                    else "Possible regression target"
                ),
                "note": "Candidate only; confirm with the project requirements."
            })

    warnings = []
    missing_total = int(missing.sum())
    if missing_total:
        warnings.append(
            f"{missing_total} missing cells found "
            f"({missing_total / (rows * cols) * 100:.2f}% of all cells)."
        )
    if duplicated:
        warnings.append(f"{duplicated} duplicate rows found.")
    constants = [x["name"] for x in column_details if x["constant"]]
    if constants:
        warnings.append("Constant or nearly empty columns: " + ", ".join(constants))
    if rows < 100:
        warnings.append("Small dataset: model estimates may be unstable.")
    if rows < 20:
        warnings.append("Very small dataset: treat any model evaluation cautiously.")
    if not numeric_columns:
        warnings.append("No numeric columns detected; check whether numeric values were imported as text.")
    if not warnings:
        warnings.append("No basic structural warning was detected. This does not prove the dataset is suitable.")

    quality_score = max(0, round(
        100
        - (missing_total / (rows * cols) * 45)
        - (duplicated / rows * 20)
        - (len(constants) / cols * 20)
        - (15 if rows < 20 else 5 if rows < 100 else 0)
    ))

    return {
        "row_count": int(rows),
        "column_count": int(cols),
        "duplicate_rows": duplicated,
        "missing_cells": missing_total,
        "missing_percent": round(missing_total / (rows * cols) * 100, 2),
        "numeric_column_count": len(numeric_columns),
        "categorical_column_count": len(categorical_columns),
        "quality_indicator": quality_score,
        "quality_indicator_note": "Heuristic screening indicator, not a scientific suitability score.",
        "columns": column_details,
        "correlations": correlations,
        "possible_targets": possible_targets[:50],
        "warnings": warnings,
    }

def safe_number(value):
    if pd.isna(value):
        return None
    return float(value)
