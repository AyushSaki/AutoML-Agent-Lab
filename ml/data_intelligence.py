
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px


def load_dataset(path):
    """Load supported CSV and Excel datasets."""
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".csv":
        try:
            return pd.read_csv(path)
        except UnicodeDecodeError:
            return pd.read_csv(path, encoding="latin-1")

    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)

    raise ValueError("Unsupported file format. Upload CSV or Excel.")


def _clean_number(value):
    """Convert NumPy/Pandas numeric values to JSON-safe Python values."""
    if pd.isna(value):
        return None
    return float(value)


def _chart_json(figure):
    """Return a JSON-safe Plotly figure for the browser."""
    return json.loads(figure.to_json())


def analyze_dataset(path):
    """Generate a detailed, JSON-safe dataset profile and interactive charts."""
    df = load_dataset(path)

    if df.empty:
        raise ValueError("The uploaded dataset contains no rows.")

    if len(df.columns) == 0:
        raise ValueError("The uploaded dataset contains no columns.")

    numeric_columns = df.select_dtypes(include="number").columns.tolist()
    categorical_columns = [
        col for col in df.columns if col not in numeric_columns
    ]

    rows, columns = df.shape
    missing_by_column = []
    statistics = []
    outlier_summary = []
    categorical_distributions = []

    # Missing values and data types
    for column in df.columns:
        missing_count = int(df[column].isna().sum())

        missing_by_column.append({
            "column": str(column),
            "missing": missing_count,
            "missing_percent": round(missing_count / rows * 100, 2),
            "dtype": str(df[column].dtype),
            "unique_values": int(df[column].nunique(dropna=True)),
        })

    missing_by_column.sort(
        key=lambda item: item["missing"], reverse=True
    )

    # Numeric statistics and IQR-based outlier detection
    for column in numeric_columns:
        series = pd.to_numeric(df[column], errors="coerce").dropna()

        if series.empty:
            continue

        q1 = float(series.quantile(0.25))
        median = float(series.median())
        q3 = float(series.quantile(0.75))
        iqr = q3 - q1

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        outlier_mask = (series < lower_bound) | (series > upper_bound)

        statistics.append({
            "column": str(column),
            "count": int(series.count()),
            "mean": _clean_number(series.mean()),
            "std": _clean_number(series.std()),
            "min": _clean_number(series.min()),
            "q1": q1,
            "median": median,
            "q3": q3,
            "max": _clean_number(series.max()),
        })

        outlier_summary.append({
            "column": str(column),
            "outlier_count": int(outlier_mask.sum()),
            "outlier_percent": round(
                float(outlier_mask.mean() * 100), 2
            ),
            "lower_bound": round(lower_bound, 4),
            "upper_bound": round(upper_bound, 4),
        })

    outlier_summary.sort(
        key=lambda item: item["outlier_count"], reverse=True
    )

    # Most frequent values in categorical columns
    for column in categorical_columns:
        counts = (
            df[column]
            .astype("string")
            .fillna("(Missing)")
            .value_counts()
            .head(15)
        )

        categorical_distributions.append({
            "column": str(column),
            "categories": [
                {"value": str(value), "count": int(count)}
                for value, count in counts.items()
            ],
        })

    # Interactive Plotly charts
    charts = {}

    if numeric_columns:
        histogram_column = numeric_columns[0]
        histogram = px.histogram(
            df,
            x=histogram_column,
            title=f"Distribution: {histogram_column}",
            marginal="box",
            template="plotly_white",
        )
        histogram.update_layout(
            autosize=True,
            margin=dict(l=35, r=20, t=55, b=45),
        )
        charts["histogram"] = _chart_json(histogram)

        box_data = df[numeric_columns].melt(
            var_name="Column", value_name="Value"
        )
        box_data = box_data.dropna(subset=["Value"])

        boxplot = px.box(
            box_data,
            x="Column",
            y="Value",
            title="Numeric Columns and Potential Outliers",
            points="outliers",
            template="plotly_white",
        )
        boxplot.update_layout(
            autosize=True,
            margin=dict(l=35, r=20, t=55, b=90),
        )
        charts["boxplot"] = _chart_json(boxplot)

    if categorical_columns:
        category_column = categorical_columns[0]
        category_counts = (
            df[category_column]
            .astype("string")
            .fillna("(Missing)")
            .value_counts()
            .head(15)
            .rename_axis("Category")
            .reset_index(name="Count")
        )

        category_chart = px.bar(
            category_counts,
            x="Category",
            y="Count",
            title=f"Top Categories: {category_column}",
            template="plotly_white",
        )
        category_chart.update_layout(
            autosize=True,
            margin=dict(l=35, r=20, t=55, b=100),
        )
        charts["categories"] = _chart_json(category_chart)

    if len(numeric_columns) >= 2:
        correlation = df[numeric_columns].corr()

        heatmap = px.imshow(
            correlation,
            text_auto=".2f",
            aspect="auto",
            color_continuous_scale="RdBu_r",
            zmin=-1,
            zmax=1,
            title="Numeric Feature Correlation",
            template="plotly_white",
        )
        heatmap.update_layout(
            autosize=True,
            margin=dict(l=35, r=20, t=55, b=70),
        )
        charts["correlation"] = _chart_json(heatmap)

    total_missing = int(df.isna().sum().sum())
    total_cells = rows * columns

    return {
        "file_path": str(path),
        "rows": int(rows),
        "columns": int(columns),
        "column_names": [str(col) for col in df.columns],
        "numeric_columns": [str(col) for col in numeric_columns],
        "categorical_columns": [str(col) for col in categorical_columns],
        "total_missing": total_missing,
        "missing_values": total_missing,
        "missing_percent": round(total_missing / total_cells * 100, 2),
        "duplicate_rows": int(df.duplicated().sum()),
        "dtypes": {str(col): str(df[col].dtype) for col in df.columns},
        "preview": json.loads(
            df.head(8).to_json(orient="records", date_format="iso")
        ),
        "missing_by_column": missing_by_column,
        "statistics": statistics,
        "outlier_summary": outlier_summary,
        "categorical_distributions": categorical_distributions,
        "charts": charts,
    }


