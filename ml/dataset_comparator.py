import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    precision_score, recall_score, mean_absolute_error,
    mean_squared_error, r2_score
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def _number(value):
    try:
        value = float(value)
        return value if np.isfinite(value) else None
    except (TypeError, ValueError):
        return None


def _read_dataset(file_bytes, filename):
    from io import BytesIO

    suffix = filename.lower().rsplit(".", 1)[-1]
    stream = BytesIO(file_bytes)
    if suffix == "csv":
        try:
            return pd.read_csv(stream, low_memory=False)
        except UnicodeDecodeError:
            stream.seek(0)
            return pd.read_csv(stream, encoding="latin-1", low_memory=False)
    if suffix in ("xlsx", "xls"):
        return pd.read_excel(stream)
    raise ValueError("Only CSV, XLSX, and XLS files are supported.")


def _make_preprocessor(X):
    numeric = X.select_dtypes(include=["number"]).columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]
    transformers = []

    if numeric:
        numeric_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ])
        transformers.append(("numeric", numeric_pipe, numeric))

    if categorical:
        try:
            encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
        except TypeError:
            encoder = OneHotEncoder(handle_unknown="ignore", sparse=True)

        categorical_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encode", encoder),
        ])
        transformers.append(("categorical", categorical_pipe, categorical))

    if not transformers:
        raise ValueError("No usable feature columns remain after removing the target.")

    return ColumnTransformer(transformers=transformers, remainder="drop")


def compare_one_dataset(file_bytes, filename, target, task):
    df = _read_dataset(file_bytes, filename)

    if df.empty:
        raise ValueError("Dataset contains no data rows.")
    if len(df) > 200000 or len(df.columns) > 300:
        raise ValueError("Comparison limit is 200,000 rows and 300 columns per dataset.")

    df.columns = [str(c).strip() for c in df.columns]
    if len(set(df.columns)) != len(df.columns):
        raise ValueError("Column names are duplicated after trimming whitespace.")

    if target not in df.columns:
        raise ValueError(f"Target column '{target}' was not found.")
    if len(df) < 20:
        raise ValueError("At least 20 rows are required for this initial comparison.")

    original_rows = len(df)
    missing_target = int(df[target].isna().sum())
    df = df.dropna(subset=[target]).copy()

    if len(df) < 20:
        raise ValueError("Fewer than 20 rows remain after removing missing target values.")

    y = df[target]
    X = df.drop(columns=[target])

    constant_cols = [c for c in X.columns if X[c].nunique(dropna=True) <= 1]
    X = X.drop(columns=constant_cols)

    # Remove columns with no usable observations.
    empty_cols = [c for c in X.columns if X[c].isna().all()]
    X = X.drop(columns=empty_cols)

    # Exclude obvious unique identifiers from the first baseline.
    identifier_cols = [
        c for c in X.columns
        if len(X) > 0 and X[c].nunique(dropna=True) == len(X)
        and any(token in c.lower() for token in ("id", "uuid", "record_number"))
    ]
    X = X.drop(columns=identifier_cols)

    if X.empty:
        raise ValueError("No usable feature columns remain after excluding constants and identifiers.")

    # Very high-cardinality text can create enormous one-hot encoded matrices.
    high_cardinality = [
        c for c in X.select_dtypes(exclude="number").columns
        if X[c].nunique(dropna=True) > min(500, max(100, int(len(X) * 0.8)))
    ]
    X = X.drop(columns=high_cardinality)
    if X.empty:
        raise ValueError("All remaining features have excessive categorical cardinality.")

    if task == "classification":
        y = y.astype(str)
        class_counts = y.value_counts()
        if len(class_counts) < 2:
            raise ValueError("Classification needs at least two target classes.")
        if len(class_counts) > 100:
            raise ValueError("More than 100 target classes detected. Confirm that this is classification, not regression.")
        stratify = y if class_counts.min() >= 2 else None
        if stratify is None:
            raise ValueError("Each target class needs at least two examples for a reliable split.")
        if class_counts.min() < 4:
            raise ValueError("Each target class needs at least four examples for this initial comparison.")
    else:
        y = pd.to_numeric(y, errors="coerce")
        valid = y.notna()
        X, y = X.loc[valid], y.loc[valid]
        if len(y) < 20:
            raise ValueError("Fewer than 20 numeric target values remain for regression.")
        stratify = None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=stratify
    )

    preprocessor = _make_preprocessor(X_train)

    if task == "classification":
        candidates = [
            ("Dummy baseline", DummyClassifier(strategy="most_frequent")),
            ("Random Forest", RandomForestClassifier(
                n_estimators=150, random_state=42, class_weight="balanced",
                n_jobs=-1, max_features="sqrt"
            )),
        ]
    else:
        candidates = [
            ("Dummy baseline", DummyRegressor(strategy="mean")),
            ("Random Forest", RandomForestRegressor(
                n_estimators=150, random_state=42, n_jobs=-1,
                max_features=1.0
            )),
        ]

    results = []
    for name, estimator in candidates:
        model = Pipeline([
            ("preprocessing", _make_preprocessor(X_train)),
            ("model", estimator),
        ])
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        if task == "classification":
            metrics = {
                "accuracy": _number(accuracy_score(y_test, predictions)),
                "balanced_accuracy": _number(balanced_accuracy_score(y_test, predictions)),
                "precision_weighted": _number(precision_score(
                    y_test, predictions, average="weighted", zero_division=0
                )),
                "recall_weighted": _number(recall_score(
                    y_test, predictions, average="weighted", zero_division=0
                )),
                "f1_weighted": _number(f1_score(
                    y_test, predictions, average="weighted", zero_division=0
                )),
            }
        else:
            mse = mean_squared_error(y_test, predictions)
            metrics = {
                "mae": _number(mean_absolute_error(y_test, predictions)),
                "rmse": _number(np.sqrt(mse)),
                "r2": _number(r2_score(y_test, predictions)),
            }

        results.append({"model": name, "metrics": metrics})

    # Ranking is task-specific and deliberately prioritizes generalization metrics.
    if task == "classification":
        ranked = sorted(
            results,
            key=lambda r: (
                r["metrics"]["f1_weighted"] if r["metrics"]["f1_weighted"] is not None else -1,
                r["metrics"]["balanced_accuracy"] if r["metrics"]["balanced_accuracy"] is not None else -1,
            ),
            reverse=True,
        )
    else:
        ranked = sorted(
            results,
            key=lambda r: (
                r["metrics"]["mae"] if r["metrics"]["mae"] is not None else float("inf")
            ),
        )

    missing_cells = int(df.isna().sum().sum())
    duplicate_rows = int(df.duplicated().sum())
    total_cells = int(df.shape[0] * df.shape[1])
    quality_notes = []

    if missing_cells:
        quality_notes.append(f"{missing_cells} missing feature cells after target filtering.")
    if duplicate_rows:
        quality_notes.append(f"{duplicate_rows} duplicate rows detected.")
    if missing_target:
        quality_notes.append(f"{missing_target} rows excluded due to missing target values.")
    if high_cardinality:
        quality_notes.append("High-cardinality features excluded: " + ", ".join(high_cardinality[:10]))
    if identifier_cols:
        quality_notes.append("Likely identifier features excluded: " + ", ".join(identifier_cols[:10]))
    if constant_cols:
        quality_notes.append("Constant features excluded: " + ", ".join(constant_cols[:10]))
    if not quality_notes:
        quality_notes.append("No major issue detected by these basic checks; domain suitability is not guaranteed.")

    # This is an explicitly heuristic indicator, not a model score.
    quality_indicator = max(0, round(
        100
        - (missing_cells / max(1, total_cells) * 45)
        - (duplicate_rows / max(1, len(df)) * 20)
        - (missing_target / max(1, original_rows) * 20)
        - (15 if len(df) < 100 else 0)
    ))

    return {
        "filename": filename,
        "task": task,
        "rows_original": original_rows,
        "rows_used": int(len(y)),
        "features_used": int(X.shape[1]),
        "target": target,
        "target_unique_values": int(y.nunique()),
        "missing_feature_cells": missing_cells,
        "duplicate_rows": duplicate_rows,
        "missing_target_rows_excluded": missing_target,
        "quality_indicator": quality_indicator,
        "quality_indicator_note": "Heuristic structural indicator only; not a scientific dataset suitability score.",
        "quality_notes": quality_notes,
        "models": results,
        "best_model": ranked[0]["model"],
        "best_metrics": ranked[0]["metrics"],
        "ranking_rule": (
            "Highest weighted F1, with balanced accuracy as tie-breaker."
            if task == "classification"
            else "Lowest held-out mean absolute error (MAE)."
        ),
        "evaluation_method": "One 75/25 train/test split, random_state=42; preprocessing is fitted on training data only.",
        "limitations": [
            "Results depend on the selected target, task, and random split.",
            "A single holdout split is not a substitute for cross-validation.",
            "Dataset suitability depends on the client's actual problem and data provenance.",
            "The best model is only the best among the two tested candidates.",
        ],
    }
