from pathlib import Path
from datetime import datetime
import shutil
import sys

root = Path.cwd()
app_file = root / "app.py"
ml_dir = root / "ml"
templates_dir = root / "templates"

if not app_file.exists():
    sys.exit("ERROR: app.py not found. Run from AutoML-Agent-Lab.")

source = app_file.read_text()
if "/dataset-compare" in source:
    sys.exit("Dataset comparison route already exists. No changes made.")

startup = "if __name__ == '__main__':"
if startup not in source:
    startup = 'if __name__ == "__main__":'
if startup not in source:
    sys.exit("Cannot safely locate the app startup block. No changes made.")

backup = root / "backups" / ("dataset_compare_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
backup.mkdir(parents=True, exist_ok=True)
shutil.copy2(app_file, backup / "app.py")

base_file = templates_dir / "base.html"
if base_file.exists():
    shutil.copy2(base_file, backup / "base.html")

ml_dir.mkdir(exist_ok=True)
templates_dir.mkdir(exist_ok=True)
(root / "reports" / "dataset_comparisons").mkdir(parents=True, exist_ok=True)

module = r'''
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
'''

(ml_dir / "dataset_comparator.py").write_text(module.strip() + "\n")

template = r'''{% extends "base.html" %}
{% block title %}Dataset Comparison | AutoML-Agent-Lab{% endblock %}
{% block content %}
<div class="container py-4">
  <h1>Dataset &amp; Model Comparison Lab</h1>
  <p>Compare candidate datasets for the same target and task. Structural quality and held-out model performance are reported separately.</p>

  <form method="post" enctype="multipart/form-data" class="card p-4 mb-4">
    <div class="mb-3">
      <label class="form-label" for="datasets">Candidate datasets (choose 2–5 files)</label>
      <input class="form-control" type="file" id="datasets" name="datasets"
             accept=".csv,.xlsx,.xls" multiple required>
    </div>
    <div class="mb-3">
      <label class="form-label" for="target">Target column name (must exist in each dataset)</label>
      <input class="form-control" id="target" name="target" required
             value="{{ form.get('target', '') }}" placeholder="Example: SalePrice">
    </div>
    <div class="mb-3">
      <label class="form-label" for="task">Machine learning task</label>
      <select class="form-select" id="task" name="task">
        <option value="regression" {% if form.get('task', 'regression') == 'regression' %}selected{% endif %}>Regression — predict a numeric value</option>
        <option value="classification" {% if form.get('task') == 'classification' %}selected{% endif %}>Classification — predict a category</option>
      </select>
    </div>
    <button class="btn btn-primary" type="submit">Compare Datasets and Models</button>
    <small class="text-muted mt-2">Each file is evaluated separately with the same split seed and baseline algorithms. Files are processed in memory and not saved as uploaded datasets by this feature.</small>
  </form>

  {% if error %}
  <div class="alert alert-danger">{{ error }}</div>
  {% endif %}

  {% if comparison %}
  <div class="card p-4 mb-4">
    <h2>Comparison Summary</h2>
    <p><strong>Target:</strong> {{ comparison.target }} | <strong>Task:</strong> {{ comparison.task }}</p>
    <p class="text-muted">Do not treat the highest quality indicator as the automatic winner. Review model metrics and each dataset's suitability for the client requirements.</p>
    <div class="table-responsive">
      <table class="table table-striped">
        <thead><tr>
          <th>Dataset</th><th>Rows used</th><th>Features</th><th>Quality indicator</th><th>Best tested model</th><th>Best metrics</th>
        </tr></thead>
        <tbody>
        {% for item in comparison.datasets %}
        <tr>
          <td>{{ item.filename }}</td>
          <td>{{ item.rows_used }}</td>
          <td>{{ item.features_used }}</td>
          <td>{{ item.quality_indicator }}/100</td>
          <td>{{ item.best_model }}</td>
          <td>{% for key, value in item.best_metrics.items() %}
            <div><small>{{ key }}: {{ value if value is not none else 'N/A' }}</small></div>
          {% endfor %}</td>
        </tr>
        {% endfor %}
        </tbody>
      </table>
    </div>

    <h3 class="mt-4">Dataset-specific observations</h3>
    {% for item in comparison.datasets %}
    <div class="border rounded p-3 mb-3">
      <h4>{{ item.filename }}</h4>
      <p><strong>Evaluation:</strong> {{ item.evaluation_method }}</p>
      <p><strong>Ranking rule:</strong> {{ item.ranking_rule }}</p>
      <h5>Model results</h5>
      <div class="table-responsive">
        <table class="table table-sm">
          <thead><tr><th>Model</th>{% for metric in item.models[0].metrics.keys() %}<th>{{ metric }}</th>{% endfor %}</tr></thead>
          <tbody>{% for model in item.models %}
            <tr><td>{{ model.model }}</td>{% for value in model.metrics.values() %}<td>{{ value if value is not none else 'N/A' }}</td>{% endfor %}</tr>
          {% endfor %}</tbody>
        </table>
      </div>
      <h5>Data quality notes</h5>
      <ul>{% for note in item.quality_notes %}<li>{{ note }}</li>{% endfor %}</ul>
    </div>
    {% endfor %}

    <h3 class="mt-4">Interpretation limits</h3>
    <ul>{% for note in comparison.limitations %}<li>{{ note }}</li>{% endfor %}</ul>
    <a class="btn btn-outline-primary" href="{{ url_for('dataset_comparison_json', report_id=report_id) }}">Download comparison JSON</a>
  </div>
  {% endif %}
</div>
{% endblock %}
'''

(templates_dir / "dataset_compare.html").write_text(template.strip() + "\n")

routes = r'''

@app.route("/dataset-compare", methods=["GET", "POST"])
def dataset_compare():
    import json
    import uuid
    from flask import request, render_template
    from ml.dataset_comparator import compare_one_dataset

    form = {"target": "", "task": "regression"}
    comparison = None
    report_id = None
    error = None

    if request.method == "POST":
        form = {
            "target": request.form.get("target", "").strip(),
            "task": request.form.get("task", "regression").strip(),
        }
        files = [f for f in request.files.getlist("datasets") if f and f.filename]

        if len(files) < 2 or len(files) > 5:
            error = "Choose between 2 and 5 datasets."
        elif not form["target"]:
            error = "Enter the target column name."
        elif form["task"] not in ("regression", "classification"):
            error = "Unsupported task."
        else:
            results = []
            failures = []
            total_bytes = 0

            for upload in files:
                filename = Path(upload.filename).name
                suffix = Path(filename).suffix.lower()
                if suffix not in (".csv", ".xlsx", ".xls"):
                    failures.append(f"{filename}: unsupported file type.")
                    continue

                content = upload.read(50 * 1024 * 1024 + 1)
                total_bytes += len(content)
                if len(content) > 50 * 1024 * 1024:
                    failures.append(f"{filename}: file exceeds 50 MB.")
                    continue
                if total_bytes > 100 * 1024 * 1024:
                    failures.append("Combined upload exceeds 100 MB.")
                    break

                try:
                    result = compare_one_dataset(
                        content, filename, form["target"], form["task"]
                    )
                    results.append(result)
                except Exception as exc:
                    failures.append(f"{filename}: {exc}")

            if failures:
                error = "Some datasets could not be compared: " + " | ".join(failures)
            if len(results) >= 2:
                report_id = uuid.uuid4().hex
                comparison = {
                    "report_id": report_id,
                    "target": form["target"],
                    "task": form["task"],
                    "datasets": results,
                    "limitations": [
                        "Only datasets that completed evaluation appear in the comparison.",
                        "Dataset quality indicator is heuristic and does not establish relevance to the client problem.",
                        "Model scores from different datasets may not be directly comparable if rows, collection periods, populations, or feature definitions differ.",
                        "Use cross-validation and a client-approved success criterion before selecting a production model.",
                    ],
                }
                report_dir = Path(__file__).resolve().parent / "reports" / "dataset_comparisons"
                report_dir.mkdir(parents=True, exist_ok=True)
                (report_dir / f"{report_id}.json").write_text(
                    json.dumps(comparison, indent=2, ensure_ascii=False, allow_nan=False)
                )
                if failures:
                    error += " Successful datasets are shown below; fix the failed files for a complete comparison."
            elif not error:
                error = "At least two datasets must evaluate successfully."

    return render_template(
        "dataset_compare.html",
        form=form,
        comparison=comparison,
        report_id=report_id,
        error=error,
    )


@app.route("/dataset-compare/report/<report_id>.json")
def dataset_comparison_json(report_id):
    import re
    from flask import abort, send_file

    if not re.fullmatch(r"[a-f0-9]{32}", report_id):
        abort(404)
    path = Path(__file__).resolve().parent / "reports" / "dataset_comparisons" / f"{report_id}.json"
    if not path.is_file():
        abort(404)
    return send_file(path, as_attachment=True, download_name=f"dataset_comparison_{report_id}.json")
'''

source = source.replace(startup, routes + "\n\n" + startup, 1)
app_file.write_text(source)

if base_file.exists():
    html = base_file.read_text()
    if "dataset-compare" not in html:
        link = '<a class="nav-link" href="{{ url_for(\'dataset_compare\') }}">Compare Datasets</a>\n'
        if "</nav>" in html:
            html = html.replace("</nav>", link + "</nav>", 1)
        else:
            print("NOTE: Navbar not detected; open /dataset-compare directly.")
        base_file.write_text(html)

print("Dataset & Model Comparison Lab installed.")
print("Backup:", backup)
print("Route: /dataset-compare")
