from pathlib import Path
from datetime import datetime
import shutil
import sys

root = Path.cwd()
app_file = root / "app.py"
ml_dir = root / "ml"
templates_dir = root / "templates"

if not app_file.exists():
    sys.exit("ERROR: app.py not found. Run this from AutoML-Agent-Lab.")

source = app_file.read_text()

if "/dataset-lab" in source:
    sys.exit("Dataset Lab route already exists. No changes made.")

startup = "if __name__ == '__main__':"
if startup not in source:
    startup = 'if __name__ == "__main__":'
if startup not in source:
    sys.exit("Could not locate app startup block. No changes made.")

backup = root / "backups" / ("dataset_lab_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
backup.mkdir(parents=True, exist_ok=True)
shutil.copy2(app_file, backup / "app.py")

base_file = templates_dir / "base.html"
if base_file.exists():
    shutil.copy2(base_file, backup / "base.html")

ml_dir.mkdir(exist_ok=True)
templates_dir.mkdir(exist_ok=True)
(root / "uploads").mkdir(exist_ok=True)
(root / "reports" / "dataset_profiles").mkdir(parents=True, exist_ok=True)

module = r'''
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
'''

(ml_dir / "dataset_evaluator.py").write_text(module.strip() + "\n")

template = r'''{% extends "base.html" %}
{% block title %}Dataset Intelligence Lab | AutoML-Agent-Lab{% endblock %}
{% block content %}
<div class="container py-4">
  <h1>Dataset Intelligence Lab</h1>
  <p>Profile CSV and Excel files before training. Your uploaded file is processed locally by this app.</p>

  <form method="post" enctype="multipart/form-data" class="card p-4 mb-4">
    <label class="form-label" for="dataset">Choose a CSV or Excel dataset</label>
    <input class="form-control mb-3" type="file" id="dataset" name="dataset"
           accept=".csv,.xlsx,.xls" required>
    <button class="btn btn-primary" type="submit">Inspect Dataset</button>
    <small class="text-muted mt-2">First worksheet is used for Excel files. Maximum upload size: 50 MB.</small>
  </form>

  {% if error %}
  <div class="alert alert-danger">{{ error }}</div>
  {% endif %}

  {% if report %}
  <div class="card p-4 mb-4">
    <div class="d-flex justify-content-between flex-wrap gap-2">
      <div>
        <h2>{{ filename }}</h2>
        <p class="text-muted">Structural assessment — not yet a model-performance evaluation.</p>
      </div>
      <a class="btn btn-outline-primary align-self-start"
         href="{{ url_for('dataset_profile_json', report_id=report_id) }}">Download JSON report</a>
    </div>

    <div class="row g-3 my-2">
      {% for label, value in [
        ('Rows', report.row_count),
        ('Columns', report.column_count),
        ('Missing cells', report.missing_cells),
        ('Duplicate rows', report.duplicate_rows),
        ('Numeric columns', report.numeric_column_count),
        ('Categorical columns', report.categorical_column_count)
      ] %}
      <div class="col-6 col-md-4">
        <div class="border rounded p-3 h-100">
          <div class="text-muted">{{ label }}</div>
          <div class="fs-4 fw-semibold">{{ value }}</div>
        </div>
      </div>
      {% endfor %}
    </div>

    <h3 class="mt-4">Initial quality indicator: {{ report.quality_indicator }}/100</h3>
    <p class="text-muted">{{ report.quality_indicator_note }}</p>
    <div class="progress mb-4" role="progressbar" aria-valuenow="{{ report.quality_indicator }}"
         aria-valuemin="0" aria-valuemax="100">
      <div class="progress-bar" style="width: {{ report.quality_indicator }}%"></div>
    </div>

    <h3>Warnings and observations</h3>
    <ul>{% for warning in report.warnings %}<li>{{ warning }}</li>{% endfor %}</ul>

    <h3 class="mt-4">Column-by-column analysis</h3>
    <div class="table-responsive">
      <table class="table table-striped table-sm">
        <thead><tr>
          <th>Column</th><th>Type</th><th>Missing</th><th>Unique</th><th>Outliers</th><th>Statistics / categories</th>
        </tr></thead>
        <tbody>
        {% for column in report.columns %}
          <tr>
            <td>{{ column.name }}</td>
            <td>{{ column.dtype }}</td>
            <td>{{ column.missing_count }} ({{ column.missing_percent }}%)</td>
            <td>{{ column.unique_count }}</td>
            <td>{{ column.get('iqr_outlier_count', '—') }}</td>
            <td>
              {% if column.statistics %}
                {% for key, value in column.statistics.items() %}<div><small>{{ key }}: {{ value if value is not none else 'N/A' }}</small></div>{% endfor %}
              {% endif %}
              {% if column.top_categories %}
                {% for category in column.top_categories[:5] %}
                  <div><small>{{ category.value }}: {{ category.count }}</small></div>
                {% endfor %}
              {% endif %}
              {% if column.constant %}<strong>Constant column</strong>{% endif %}
            </td>
          </tr>
        {% endfor %}
        </tbody>
      </table>
    </div>

    <h3 class="mt-4">Strongest numeric correlations</h3>
    {% if report.correlations %}
    <div class="table-responsive">
      <table class="table table-sm table-striped">
        <thead><tr><th>Column A</th><th>Column B</th><th>Correlation</th><th>Strength</th></tr></thead>
        <tbody>{% for item in report.correlations %}
          <tr><td>{{ item.column_a }}</td><td>{{ item.column_b }}</td>
          <td>{{ item.correlation }}</td><td>{{ item.strength }}</td></tr>
        {% endfor %}</tbody>
      </table>
    </div>
    {% else %}
    <p>At least two numeric columns are needed to calculate correlations.</p>
    {% endif %}

    <h3 class="mt-4">Possible target columns</h3>
    <p class="text-muted">These are suggestions only. Confirm the target using the client brief.</p>
    {% if report.possible_targets %}
    <div class="table-responsive">
      <table class="table table-sm table-striped">
        <thead><tr><th>Column</th><th>Type</th><th>Unique values</th><th>Suggestion</th></tr></thead>
        <tbody>{% for target in report.possible_targets %}
          <tr><td>{{ target.column }}</td><td>{{ target.dtype }}</td>
          <td>{{ target.unique_values }}</td><td>{{ target.suggestion }}</td></tr>
        {% endfor %}</tbody>
      </table>
    </div>
    {% else %}
    <p>No likely target columns found automatically.</p>
    {% endif %}
    <p class="text-muted mt-3">Outliers use the 1.5×IQR rule. Outliers are not automatically errors. The quality indicator is heuristic and should not be used alone to select a dataset.</p>
  </div>
  {% endif %}
</div>
{% endblock %}
'''

(templates_dir / "dataset_lab.html").write_text(template.strip() + "\n")

routes = r'''

@app.route("/dataset-lab", methods=["GET", "POST"])
def dataset_lab():
    import json
    import uuid
    from flask import request, render_template
    from werkzeug.utils import secure_filename
    from ml.dataset_evaluator import evaluate_dataset
    import pandas as pd

    report = None
    report_id = None
    filename = None
    error = None

    if request.method == "POST":
        upload = request.files.get("dataset")
        if not upload or not upload.filename:
            error = "Choose a CSV or Excel file."
        else:
            safe_name = secure_filename(upload.filename)
            suffix = Path(safe_name).suffix.lower()
            if suffix not in (".csv", ".xlsx", ".xls"):
                error = "Unsupported file type. Use CSV, XLSX, or XLS."
            else:
                raw = upload.read(50 * 1024 * 1024 + 1)
                if len(raw) > 50 * 1024 * 1024:
                    error = "File exceeds the 50 MB upload limit."
                else:
                    upload.seek(0)
                    upload_dir = Path(__file__).resolve().parent / "uploads"
                    report_dir = Path(__file__).resolve().parent / "reports" / "dataset_profiles"
                    upload_dir.mkdir(parents=True, exist_ok=True)
                    report_dir.mkdir(parents=True, exist_ok=True)
                    stored_name = uuid.uuid4().hex + "_" + safe_name
                    saved_path = upload_dir / stored_name
                    saved_path.write_bytes(raw)
                    try:
                        if suffix == ".csv":
                            try:
                                df = pd.read_csv(saved_path, low_memory=False)
                            except UnicodeDecodeError:
                                df = pd.read_csv(saved_path, encoding="latin-1", low_memory=False)
                        else:
                            df = pd.read_excel(saved_path)
                        if df.empty:
                            raise ValueError("The selected file contains no data rows.")
                        if len(df) > 500000 or len(df.columns) > 500:
                            raise ValueError("For safety, this first version supports up to 500,000 rows and 500 columns.")
                        report = evaluate_dataset(df)
                        report_id = uuid.uuid4().hex
                        filename = safe_name
                        report["filename"] = safe_name
                        report["source"] = "User-uploaded dataset"
                        report["report_id"] = report_id
                        report["synthetic"] = False
                        (report_dir / f"{report_id}.json").write_text(
                            json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)
                        )
                    except Exception as exc:
                        saved_path.unlink(missing_ok=True)
                        error = f"Could not inspect this file: {exc}"

    return render_template(
        "dataset_lab.html",
        report=report,
        report_id=report_id,
        filename=filename,
        error=error,
    )


@app.route("/dataset-lab/report/<report_id>.json")
def dataset_profile_json(report_id):
    import re
    from flask import abort, send_file

    if not re.fullmatch(r"[a-f0-9]{32}", report_id):
        abort(404)
    path = Path(__file__).resolve().parent / "reports" / "dataset_profiles" / f"{report_id}.json"
    if not path.is_file():
        abort(404)
    return send_file(path, as_attachment=True, download_name=f"dataset_profile_{report_id}.json")
'''

source = source.replace(startup, routes + "\n\n" + startup, 1)
app_file.write_text(source)

if base_file.exists():
    html = base_file.read_text()
    if "dataset-lab" not in html:
        link = '<a class="nav-link" href="{{ url_for(\'dataset_lab\') }}">Dataset Lab</a>\n'
        if "</nav>" in html:
            html = html.replace("</nav>", link + "</nav>", 1)
        else:
            print("NOTE: Navbar not detected; use /dataset-lab directly.")
        base_file.write_text(html)

print("Dataset Intelligence Lab installed.")
print("Backup:", backup)
print("Route: /dataset-lab")
