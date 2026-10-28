from pathlib import Path
from datetime import datetime
import shutil
import sys

root = Path.cwd()
app_file = root / "app.py"
base_file = root / "templates" / "base.html"
results_file = root / "templates" / "results.html"
store_file = root / "ml" / "experiment_store.py"
reporting_file = root / "ml" / "reporting.py"

required = [app_file, base_file, results_file, store_file, reporting_file]
missing = [str(p.relative_to(root)) for p in required if not p.is_file()]
if missing:
    print("STOP: Required files are missing:", missing)
    sys.exit(1)

app = app_file.read_text(encoding="utf-8")
base = base_file.read_text(encoding="utf-8")

if "from ml.experiment_store import ExperimentStore" in app:
    print("Experiment History imports already exist. No changes made.")
    sys.exit(0)

train_anchor = "        result = train_models(path, target=target, task=task)"
health_anchor = '@app.route("/health")'
navbar_anchor = '<span class="navbar-text text-secondary small">Autonomous Machine Learning Platform</span>'

checks = [
    (train_anchor in app, "training call"),
    (health_anchor in app, "health route"),
    (navbar_anchor in base, "navbar"),
]
failed = [name for ok, name in checks if not ok]
if failed:
    print("STOP: Could not safely locate:", ", ".join(failed))
    print("No project files changed.")
    sys.exit(1)

if "class ExperimentStore" not in store_file.read_text(encoding="utf-8"):
    print("STOP: ExperimentStore class was not found.")
    sys.exit(1)

if "def generate_markdown_report" not in reporting_file.read_text(encoding="utf-8"):
    print("STOP: generate_markdown_report was not found.")
    sys.exit(1)

backup = root / "backups" / f"experiment_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
backup.mkdir(parents=True, exist_ok=False)
for p in [app_file, base_file, results_file]:
    shutil.copy2(p, backup / p.name)

app = app.replace(
    "from flask import Flask, render_template, request, redirect, url_for, flash, jsonify",
    "from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, send_file, abort\nfrom io import BytesIO\nfrom ml.experiment_store import ExperimentStore\nfrom ml.reporting import generate_markdown_report",
    1,
)

app = app.replace(
    'REPORT_DIR = BASE_DIR / "reports"',
    'REPORT_DIR = BASE_DIR / "reports"\nDATA_DIR = BASE_DIR / "data"\nDATA_DIR.mkdir(exist_ok=True)\nEXPERIMENT_DB = DATA_DIR / "experiments.db"\nexperiment_store = ExperimentStore(db_path=EXPERIMENT_DB)',
    1,
)

logging_block = '''        best_model = result.get("best_model") or {}
        saved_model = result.get("saved_model") or {}
        metrics = {
            key: value
            for key, value in best_model.items()
            if key != "model"
        }
        experiment_store.log_experiment(
            name=f"{result.get('dataset', path.name)} - {result.get('target', 'model')}",
            task=str(result.get("task", task)),
            algorithm=str(best_model.get("model", "Unknown")),
            dataset_name=str(result.get("dataset", path.name)),
            rows_count=int(result.get("rows", 0)),
            features_count=int(result.get("features", 0)),
            metrics=metrics,
            parameters={
                "target": result.get("target"),
                "evaluation": result.get("evaluation", {}),
                "warnings": result.get("warnings", []),
                "saved_model_id": saved_model.get("model_id"),
            },
            model_path=saved_model.get("filename"),
            notes="Automatically recorded after successful model training.",
        )

'''

app = app.replace(train_anchor, train_anchor + "\n\n" + logging_block.rstrip(), 1)

history_routes = '''@app.route("/experiments")
def experiments():
    records = experiment_store.list_experiments(limit=100)
    return render_template("experiments.html", experiments=records)


@app.route("/experiments/<int:experiment_id>/report")
def experiment_report(experiment_id):
    experiment = experiment_store.get_experiment(experiment_id)
    if experiment is None:
        abort(404)

    markdown = generate_markdown_report(experiment)
    content = markdown.encode("utf-8")
    return send_file(
        BytesIO(content),
        mimetype="text/markdown; charset=utf-8",
        as_attachment=True,
        download_name=f"automl-experiment-{experiment_id}-report.md",
    )


'''

app = app.replace(health_anchor, history_routes + health_anchor, 1)

base = base.replace(
    navbar_anchor,
    '<a class="nav-link" href="{{ url_for(\'experiments\') }}">Experiment History</a>\n        ' + navbar_anchor,
    1,
)

experiments_template = '''{% extends "base.html" %}

{% block title %}Experiment History | AutoML-Agent-Lab{% endblock %}

{% block content %}
<div class="container py-4">
    <div class="d-flex flex-wrap justify-content-between align-items-center gap-2 mb-4">
        <div>
            <h1 class="h3 mb-1">Experiment History</h1>
            <p class="text-secondary mb-0">Your most recent model-training runs.</p>
        </div>
        <a class="btn btn-outline-secondary" href="{{ url_for('index') }}">Back to Dashboard</a>
    </div>

    {% if experiments %}
    <div class="table-responsive">
        <table class="table table-striped table-hover align-middle">
            <thead>
                <tr>
                    <th>ID</th>
                    <th>Experiment</th>
                    <th>Task</th>
                    <th>Algorithm</th>
                    <th>Dataset</th>
                    <th>Rows</th>
                    <th>Created</th>
                    <th>Report</th>
                </tr>
            </thead>
            <tbody>
                {% for exp in experiments %}
                <tr>
                    <td>{{ exp.id }}</td>
                    <td>{{ exp.name }}</td>
                    <td>{{ exp.task }}</td>
                    <td>{{ exp.algorithm or "—" }}</td>
                    <td>{{ exp.dataset_name or "—" }}</td>
                    <td>{{ exp.rows_count if exp.rows_count is not none else "—" }}</td>
                    <td>{{ exp.created_at }}</td>
                    <td>
                        <a class="btn btn-sm btn-outline-primary"
                           href="{{ url_for('experiment_report', experiment_id=exp.id) }}">
                            Download Markdown
                        </a>
                    </td>
                </tr>
                {% endfor %}
            </tbody>
        </table>
    </div>
    {% else %}
    <div class="alert alert-info">
        No experiments have been recorded yet. Train a model from the dashboard to start your history.
    </div>
    {% endif %}
</div>
{% endblock %}
'''

new_template = root / "templates" / "experiments.html"
if new_template.exists():
    print("STOP: templates/experiments.html already exists; no files written.")
    sys.exit(1)

app_file.write_text(app, encoding="utf-8")
base_file.write_text(base, encoding="utf-8")
new_template.write_text(experiments_template, encoding="utf-8")

print("Experiment History integration files updated.")
print("Backup:", backup.relative_to(root))
