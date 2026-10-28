from pathlib import Path
import shutil
import sys
from datetime import datetime

ROOT = Path.cwd()

APP = ROOT / "app.py"
BASE = ROOT / "templates" / "base.html"
EXPERIMENTS = ROOT / "templates" / "experiments.html"

required = [APP, BASE, EXPERIMENTS]

for path in required:
    if not path.exists():
        print(f"ERROR: Missing {path}")
        sys.exit(1)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
backup = ROOT / "backups" / f"history_upgrade_{timestamp}"
backup.mkdir(parents=True, exist_ok=False)

for path in required:
    shutil.copy2(path, backup / path.name)

print("Backup created:", backup)

app = APP.read_text(encoding="utf-8")
base = BASE.read_text(encoding="utf-8")

# ---------------------------------------------------------
# 1. Add experiment detail route
# ---------------------------------------------------------

route_anchor = '@app.route("/experiments/<int:experiment_id>/report")'

if route_anchor not in app:
    print("ERROR: Experiment report route not found.")
    sys.exit(1)

detail_route = '''@app.route("/experiments/<int:experiment_id>")
def experiment_detail(experiment_id):
    experiment = experiment_store.get_experiment(experiment_id)

    if experiment is None:
        abort(404)

    return render_template(
        "experiment_detail.html",
        experiment=experiment,
    )


'''

if "def experiment_detail(experiment_id):" not in app:
    app = app.replace(
        route_anchor,
        detail_route + route_anchor,
        1,
    )

# ---------------------------------------------------------
# 2. Upgrade experiment recording
# ---------------------------------------------------------

old = '''        metrics = {
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

new = '''        metrics = {
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
                "best_model": best_model,
                "evaluation": result.get("evaluation", {}),
                "warnings": result.get("warnings", []),
                "saved_model": saved_model,
                "all_models": result.get("models", []),
            },
            model_path=saved_model.get("filename"),
            notes="Automatically recorded after successful model training.",
        )
'''

if old not in app:
    print("ERROR: Existing experiment logging block not found.")
    print("No source files changed except backups.")
    sys.exit(1)

app = app.replace(old, new, 1)

# ---------------------------------------------------------
# 3. Create detailed experiment template
# ---------------------------------------------------------

detail_template = '''{% extends "base.html" %}

{% block title %}Experiment {{ experiment.id }} | AutoML-Agent-Lab{% endblock %}

{% block content %}

<div class="d-flex flex-wrap justify-content-between align-items-center gap-2 mb-4">

    <div>
        <a href="{{ url_for('experiments') }}"
           class="btn btn-outline-secondary btn-sm mb-3">
            &larr; Experiment History
        </a>

        <h2 class="fw-bold mb-1">
            Experiment #{{ experiment.id }}
        </h2>

        <p class="text-secondary mb-0">
            {{ experiment.dataset_name }}
            · {{ experiment.task|title }}
            · {{ experiment.algorithm or "Unknown model" }}
        </p>
    </div>

    <a class="btn btn-primary"
       href="{{ url_for('experiment_report', experiment_id=experiment.id) }}">
        Download Markdown Report
    </a>

</div>

<div class="row g-3 mb-4">

    <div class="col-md-3">
        <div class="metric-card h-100">
            <span>Task</span>
            <strong>{{ experiment.task|title }}</strong>
            <small class="text-secondary">
                Machine learning problem type
            </small>
        </div>
    </div>

    <div class="col-md-3">
        <div class="metric-card h-100">
            <span>Best Model</span>
            <strong>{{ experiment.algorithm or "N/A" }}</strong>
            <small class="text-secondary">
                Selected by cross-validation
            </small>
        </div>
    </div>

    <div class="col-md-3">
        <div class="metric-card h-100">
            <span>Rows</span>
            <strong>{{ experiment.rows_count or "N/A" }}</strong>
            <small class="text-secondary">
                Training dataset rows
            </small>
        </div>
    </div>

    <div class="col-md-3">
        <div class="metric-card h-100">
            <span>Features</span>
            <strong>{{ experiment.features_count or "N/A" }}</strong>
            <small class="text-secondary">
                Input columns
            </small>
        </div>
    </div>

</div>

{% set params = experiment.parameters or {} %}
{% set evaluation = params.get("evaluation", {}) %}
{% set best = params.get("best_model", {}) %}
{% set warnings = params.get("warnings", []) %}
{% set saved = params.get("saved_model", {}) %}
{% set all_models = params.get("all_models", []) %}

<div class="card border-0 shadow-sm mb-4">
    <div class="card-body">

        <h4 class="fw-bold">Experiment Information</h4>

        <div class="row">

            <div class="col-md-6">
                <dl class="row mb-0">

                    <dt class="col-sm-5">Experiment ID</dt>
                    <dd class="col-sm-7">{{ experiment.id }}</dd>

                    <dt class="col-sm-5">Dataset</dt>
                    <dd class="col-sm-7">{{ experiment.dataset_name }}</dd>

                    <dt class="col-sm-5">Task</dt>
                    <dd class="col-sm-7">{{ experiment.task }}</dd>

                    <dt class="col-sm-5">Algorithm</dt>
                    <dd class="col-sm-7">{{ experiment.algorithm or "—" }}</dd>

                </dl>
            </div>

            <div class="col-md-6">
                <dl class="row mb-0">

                    <dt class="col-sm-5">Created</dt>
                    <dd class="col-sm-7">{{ experiment.created_at }}</dd>

                    <dt class="col-sm-5">Target</dt>
                    <dd class="col-sm-7">
                        {{ params.get("target") or "—" }}
                    </dd>

                    <dt class="col-sm-5">Model ID</dt>
                    <dd class="col-sm-7">
                        {{ saved.get("model_id", "—") }}
                    </dd>

                    <dt class="col-sm-5">Model File</dt>
                    <dd class="col-sm-7">
                        {{ experiment.model_path or "—" }}
                    </dd>

                </dl>
            </div>

        </div>

    </div>
</div>

<div class="card border-0 shadow-sm mb-4">

    <div class="card-body">

        <h4 class="fw-bold mb-3">Best Model Metrics</h4>

        <div class="table-responsive">

            <table class="table table-bordered align-middle">

                <thead class="table-light">
                    <tr>
                        <th>Metric</th>
                        <th>Value</th>
                    </tr>
                </thead>

                <tbody>

                {% for key, value in experiment.metrics.items() %}

                    <tr>
                        <td>{{ key|replace("_", " ")|title }}</td>
                        <td><strong>{{ value }}</strong></td>
                    </tr>

                {% else %}

                    <tr>
                        <td colspan="2" class="text-secondary">
                            No metrics recorded.
                        </td>
                    </tr>

                {% endfor %}

                </tbody>

            </table>

        </div>

    </div>

</div>

{% if evaluation %}

<div class="card border-0 shadow-sm mb-4">

    <div class="card-body">

        <h4 class="fw-bold">Evaluation</h4>

        <div class="row g-3">

            {% if evaluation.get("train_rows") is not none %}
            <div class="col-md-4">
                <div class="p-3 border rounded">
                    <small class="text-secondary">Training Rows</small>
                    <div class="fs-4 fw-bold">
                        {{ evaluation.get("train_rows") }}
                    </div>
                </div>
            </div>
            {% endif %}

            {% if evaluation.get("test_rows") is not none %}
            <div class="col-md-4">
                <div class="p-3 border rounded">
                    <small class="text-secondary">Test Rows</small>
                    <div class="fs-4 fw-bold">
                        {{ evaluation.get("test_rows") }}
                    </div>
                </div>
            </div>
            {% endif %}

            {% if best.get("cv_folds") is not none %}
            <div class="col-md-4">
                <div class="p-3 border rounded">
                    <small class="text-secondary">CV Folds</small>
                    <div class="fs-4 fw-bold">
                        {{ best.get("cv_folds") }}
                    </div>
                </div>
            </div>
            {% endif %}

        </div>

        {% if experiment.task == "classification" and evaluation.get("confusion_matrix") %}

        {% set cm = evaluation.get("confusion_matrix") %}

        <hr>

        <h5 class="fw-bold">Confusion Matrix</h5>

        <div class="table-responsive">

            <table class="table table-bordered text-center">

                <thead class="table-light">
                    <tr>
                        <th>Actual / Predicted</th>

                        {% for label in cm.labels %}
                        <th>{{ label }}</th>
                        {% endfor %}

                    </tr>
                </thead>

                <tbody>

                    {% for row in cm.matrix %}

                    <tr>

                        <th class="table-light">
                            {{ cm.labels[loop.index0] }}
                        </th>

                        {% for value in row %}
                        <td>{{ value }}</td>
                        {% endfor %}

                    </tr>

                    {% endfor %}

                </tbody>

            </table>

        </div>

        {% endif %}

    </div>

</div>

{% endif %}

{% if warnings %}

<div class="card border-warning mb-4">

    <div class="card-body">

        <h4 class="fw-bold text-warning-emphasis">
            Dataset Warnings
        </h4>

        <ul class="mb-0">

        {% for warning in warnings %}
            <li class="mb-2">{{ warning }}</li>
        {% endfor %}

        </ul>

    </div>

</div>

{% endif %}

{% if all_models %}

<div class="card border-0 shadow-sm mb-4">

    <div class="card-body">

        <h4 class="fw-bold">Model Comparison</h4>

        <div class="table-responsive">

            <table class="table table-hover align-middle">

                <thead class="table-light">

                    <tr>

                        <th>Model</th>

                        {% if experiment.task == "classification" %}

                        <th>Accuracy</th>
                        <th>Precision</th>
                        <th>Recall</th>
                        <th>F1</th>
                        <th>CV Accuracy</th>
                        <th>CV F1</th>

                        {% else %}

                        <th>MAE</th>
                        <th>RMSE</th>
                        <th>R²</th>
                        <th>CV MAE</th>
                        <th>CV R²</th>

                        {% endif %}

                    </tr>

                </thead>

                <tbody>

                {% for model in all_models %}

                    <tr class="{% if model.get('model') == experiment.algorithm %}table-success{% endif %}">

                        <td>
                            <strong>{{ model.get("model", "Unknown") }}</strong>

                            {% if model.get("model") == experiment.algorithm %}
                            <span class="badge text-bg-success">Best</span>
                            {% endif %}
                        </td>

                        {% if experiment.task == "classification" %}

                        <td>{{ model.get("accuracy", "—") }}</td>
                        <td>{{ model.get("precision", "—") }}</td>
                        <td>{{ model.get("recall", "—") }}</td>
                        <td>{{ model.get("f1", "—") }}</td>
                        <td>{{ model.get("cv_accuracy_mean", "—") }}</td>
                        <td>{{ model.get("cv_f1_mean", "—") }}</td>

                        {% else %}

                        <td>{{ model.get("mae", "—") }}</td>
                        <td>{{ model.get("rmse", "—") }}</td>
                        <td>{{ model.get("r2", "—") }}</td>
                        <td>{{ model.get("cv_mae_mean", "—") }}</td>
                        <td>{{ model.get("cv_r2_mean", "—") }}</td>

                        {% endif %}

                    </tr>

                {% endfor %}

                </tbody>

            </table>

        </div>

    </div>

</div>

{% endif %}

<div class="d-flex gap-2 mb-5">

    <a href="{{ url_for('experiments') }}"
       class="btn btn-outline-secondary">
        Back to History
    </a>

    <a href="{{ url_for('index') }}"
       class="btn btn-primary">
        New Experiment
    </a>

</div>

{% endblock %}
'''

detail_file = ROOT / "templates" / "experiment_detail.html"
detail_file.write_text(detail_template, encoding="utf-8")

# ---------------------------------------------------------
# 4. Upgrade experiment list
# ---------------------------------------------------------

history_template = '''{% extends "base.html" %}

{% block title %}Experiment History | AutoML-Agent-Lab{% endblock %}

{% block content %}

<div class="d-flex flex-wrap justify-content-between align-items-center gap-2 mb-4">

    <div>
        <h2 class="fw-bold mb-1">Experiment History</h2>
        <p class="text-secondary mb-0">
            Saved AutoML training experiments.
        </p>
    </div>

    <a href="{{ url_for('index') }}"
       class="btn btn-primary">
        + New Experiment
    </a>

</div>

{% if experiments %}

<div class="row g-4">

{% for exp in experiments %}

    {% set params = exp.parameters or {} %}
    {% set best = params.get("best_model", {}) %}
    {% set warnings = params.get("warnings", []) %}

    <div class="col-lg-6">

        <div class="card border-0 shadow-sm h-100">

            <div class="card-body">

                <div class="d-flex justify-content-between align-items-start gap-3 mb-3">

                    <div>

                        <span class="badge text-bg-light border">
                            #{{ exp.id }}
                        </span>

                        <h4 class="fw-bold mt-2 mb-1">
                            {{ exp.dataset_name }}
                        </h4>

                        <div class="text-secondary small">
                            {{ exp.task|title }}
                            ·
                            {{ exp.algorithm or "Unknown model" }}
                        </div>

                    </div>

                    <span class="badge text-bg-success">
                        Completed
                    </span>

                </div>

                <div class="row g-2 mb-3">

                    <div class="col-6">

                        <div class="border rounded p-2">

                            <small class="text-secondary d-block">
                                Best Model
                            </small>

                            <strong>
                                {{ exp.algorithm or "N/A" }}
                            </strong>

                        </div>

                    </div>

                    <div class="col-6">

                        <div class="border rounded p-2">

                            <small class="text-secondary d-block">
                                Rows
                            </small>

                            <strong>
                                {{ exp.rows_count or "N/A" }}
                            </strong>

                        </div>

                    </div>

                    <div class="col-6">

                        <div class="border rounded p-2">

                            <small class="text-secondary d-block">
                                Features
                            </small>

                            <strong>
                                {{ exp.features_count or "N/A" }}
                            </strong>

                        </div>

                    </div>

                    <div class="col-6">

                        <div class="border rounded p-2">

                            <small class="text-secondary d-block">
                                Warnings
                            </small>

                            <strong>
                                {{ warnings|length }}
                            </strong>

                        </div>

                    </div>

                </div>

                {% if exp.metrics %}

                <div class="border rounded p-3 mb-3">

                    <h6 class="fw-bold">
                        Best Model Metrics
                    </h6>

                    {% for key, value in exp.metrics.items() %}

                    <div class="d-flex justify-content-between small py-1">

                        <span class="text-secondary">
                            {{ key|replace("_", " ")|title }}
                        </span>

                        <strong>{{ value }}</strong>

                    </div>

                    {% endfor %}

                </div>

                {% endif %}

                <div class="small text-secondary mb-3">
                    {{ exp.created_at }}
                </div>

                <div class="d-flex flex-wrap gap-2">

                    <a class="btn btn-primary btn-sm"
                       href="{{ url_for('experiment_detail', experiment_id=exp.id) }}">
                        View Details
                    </a>

                    <a class="btn btn-outline-primary btn-sm"
                       href="{{ url_for('experiment_report', experiment_id=exp.id) }}">
                        Markdown Report
                    </a>

                </div>

            </div>

        </div>

    </div>

{% endfor %}

</div>

{% else %}

<div class="card border-0 shadow-sm">

    <div class="card-body text-center py-5">

        <h4 class="fw-bold">No experiments yet</h4>

        <p class="text-secondary">
            Train your first model to create an experiment record.
        </p>

        <a href="{{ url_for('index') }}"
           class="btn btn-primary">
            Start Experiment
        </a>

    </div>

</div>

{% endif %}

{% endblock %}
'''

EXPERIMENTS.write_text(history_template, encoding="utf-8")

# ---------------------------------------------------------
# 5. Make sure navbar contains history
# ---------------------------------------------------------

navbar_marker = '<a class="nav-link" href="{{ url_for(\'experiments\') }}">Experiment History</a>'

if navbar_marker not in base:
    anchor = '<span class="navbar-text text-secondary small">Autonomous Machine Learning Platform</span>'

    if anchor not in base:
        print("ERROR: Navbar anchor not found.")
        sys.exit(1)

    base = base.replace(
        anchor,
        navbar_marker + "\n        " + anchor,
        1,
    )

BASE.write_text(base, encoding="utf-8")
APP.write_text(app, encoding="utf-8")

print()
print("======================================")
print("Experiment History upgrade complete")
print("======================================")
print("Changed:")
print("  - app.py")
print("  - templates/base.html")
print("  - templates/experiments.html")
print("  - templates/experiment_detail.html")
print()
print("Backup:", backup)
