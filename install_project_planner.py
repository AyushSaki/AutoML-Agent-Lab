from pathlib import Path
from datetime import datetime
import shutil
import sys

ROOT = Path.cwd()
APP = ROOT / "app.py"
ML = ROOT / "ml"
TEMPLATES = ROOT / "templates"
DATA = ROOT / "data" / "project_briefs"

if not APP.exists():
    sys.exit("ERROR: app.py not found. Run this installer from AutoML-Agent-Lab.")

source = APP.read_text()
if "def project_planner" in source or "/project-planner" in source:
    sys.exit("Project Planner routes already appear to exist. No changes made.")

if "if __name__ == '__main__':" not in source and 'if __name__ == "__main__":' not in source:
    sys.exit("Could not safely locate the app startup block. No files changed.")

backup = ROOT / "backups" / ("project_planner_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
backup.mkdir(parents=True, exist_ok=True)
shutil.copy2(APP, backup / "app.py")

for name in ("base.html", "index.html"):
    path = TEMPLATES / name
    if path.exists():
        shutil.copy2(path, backup / name)

ML.mkdir(exist_ok=True)
TEMPLATES.mkdir(exist_ok=True)
DATA.mkdir(parents=True, exist_ok=True)

module = r'''
import re

def analyse_requirements(topic, problem, users="", constraints="", success="", deployment="Flask"):
    text = " ".join([topic, problem]).lower()
    task_rules = [
        ("computer_vision", ["image", "photo", "visual defect", "object detection", "camera"]),
        ("nlp", ["text", "document", "chatbot", "language", "review", "summar"]),
        ("time_series", ["forecast", "forecasting", "future sales", "demand prediction", "time series"]),
        ("classification", ["classify", "classification", "fraud", "churn", "loan approval", "yes or no", "category"]),
        ("regression", ["price prediction", "predict price", "predict cost", "house price", "regression", "predict sales"]),
        ("clustering", ["segment", "segmentation", "group customers", "clustering"]),
        ("anomaly_detection", ["anomaly", "outlier detection", "detect defects", "unusual activity"]),
    ]

    detected = "needs_clarification"
    for task, keywords in task_rules:
        if any(keyword in text for keyword in keywords):
            detected = task
            break

    questions = []
    if not topic.strip():
        questions.append("What is the project topic and business problem?")
    if not problem.strip():
        questions.append("What decision or outcome should the system help with?")
    if not users.strip():
        questions.append("Who will use the system and what will they do with its output?")
    if not success.strip():
        questions.append("How will success be measured, and what level of error is acceptable?")
    if detected in ("regression", "classification") and not any(
        word in text for word in ["target", "column", "label", "price", "cost", "sales",
                                  "fraud", "churn", "approval", "predict"]
    ):
        questions.append("Which column should the model predict?")
    if not constraints.strip():
        questions.append("Are there constraints on budget, privacy, hardware, data volume, or tools?")

    phases = [
        {"phase": 1, "name": "Clarify requirements", "deliverable": "Approved project brief and success criteria"},
        {"phase": 2, "name": "Assess datasets", "deliverable": "Data quality, suitability, source and licence review"},
        {"phase": 3, "name": "Build baseline", "deliverable": "Preprocessing pipeline and baseline model"},
        {"phase": 4, "name": "Compare and validate", "deliverable": "Consistent holdout or cross-validation results"},
        {"phase": 5, "name": "Package delivery", "deliverable": "Model artifact, inference code, report and setup guide"},
    ]

    if detected == "needs_clarification":
        recommendation = "Confirm the problem type and intended output before selecting algorithms."
    elif detected == "regression":
        recommendation = "Predict a numeric value. Confirm the target column, units, and suitable error metrics."
    elif detected == "classification":
        recommendation = "Predict a class or category. Confirm the target column, class balance, and precision/recall priorities."
    elif detected == "clustering":
        recommendation = "Discover groups without a labelled target. Define how useful clusters will be judged."
    elif detected == "anomaly_detection":
        recommendation = "Identify unusual cases. Confirm whether labelled examples exist and how false alarms will be handled."
    elif detected == "nlp":
        recommendation = "Identify the language task, text fields, expected output, and any sensitive information."
    elif detected == "computer_vision":
        recommendation = "Confirm image types, defect/object labels, image volume, and annotation availability."
    else:
        recommendation = "Confirm the time column, forecast horizon, frequency, and baseline forecasting metric."

    return {
        "topic": topic.strip(),
        "business_problem": problem.strip(),
        "intended_users": users.strip(),
        "constraints": constraints.strip(),
        "success_criteria": success.strip(),
        "deployment": deployment.strip() or "To be decided",
        "detected_task": detected,
        "task_confidence": "Initial keyword-based suggestion; requires user confirmation",
        "recommendation": recommendation,
        "clarifying_questions": questions,
        "future_scope": phases,
        "data_policy": [
            "Validate actual dataset contents before making suitability claims.",
            "Record source and licence for externally obtained datasets.",
            "Clearly label synthetic data and never describe it as real-world data.",
            "Compare dataset quality separately from model performance.",
            "Use held-out data or cross-validation to reduce evaluation leakage.",
        ],
    }
'''

(ML / "project_analyst.py").write_text(module.strip() + "\n")

template = r'''{% extends "base.html" %}
{% block title %}Project Planner | AutoML-Agent-Lab{% endblock %}
{% block content %}
<div class="container py-4">
  <div class="mb-4">
    <h1>Autonomous Project Planner</h1>
    <p>Describe the client's problem. Generate a structured project brief before choosing datasets or training models.</p>
  </div>

  <form method="post" class="card p-4 mb-4">
    <div class="mb-3">
      <label class="form-label" for="topic">Project topic</label>
      <input class="form-control" id="topic" name="topic" required
             value="{{ form.get('topic', '') }}" placeholder="Example: House price prediction">
    </div>
    <div class="mb-3">
      <label class="form-label" for="problem">Client's business problem</label>
      <textarea class="form-control" id="problem" name="problem" rows="3" required
                placeholder="What problem should the system solve?">{{ form.get('problem', '') }}</textarea>
    </div>
    <div class="mb-3">
      <label class="form-label" for="users">Intended users</label>
      <input class="form-control" id="users" name="users"
             value="{{ form.get('users', '') }}" placeholder="Example: buyers, analysts, internal staff">
    </div>
    <div class="mb-3">
      <label class="form-label" for="success">Success criteria</label>
      <textarea class="form-control" id="success" name="success" rows="2"
                placeholder="Example: MAE below a business-approved threshold">{{ form.get('success', '') }}</textarea>
    </div>
    <div class="mb-3">
      <label class="form-label" for="constraints">Constraints and requirements</label>
      <textarea class="form-control" id="constraints" name="constraints" rows="2"
                placeholder="Example: free tools, local processing, privacy, available hardware">{{ form.get('constraints', '') }}</textarea>
    </div>
    <div class="mb-3">
      <label class="form-label" for="deployment">Preferred deployment</label>
      <select class="form-select" id="deployment" name="deployment">
        {% for option in ['Flask', 'Local notebook', 'REST API', 'Cloud deployment', 'Not decided'] %}
        <option value="{{ option }}" {% if form.get('deployment', 'Flask') == option %}selected{% endif %}>{{ option }}</option>
        {% endfor %}
      </select>
    </div>
    <button class="btn btn-primary" type="submit">Analyse Requirements</button>
  </form>

  {% if brief %}
  <section class="card p-4 mb-4">
    <h2>Project Brief</h2>
    <p><strong>Topic:</strong> {{ brief.topic }}</p>
    <p><strong>Business problem:</strong> {{ brief.business_problem }}</p>
    <p><strong>Suggested task:</strong> <code>{{ brief.detected_task }}</code></p>
    <p><strong>Confidence note:</strong> {{ brief.task_confidence }}</p>
    <p>{{ brief.recommendation }}</p>
    <p><strong>Deployment:</strong> {{ brief.deployment }}</p>

    <h3 class="mt-3">Clarify before building</h3>
    {% if brief.clarifying_questions %}
    <ul>{% for question in brief.clarifying_questions %}<li>{{ question }}</li>{% endfor %}</ul>
    {% else %}
    <p>No basic gaps were detected. Review the brief and confirm the requirements before training.</p>
    {% endif %}

    <h3 class="mt-3">Implementation roadmap</h3>
    {% for phase in brief.future_scope %}
    <div class="border rounded p-3 mb-2">
      <strong>{{ phase.phase }}. {{ phase.name }}</strong>
      <div>{{ phase.deliverable }}</div>
    </div>
    {% endfor %}

    <h3 class="mt-3">Data and evaluation safeguards</h3>
    <ul>{% for policy in brief.data_policy %}<li>{{ policy }}</li>{% endfor %}</ul>
    <p class="text-muted">This first version uses deterministic keyword-based task suggestions. It does not claim to have searched or verified external datasets.</p>
    <a class="btn btn-outline-primary" href="{{ url_for('project_brief_json', brief_id=brief_id) }}">Download Project Brief JSON</a>
  </section>
  {% endif %}
</div>
{% endblock %}
'''

(TEMPLATES / "project_planner.html").write_text(template.strip() + "\n")

routes = r'''

@app.route("/project-planner", methods=["GET", "POST"])
def project_planner():
    from ml.project_analyst import analyse_requirements
    import json
    import uuid
    from datetime import datetime, timezone

    form = {}
    brief = None
    brief_id = None

    if request.method == "POST":
        form = {
            "topic": request.form.get("topic", "").strip(),
            "problem": request.form.get("problem", "").strip(),
            "users": request.form.get("users", "").strip(),
            "success": request.form.get("success", "").strip(),
            "constraints": request.form.get("constraints", "").strip(),
            "deployment": request.form.get("deployment", "Flask").strip(),
        }
        if not form["topic"] or not form["problem"]:
            return render_template("project_planner.html", form=form, brief=None, brief_id=None), 400

        brief = analyse_requirements(
            topic=form["topic"],
            problem=form["problem"],
            users=form["users"],
            constraints=form["constraints"],
            success=form["success"],
            deployment=form["deployment"],
        )
        brief_id = uuid.uuid4().hex
        brief["brief_id"] = brief_id
        brief["created_at"] = datetime.now(timezone.utc).isoformat()
        brief["status"] = "draft_requires_user_review"

        storage = Path(__file__).resolve().parent / "data" / "project_briefs"
        storage.mkdir(parents=True, exist_ok=True)
        (storage / f"{brief_id}.json").write_text(
            json.dumps(brief, indent=2, ensure_ascii=False)
        )

    return render_template("project_planner.html", form=form, brief=brief, brief_id=brief_id)


@app.route("/project-planner/<brief_id>.json")
def project_brief_json(brief_id):
    import re
    from flask import abort, send_file

    if not re.fullmatch(r"[a-f0-9]{32}", brief_id):
        abort(404)
    path = Path(__file__).resolve().parent / "data" / "project_briefs" / f"{brief_id}.json"
    if not path.is_file():
        abort(404)
    return send_file(path, as_attachment=True, download_name=f"project_brief_{brief_id}.json")
'''

# Route handlers must be defined before app.run() executes.
marker = "if __name__ == '__main__':"
if marker not in source:
    marker = 'if __name__ == "__main__":'
source = source.replace(marker, routes + "\n\n" + marker, 1)

# Confirm the app has the imports required by the new routes.
if "from pathlib import Path" not in source:
    source = "from pathlib import Path\n" + source
if "from flask import" in source:
    lines = source.splitlines()
    for i, line in enumerate(lines):
        if line.startswith("from flask import "):
            names = line[len("from flask import "):].split(", ")
            for required in ("render_template", "request"):
                if required not in names:
                    names.append(required)
            lines[i] = "from flask import " + ", ".join(names)
            break
    source = "\n".join(lines) + "\n"
else:
    source = "from flask import render_template, request\n" + source

APP.write_text(source)

# Add a navbar link when the known base template has a navbar block.
base = TEMPLATES / "base.html"
if base.exists():
    html = base.read_text()
    if "project-planner" not in html:
        anchor = "</nav>"
        link = '<a class="nav-link" href="{{ url_for(\'project_planner\') }}">Project Planner</a>\n'
        if anchor in html:
            html = html.replace(anchor, link + anchor, 1)
        else:
            print("NOTE: No closing nav tag found; open /project-planner directly.")
        base.write_text(html)

print("Project Planner files installed.")
print(f"Backup saved at: {backup}")
print("Next: compile the app and run tests.")
