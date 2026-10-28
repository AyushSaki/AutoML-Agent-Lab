from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, send_file, abort
from io import BytesIO
from ml.experiment_store import ExperimentStore
from ml.reporting import generate_markdown_report
from werkzeug.utils import secure_filename
from pathlib import Path
import pandas as pd
import os
import json
import uuid
import traceback

from ml.pipeline import train_models, predict_with_saved_model
from ml.data_intelligence import analyze_dataset
from ai.ollama_client import ask_ollama

from ml.autonomous_scientist import inspect_dataset
app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "automl-agent-lab-dev")

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
REPORT_DIR = BASE_DIR / "reports"
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)
EXPERIMENT_DB = DATA_DIR / "experiments.db"
experiment_store = ExperimentStore(db_path=EXPERIMENT_DB)

UPLOAD_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)

ALLOWED_EXTENSIONS = {"csv", "xlsx", "xls"}


def allowed(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    file = request.files.get("dataset")

    if not file or not file.filename:
        flash("Please select a CSV or Excel dataset.", "error")
        return redirect(url_for("index"))

    if not allowed(file.filename):
        flash("Only CSV, XLSX and XLS files are supported.", "error")
        return redirect(url_for("index"))

    safe = secure_filename(file.filename)
    path = UPLOAD_DIR / f"{uuid.uuid4().hex}_{safe}"

    try:
        file.save(path)
        profile = analyze_dataset(path)

        # Detect a likely prediction target without replacing the EDA profile.
        target_report = inspect_dataset(path)
        target_info = target_report.get("target")
        profile["target_info"] = target_info
        profile["target_candidates"] = target_report.get(
            "target_candidates", []
        )
        profile["suggested_target"] = (
            target_info.get("column")
            if isinstance(target_info, dict)
            and target_info.get("auto_selected")
            else ""
        )
        profile["suggested_task"] = target_report.get("recommended_task", "")

        # The analysis template uses profile.file_path for model training.
        profile["file_path"] = str(path)

        # Normalize Data Intelligence Engine results for the dashboard.
        # Some profile sections are lists of dictionaries, not dictionaries.
        def normalize_rows(value, key_field, transform=None):
            if isinstance(value, dict):
                return value
            if not isinstance(value, list):
                return {}
            result = {}
            for item in value:
                if not isinstance(item, dict):
                    continue
                key = item.get(key_field)
                if key is None:
                    continue
                result[str(key)] = transform(item) if transform else item
            return result

        profile["missing_by_column"] = normalize_rows(
            profile.get("missing_by_column", {}),
            "column",
            lambda item: item.get("missing", 0)
        )

        profile["statistics"] = normalize_rows(
            profile.get("statistics", {}),
            "column",
            lambda item: {k: v for k, v in item.items() if k != "column"}
        )

        profile["outlier_summary"] = normalize_rows(
            profile.get("outlier_summary", {}),
            "column",
            lambda item: {k: v for k, v in item.items() if k != "column"}
        )

        def normalize_categories(item):
            categories = item.get("categories", [])
            if isinstance(categories, dict):
                return categories
            if isinstance(categories, list):
                return {
                    str(category.get("value", "Unknown")): category.get("count", 0)
                    for category in categories
                    if isinstance(category, dict)
                }
            return {}

        profile["categorical_distributions"] = normalize_rows(
            profile.get("categorical_distributions", {}),
            "column",
            normalize_categories
        )

        return render_template(
            "analysis.html",
            profile=profile,
            filename=safe,
            file_path=str(path),
        )

    except Exception as exc:
        traceback.print_exc()
        flash(f"Analysis failed: {exc}", "error")
        return redirect(url_for("index"))


@app.route("/train", methods=["POST"])
def train():
    file_path = request.form.get("file_path", "").strip()

    if not file_path:
        flash("Dataset path is missing. Please upload the dataset again.", "error")
        return redirect(url_for("index"))

    # Only accept datasets stored in this application's uploads directory.
    path = Path(file_path).resolve()

    try:
        path.relative_to(UPLOAD_DIR.resolve())
    except ValueError:
        flash("Invalid dataset path. Please upload the dataset again.", "error")
        return redirect(url_for("index"))

    if not path.is_file() or path.suffix.lower() not in {
        ".csv", ".xlsx", ".xls"
    }:
        flash("Dataset is no longer available. Please upload it again.", "error")
        return redirect(url_for("index"))

    target = request.form.get("target", "").strip() or None
    task = request.form.get("task", "auto").strip()

    try:
        result = train_models(path, target=target, task=task)

        best_model = result.get("best_model") or {}
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
                "best_model": best_model,
                "evaluation": result.get("evaluation", {}),
                "warnings": result.get("warnings", []),
                "saved_model": saved_model,
                "all_models": result.get("models", []),
            },
            model_path=saved_model.get("filename"),
            notes="Automatically recorded after successful model training.",
        )

        report_file = REPORT_DIR / f"{uuid.uuid4().hex}.json"
        report_file.write_text(
            json.dumps(result, indent=2, default=str),
            encoding="utf-8",
        )

        return render_template(
            "results.html",
            result=result,
            report_file=report_file.name,
        )

    except Exception as exc:
        traceback.print_exc()
        flash(f"Training failed: {exc}", "error")
        return redirect(url_for("index"))



@app.route("/predict/<model_id>", methods=["GET", "POST"])
def predict_saved_model(model_id):
    if request.method == "GET":
        return render_template("predict.html", model_id=model_id)

    file = request.files.get("dataset")
    if not file or not file.filename:
        return render_template(
            "predict.html", model_id=model_id,
            error="Please select a CSV or Excel file."
        ), 400

    if not allowed(file.filename):
        return render_template(
            "predict.html", model_id=model_id,
            error="Only CSV, XLSX and XLS files are supported."
        ), 400

    safe_name = secure_filename(file.filename)
    path = UPLOAD_DIR / f"{uuid.uuid4().hex}_{safe_name}"

    try:
        file.save(path)
        result = predict_with_saved_model(model_id, path)
        return render_template(
            "predict.html", model_id=model_id, result=result
        )
    except (ValueError, FileNotFoundError) as exc:
        return render_template(
            "predict.html", model_id=model_id, error=str(exc)
        ), 400
    except Exception:
        traceback.print_exc()
        return render_template(
            "predict.html", model_id=model_id,
            error="Prediction failed. Check the terminal for details."
        ), 500
    finally:
        path.unlink(missing_ok=True)

@app.route("/ask", methods=["POST"])
def ask():
    question = request.form.get("question", "").strip()
    context = request.form.get("context", "").strip()

    if not question:
        return jsonify({"answer": "Please enter a question."}), 400

    # Limit request size sent to the local language model.
    answer = ask_ollama(question[:4000], context=context[:20000])
    return jsonify({"answer": answer})




@app.route("/autonomous-plan", methods=["POST"])
def autonomous_plan():

    payload = request.get_json(silent=True) or {}
    file_path = str(payload.get("file_path", "")).strip()
    target = str(payload.get("target", "")).strip() or None

    if not file_path:
        return jsonify({"error": "Dataset path is missing."}), 400

    path = Path(file_path).resolve()
    try:
        path.relative_to(UPLOAD_DIR.resolve())
    except ValueError:
        return jsonify({"error": "Invalid dataset path."}), 400

    if not path.is_file() or path.suffix.lower() not in {".csv", ".xlsx", ".xls"}:
        return jsonify({"error": "Dataset is unavailable or unsupported."}), 400

    try:
        report = inspect_dataset(path, target=target)

        # Local LLM receives observed dataset facts, not the entire dataset.
        context = json.dumps(report, ensure_ascii=False, default=str)
        question = (
            "Act as a careful senior data scientist. Based only on these "
            "dataset inspection results, create a practical analysis plan. "
            "Cover data quality, target/task suitability, preprocessing, "
            "candidate models, evaluation metrics, and risks. Clearly label "
            "uncertainties. Do not invent facts or claim that models were trained. "
            "If no target is selected, explain how to choose one or when "
            "unsupervised learning may fit."
        )
        report["ai_plan"] = ask_ollama(question, context=context[:18000])
        return jsonify(report)
    except Exception as exc:
        traceback.print_exc()
        return jsonify({"error": f"Could not create analysis plan: {exc}"}), 500

@app.route("/experiments")
def experiments():
    records = experiment_store.list_experiments(limit=100)
    return render_template("experiments.html", experiments=records)


@app.route("/experiments/<int:experiment_id>")
def experiment_detail(experiment_id):
    experiment = experiment_store.get_experiment(experiment_id)

    if experiment is None:
        abort(404)

    return render_template(
        "experiment_detail.html",
        experiment=experiment,
    )


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


@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "application": "AutoML-Agent-Lab",
    })




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


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5004, debug=False)
