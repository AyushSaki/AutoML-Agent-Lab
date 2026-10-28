"""Generate Markdown reports from saved experiment results."""

import json
import math
from pathlib import Path
from datetime import datetime, timezone


def _safe(value):
    """Format values safely for readable Markdown."""
    if isinstance(value, float):
        if not math.isfinite(value):
            return "Not available"
        return f"{value:.6g}"
    if value is None:
        return "Not available"
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, indent=2, default=str, allow_nan=False)
    return str(value)


def generate_markdown_report(experiment, output_path=None):
    """Generate a report from an experiment dictionary.

    Accepts the dictionary returned by ExperimentStore.get_experiment().
    Returns the report text, and optionally writes it to output_path.
    """
    if not isinstance(experiment, dict):
        raise ValueError("experiment must be a dictionary.")

    name = experiment.get("name", "Unnamed experiment")
    task = experiment.get("task", "Not specified")
    algorithm = experiment.get("algorithm")
    dataset = experiment.get("dataset_name")
    metrics = experiment.get("metrics", {})
    parameters = experiment.get("parameters", {})

    if not isinstance(metrics, dict):
        raise ValueError("Experiment metrics must be a dictionary.")
    if not isinstance(parameters, dict):
        raise ValueError("Experiment parameters must be a dictionary.")

    lines = [
        f"# AutoML Experiment Report: {name}",
        "",
        f"- **Experiment ID:** {_safe(experiment.get('id'))}",
        f"- **Created at:** {_safe(experiment.get('created_at'))}",
        f"- **Task:** {_safe(task)}",
        f"- **Algorithm:** {_safe(algorithm)}",
        f"- **Dataset:** {_safe(dataset)}",
        f"- **Rows:** {_safe(experiment.get('rows_count'))}",
        f"- **Features:** {_safe(experiment.get('features_count'))}",
        "",
        "## Evaluation metrics",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]

    if metrics:
        for key, value in metrics.items():
            lines.append(f"| {key} | {_safe(value)} |")
    else:
        lines.append("| No metrics recorded | Not available |")

    lines.extend([
        "",
        "## Model parameters",
        "",
        "```json",
        json.dumps(parameters, indent=2, default=str, allow_nan=False),
        "```",
        "",
        "## Notes",
        "",
        _safe(experiment.get("notes")),
        "",
        "## Model artifact",
        "",
        _safe(experiment.get("model_path")),
        "",
        "## Interpretation",
        "",
        "This report records the supplied experiment metadata and metrics. "
        "It does not independently validate the dataset, establish causation, "
        "or guarantee performance on unseen data.",
        "",
    ])

    report = "\n".join(lines)

    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(report, encoding="utf-8")

    return report
