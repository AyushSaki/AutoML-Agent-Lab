"""Persistent SQLite storage for AutoML experiments."""

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path


def _json_default(value):
    """Convert common NumPy/Pandas values to JSON-compatible values."""
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    return str(value)


class ExperimentStore:
    """Store experiment metadata and metrics in a local SQLite database."""

    def __init__(self, db_path="data/experiments.db"):
        self.db_path = Path(db_path)
        if str(self.db_path) != ":memory:":
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self):
        return sqlite3.connect(str(self.db_path))

    def _initialize(self):
        with self._connect() as connection:
            connection.execute("""
                CREATE TABLE IF NOT EXISTS experiments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    name TEXT NOT NULL,
                    task TEXT NOT NULL,
                    algorithm TEXT,
                    dataset_name TEXT,
                    rows_count INTEGER,
                    features_count INTEGER,
                    metrics_json TEXT NOT NULL,
                    parameters_json TEXT NOT NULL,
                    model_path TEXT,
                    notes TEXT
                )
            """)

    def log_experiment(
        self,
        name,
        task,
        metrics,
        algorithm=None,
        dataset_name=None,
        rows_count=None,
        features_count=None,
        parameters=None,
        model_path=None,
        notes=None,
    ):
        """Save an experiment and return its database ID."""
        if not str(name).strip():
            raise ValueError("Experiment name cannot be empty.")
        if not str(task).strip():
            raise ValueError("Experiment task cannot be empty.")
        if not isinstance(metrics, dict):
            raise ValueError("metrics must be a dictionary.")

        created_at = datetime.now(timezone.utc).isoformat()
        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO experiments (
                    created_at, name, task, algorithm, dataset_name,
                    rows_count, features_count, metrics_json,
                    parameters_json, model_path, notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    created_at,
                    str(name),
                    str(task),
                    algorithm,
                    dataset_name,
                    rows_count,
                    features_count,
                    json.dumps(metrics, default=_json_default, allow_nan=False),
                    json.dumps(parameters or {}, default=_json_default,
                               allow_nan=False),
                    str(model_path) if model_path is not None else None,
                    notes,
                ),
            )
            return cursor.lastrowid

    def list_experiments(self, limit=100, task=None):
        """Return newest experiments first, with decoded metrics and parameters."""
        if not isinstance(limit, int) or limit < 1:
            raise ValueError("limit must be a positive integer.")

        query = "SELECT * FROM experiments"
        params = []
        if task is not None:
            query += " WHERE task = ?"
            params.append(task)
        query += " ORDER BY id DESC LIMIT ?"
        params.append(limit)

        with self._connect() as connection:
            connection.row_factory = sqlite3.Row
            rows = connection.execute(query, params).fetchall()

        results = []
        for row in rows:
            item = dict(row)
            item["metrics"] = json.loads(item.pop("metrics_json"))
            item["parameters"] = json.loads(item.pop("parameters_json"))
            results.append(item)
        return results

    def get_experiment(self, experiment_id):
        """Return one experiment by ID, or None if it does not exist."""
        with self._connect() as connection:
            connection.row_factory = sqlite3.Row
            row = connection.execute(
                "SELECT * FROM experiments WHERE id = ?",
                (experiment_id,),
            ).fetchone()

        if row is None:
            return None

        item = dict(row)
        item["metrics"] = json.loads(item.pop("metrics_json"))
        item["parameters"] = json.loads(item.pop("parameters_json"))
        return item
