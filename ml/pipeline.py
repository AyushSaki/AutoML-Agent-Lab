import json
from pathlib import Path

import joblib
from uuid import uuid4
import numpy as np
import pandas as pd

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    KFold,
    cross_validate,
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
    LabelEncoder,
)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor



MODEL_DIR = Path(__file__).resolve().parent.parent / "models"

def load_dataframe(path):
    suffix = Path(path).suffix.lower()

    if suffix == ".csv":
        return pd.read_csv(path)

    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)

    raise ValueError("Unsupported file format.")


def analyze_dataset(path):
    df = load_dataframe(path)
    numeric = df.select_dtypes(include=np.number)
    categorical = df.select_dtypes(exclude=np.number)
    missing = int(df.isna().sum().sum())

    return {
        "file_path": str(path),
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "column_names": list(df.columns.astype(str)),
        "numeric_columns": list(numeric.columns.astype(str)),
        "categorical_columns": list(categorical.columns.astype(str)),
        "missing_values": missing,
        "missing_percent": round(missing / max(df.size, 1) * 100, 2),
        "duplicate_rows": int(df.duplicated().sum()),
        "dtypes": {str(k): str(v) for k, v in df.dtypes.items()},
        "preview": df.head(8).fillna("").to_dict(orient="records"),
    }


def _preprocessor(X):
    numeric_cols = X.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = X.select_dtypes(exclude=np.number).columns.tolist()
    transformers = []

    if numeric_cols:
        transformers.append((
            "num",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]),
            numeric_cols,
        ))

    if categorical_cols:
        transformers.append((
            "cat",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]),
            categorical_cols,
        ))

    if not transformers:
        raise ValueError("No usable feature columns remain after selecting the target.")

    return ColumnTransformer(transformers=transformers)


def _infer_task(y):
    """Infer classification vs regression using the inspection module's rule."""
    y = pd.Series(y).dropna()

    if y.empty:
        raise ValueError("Cannot infer a task from an empty target.")

    if pd.api.types.is_bool_dtype(y) or not pd.api.types.is_numeric_dtype(y):
        return "classification"

    unique_count = int(y.nunique())
    threshold = min(12, max(2, int(len(y) * 0.1)))

    return "classification" if unique_count <= threshold else "regression"

def _dataset_warnings(df, X, y, removed_target_rows):
    warnings = []

    if len(df) < 100:
        warnings.append(
            f"Small dataset: only {len(df)} labelled rows are available. "
            "Evaluation scores may vary substantially with different samples."
        )

    if removed_target_rows:
        warnings.append(
            f"Removed {removed_target_rows} rows with missing target values. "
            "Missing feature values are imputed inside the training pipeline."
        )

    if X.shape[1] == 0:
        warnings.append("No input features remain after removing the target.")

    if df.duplicated().any():
        warnings.append(
            f"{int(df.duplicated().sum())} duplicate rows remain. "
            "Review duplicates to determine whether they are legitimate observations."
        )

    target_name = str(y.name).strip().lower().replace("_", " ")
    for column in X.columns:
        feature_name = str(column).strip().lower().replace("_", " ")
        if target_name and (
            feature_name == target_name
            or target_name in feature_name
        ):
            warnings.append(
                f"Potential target leakage: feature '{column}' resembles the target "
                f"'{y.name}'. Check whether it would be available at prediction time."
            )

    if y.nunique() < 2:
        warnings.append("The target contains fewer than two distinct values.")

    return warnings


def _classification_models(number_of_classes):
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1500),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            n_jobs=-1,
        ),
    }

    if number_of_classes == 2:
        models["XGBoost"] = XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            random_state=42,
            eval_metric="logloss",
        )

    return models


def _regression_models():
    return {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=200,
            random_state=42,
            n_jobs=-1,
        ),
        "XGBoost": XGBRegressor(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            random_state=42,
        ),
    }


def train_models(path, target=None, task="auto"):
    df = load_dataframe(path).dropna(axis=1, how="all").copy()

    if df.empty or len(df.columns) < 2:
        raise ValueError(
            "The dataset needs at least one usable feature column and a target column."
        )

    # Resolve target selection using the Autonomous Data Scientist detector.
    if not target:
        from ml.autonomous_scientist import inspect_dataset

        detection = inspect_dataset(path)
        target_info = detection.get("target") or {}

        if target_info.get("auto_selected") and target_info.get("column"):
            target = target_info["column"]
        else:
            candidates = detection.get("target_candidates") or []
            candidate_names = [
                str(item.get("column"))
                for item in candidates
                if isinstance(item, dict) and item.get("column") in df.columns
            ]
            suggestions = (
                ", ".join(candidate_names[:5])
                if candidate_names
                else "none identified confidently"
            )
            raise ValueError(
                "The AI could not confidently identify a prediction target. "
                "Please select the target column manually in the dashboard. "
                f"Possible candidates: {suggestions}."
            )

    if target not in df.columns:
        raise ValueError(f"Target '{target}' not found in the dataset.")

    # Do not invent missing target labels.
    df[target] = df[target].replace(r"^\s*$", np.nan, regex=True)
    original_rows = len(df)
    df = df.dropna(subset=[target]).copy()
    removed_target_rows = original_rows - len(df)

    if len(df) < 5:
        raise ValueError(
            "At least 5 rows with a valid target are required for training."
        )

    y = df[target]
    X = df.drop(columns=[target])

    if X.shape[1] == 0:
        raise ValueError("No input features are available.")

    if task == "auto":
        task = _infer_task(y)

    if task not in {"classification", "regression"}:
        raise ValueError(
            "Supported tasks are auto, classification, and regression."
        )

    warnings = _dataset_warnings(df, X, y, removed_target_rows)

    if task == "classification":
        y = y.astype(str)
        class_counts = y.value_counts()

        if len(class_counts) < 2:
            raise ValueError(
                "Classification requires at least two distinct target classes."
            )

        if class_counts.min() < 2:
            raise ValueError(
                "Each target class needs at least two rows. Add examples for "
                "rare classes before training."
            )

        if class_counts.min() < 5:
            warnings.append(
                "At least one class has fewer than 5 examples. "
                "Cross-validation results may be unstable."
            )

        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)

        # Leave enough samples for each class in both train and test sets.
        number_of_classes = len(label_encoder.classes_)
        test_count = max(
            int(np.ceil(len(df) * 0.2)),
            number_of_classes,
        )

        if len(df) - test_count < number_of_classes:
            raise ValueError(
                "There are too few rows for a stratified train/test split. "
                "Add more labelled examples."
            )

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y_encoded,
            test_size=test_count,
            random_state=42,
            stratify=y_encoded,
        )

        models = _classification_models(number_of_classes)
        cv_splits = min(5, int(class_counts.min()))
        cv = StratifiedKFold(
            n_splits=cv_splits,
            shuffle=True,
            random_state=42,
        )

        metrics = []
        fitted_pipelines = {}

        for name, model in models.items():
            pipe = Pipeline([
                ("prep", _preprocessor(X_train)),
                ("model", model),
            ])

            # Cross-validation is performed only on the training partition.
            cv_result = cross_validate(
                pipe,
                X_train,
                y_train,
                cv=cv,
                scoring={
                    "accuracy": "accuracy",
                    "f1_weighted": "f1_weighted",
                },
                error_score="raise",
                n_jobs=1,
            )

            pipe.fit(X_train, y_train)
            pred = pipe.predict(X_test)
            fitted_pipelines[name] = pipe

            row = {
                "model": name,
                "accuracy": round(float(accuracy_score(y_test, pred)), 4),
                "precision": round(float(precision_score(
                    y_test, pred, average="weighted", zero_division=0
                )), 4),
                "recall": round(float(recall_score(
                    y_test, pred, average="weighted", zero_division=0
                )), 4),
                "f1": round(float(f1_score(
                    y_test, pred, average="weighted", zero_division=0
                )), 4),
                "roc_auc": None,
                "cv_folds": cv_splits,
                "cv_accuracy_mean": round(
                    float(np.mean(cv_result["test_accuracy"])), 4
                ),
                "cv_accuracy_std": round(
                    float(np.std(cv_result["test_accuracy"])), 4
                ),
                "cv_f1_mean": round(
                    float(np.mean(cv_result["test_f1_weighted"])), 4
                ),
                "cv_f1_std": round(
                    float(np.std(cv_result["test_f1_weighted"])), 4
                ),
            }

            if number_of_classes == 2 and hasattr(pipe, "predict_proba"):
                try:
                    row["roc_auc"] = round(float(roc_auc_score(
                        y_test,
                        pipe.predict_proba(X_test)[:, 1],
                    )), 4)
                except (ValueError, IndexError):
                    row["roc_auc"] = None

            metrics.append(row)

        best = max(metrics, key=lambda item: item["cv_f1_mean"])
        best_pipe = fitted_pipelines[best["model"]]
        best_predictions = best_pipe.predict(X_test)

        cm = confusion_matrix(
            y_test,
            best_predictions,
            labels=np.arange(number_of_classes),
        )

        report = classification_report(
            y_test,
            best_predictions,
            labels=np.arange(number_of_classes),
            target_names=[str(label) for label in label_encoder.classes_],
            output_dict=True,
            zero_division=0,
        )

        evaluation = {
            "test_rows": int(len(y_test)),
            "train_rows": int(len(y_train)),
            "confusion_matrix": {
                "labels": [str(label) for label in label_encoder.classes_],
                "matrix": cm.tolist(),
            },
            "classification_report": report,
        }

    else:
        if not pd.api.types.is_numeric_dtype(y):
            raise ValueError(
                "Regression requires a numeric target. Select classification "
                "if the target represents categories."
            )

        y = pd.to_numeric(y, errors="coerce")
        valid_target = y.notna()
        removed_again = int((~valid_target).sum())

        if removed_again:
            X = X.loc[valid_target].copy()
            y = y.loc[valid_target].copy()
            warnings.append(
                f"Removed {removed_again} additional rows with non-numeric "
                "regression target values."
            )

        if len(y) < 5:
            raise ValueError(
                "At least 5 rows with numeric target values are required."
            )

        test_count = max(1, int(np.ceil(len(y) * 0.2)))
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_count,
            random_state=42,
        )

        cv_splits = min(5, len(X_train))
        if cv_splits < 2:
            raise ValueError(
                "Not enough training rows for cross-validation."
            )

        cv = KFold(
            n_splits=cv_splits,
            shuffle=True,
            random_state=42,
        )

        models = _regression_models()
        metrics = []
        fitted_pipelines = {}

        for name, model in models.items():
            pipe = Pipeline([
                ("prep", _preprocessor(X_train)),
                ("model", model),
            ])

            cv_result = cross_validate(
                pipe,
                X_train,
                y_train,
                cv=cv,
                scoring={
                    "mae": "neg_mean_absolute_error",
                    "r2": "r2",
                },
                error_score="raise",
                n_jobs=1,
            )

            pipe.fit(X_train, y_train)
            pred = pipe.predict(X_test)
            fitted_pipelines[name] = pipe

            metrics.append({
                "model": name,
                "mae": round(float(mean_absolute_error(y_test, pred)), 4),
                "rmse": round(float(np.sqrt(
                    mean_squared_error(y_test, pred)
                )), 4),
                "r2": round(float(r2_score(y_test, pred)), 4),
                "cv_folds": cv_splits,
                "cv_mae_mean": round(float(
                    -np.mean(cv_result["test_mae"])
                ), 4),
                "cv_mae_std": round(float(
                    np.std(cv_result["test_mae"])
                ), 4),
                "cv_r2_mean": round(float(
                    np.mean(cv_result["test_r2"])
                ), 4),
                "cv_r2_std": round(float(
                    np.std(cv_result["test_r2"])
                ), 4),
            })

        # Lower cross-validated MAE is better.
        best = min(metrics, key=lambda item: item["cv_mae_mean"])
        best_pipe = fitted_pipelines[best["model"]]
        evaluation = {
            "train_rows": int(len(y_train)),
            "test_rows": int(len(y_test)),
        }

    if len(df) < 30:
        warnings.append(
            "The held-out test set is very small. Treat test metrics as "
            "preliminary, not as proof of real-world performance."
        )

    print(
        f"Training complete: {Path(path).name}; "
        f"task={task}; rows={len(df)}; "
        f"selected by cross-validation."
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_id = uuid4().hex
    model_file = MODEL_DIR / f"{model_id}.joblib"
    metadata_file = MODEL_DIR / f"{model_id}.json"

    joblib.dump(best_pipe, model_file)

    metadata = {
        "model_id": model_id,
        "model": best["model"],
        "dataset": Path(path).name,
        "target": str(target),
        "task": task,
        "features": [str(column) for column in X.columns],
        "classes": (
            [str(label) for label in label_encoder.classes_]
            if task == "classification"
            else None
        ),
    }
    metadata_file.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    return {
        "dataset": Path(path).name,
        "target": str(target),
        "task": task,
        "rows": int(len(df)),
        "features": int(X.shape[1]),
        "models": metrics,
        "best_model": best,
        "saved_model": {
            "model_id": model_id,
            "filename": model_file.name,
            "metadata_filename": metadata_file.name,
        },
        "warnings": warnings,
        "evaluation": evaluation,
    }


def predict_with_saved_model(model_id, path):
    """Load a saved model and predict from a CSV or Excel dataset."""
    import re

    if not re.fullmatch(r"[a-f0-9]{32}", str(model_id)):
        raise ValueError("Invalid model ID.")

    model_path = MODEL_DIR / f"{model_id}.joblib"
    metadata_path = MODEL_DIR / f"{model_id}.json"

    if not model_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError("Saved model was not found.")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    required_features = metadata.get("features", [])

    if not required_features:
        raise ValueError("The saved model has no feature metadata.")

    df = load_dataframe(path)

    if df.empty:
        raise ValueError("The prediction dataset is empty.")

    missing = [column for column in required_features if column not in df.columns]
    if missing:
        raise ValueError(
            "Prediction dataset is missing required columns: "
            + ", ".join(missing)
        )

    # Preserve the feature order used during training. Extra columns are ignored.
    X = df.loc[:, required_features].copy()
    model = joblib.load(model_path)
    raw_predictions = model.predict(X)

    if metadata.get("task") == "classification":
        classes = metadata.get("classes") or []
        predictions = []
        for value in raw_predictions:
            index = int(value)
            if index < 0 or index >= len(classes):
                raise ValueError("A predicted class is outside the saved class list.")
            predictions.append(classes[index])
    else:
        predictions = [
            value.item() if isinstance(value, np.generic) else value
            for value in raw_predictions
        ]

    return {
        "model_id": model_id,
        "model": metadata.get("model", "Saved model"),
        "task": metadata.get("task", "unknown"),
        "target": metadata.get("target", "prediction"),
        "rows": len(df),
        "predictions": predictions,
        "preview": X.head(10).fillna("").to_dict(orient="records"),
    }
