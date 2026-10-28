from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_absolute_error, mean_squared_error, r2_score, roc_auc_score
)
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler as ClusterScaler
from xgboost import XGBClassifier, XGBRegressor

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
    duplicate_rows = int(df.duplicated().sum())
    return {
        "file_path": str(path),
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "column_names": list(df.columns.astype(str)),
        "numeric_columns": list(numeric.columns.astype(str)),
        "categorical_columns": list(categorical.columns.astype(str)),
        "missing_values": missing,
        "missing_percent": round(missing / max(df.size, 1) * 100, 2),
        "duplicate_rows": duplicate_rows,
        "dtypes": {str(k): str(v) for k, v in df.dtypes.items()},
        "preview": df.head(8).fillna("").to_dict(orient="records"),
    }

def _preprocessor(X):
    numeric_cols = X.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = X.select_dtypes(exclude=np.number).columns.tolist()
    transformers = []
    if numeric_cols:
        transformers.append(("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), numeric_cols))
    if categorical_cols:
        transformers.append(("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), categorical_cols))
    return ColumnTransformer(transformers=transformers)

def _infer_task(y):
    if not pd.api.types.is_numeric_dtype(y):
        return "classification"
    return "classification" if y.nunique() <= 20 else "regression"

def train_models(path, target=None, task="auto"):
    df = load_dataframe(path).dropna(axis=1, how="all")
    if not target:
        candidates = [c for c in df.columns if str(c).lower() in
                      {"target", "label", "class", "y", "outcome", "churn", "loan_status"}]
        target = candidates[0] if candidates else df.columns[-1]
    if target not in df.columns:
        raise ValueError(f"Target '{target}' not found.")
    y = df[target]
    X = df.drop(columns=[target])
    task = _infer_task(y) if task == "auto" else task

    if task == "classification":
        if y.nunique() < 2:
            raise ValueError("Classification target must contain at least two classes.")
        le = LabelEncoder()
        y_encoded = le.fit_transform(y.astype(str))
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42,
            stratify=y_encoded if len(np.unique(y_encoded)) > 1 else None
        )
        models = {
            "Logistic Regression": LogisticRegression(max_iter=1500),
            "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
        }
        if len(np.unique(y_encoded)) == 2:
            models["XGBoost"] = XGBClassifier(
                n_estimators=200, max_depth=5, learning_rate=0.05,
                random_state=42, eval_metric="logloss"
            )
        metrics = []
        for name, model in models.items():
            pipe = Pipeline([("prep", _preprocessor(X)), ("model", model)])
            pipe.fit(X_train, y_train)
            pred = pipe.predict(X_test)
            row = {
                "model": name,
                "accuracy": round(accuracy_score(y_test, pred), 4),
                "precision": round(precision_score(y_test, pred, average="weighted", zero_division=0), 4),
                "recall": round(recall_score(y_test, pred, average="weighted", zero_division=0), 4),
                "f1": round(f1_score(y_test, pred, average="weighted", zero_division=0), 4),
            }
            if len(np.unique(y_encoded)) == 2 and hasattr(pipe, "predict_proba"):
                try:
                    row["roc_auc"] = round(roc_auc_score(y_test, pipe.predict_proba(X_test)[:, 1]), 4)
                except Exception:
                    row["roc_auc"] = None
            else:
                row["roc_auc"] = None
            metrics.append(row)
        best = max(metrics, key=lambda x: x["f1"])
    elif task == "regression":
        if not pd.api.types.is_numeric_dtype(y):
            raise ValueError("Regression target must be numeric.")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        models = {
            "Linear Regression": LinearRegression(),
            "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
            "XGBoost": XGBRegressor(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42),
        }
        metrics = []
        for name, model in models.items():
            pipe = Pipeline([("prep", _preprocessor(X)), ("model", model)])
            pipe.fit(X_train, y_train)
            pred = pipe.predict(X_test)
            rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
            metrics.append({
                "model": name,
                "mae": round(mean_absolute_error(y_test, pred), 4),
                "rmse": round(rmse, 4),
                "r2": round(r2_score(y_test, pred), 4),
            })
        best = max(metrics, key=lambda x: x["r2"])
    else:
        raise ValueError("Supported tasks: auto, classification, regression.")

    return {
        "dataset": Path(path).name,
        "target": str(target),
        "task": task,
        "rows": int(len(df)),
        "features": int(X.shape[1]),
        "models": metrics,
        "best_model": best,
    }
