"""Unsupervised clustering and anomaly detection for AutoML-Agent-Lab."""

import time
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans, DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


def _prepare_features(df, feature_columns=None):
    if df.empty:
        raise ValueError("Dataset is empty.")

    if feature_columns is None:
        feature_columns = df.select_dtypes(include=np.number).columns.tolist()

    if not feature_columns:
        raise ValueError("No numeric feature columns found.")

    missing = [c for c in feature_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing features: {missing}")

    X = df[feature_columns].apply(pd.to_numeric, errors="coerce")

    if X.isna().all().any():
        raise ValueError("At least one feature contains no usable numeric values.")

    X = SimpleImputer(strategy="median").fit_transform(X)
    X = StandardScaler().fit_transform(X)

    if not np.isfinite(X).all():
        raise ValueError("Data contains invalid numeric values.")

    return X, list(feature_columns)


def run_clustering(
    df,
    algorithm="kmeans",
    n_clusters=3,
    feature_columns=None,
):
    """Cluster rows using KMeans or DBSCAN."""
    started = time.perf_counter()
    X, features = _prepare_features(df, feature_columns)

    if algorithm == "kmeans":
        if not 2 <= int(n_clusters) < len(df):
            raise ValueError("n_clusters must be at least 2 and less than row count.")

        model = KMeans(
            n_clusters=int(n_clusters),
            random_state=42,
            n_init=10,
        )
        labels = model.fit_predict(X)
        inertia = float(model.inertia_)

    elif algorithm == "dbscan":
        model = DBSCAN()
        labels = model.fit_predict(X)
        inertia = None

    else:
        raise ValueError("algorithm must be 'kmeans' or 'dbscan'.")

    cluster_ids = set(labels) - {-1}
    silhouette = None

    if len(cluster_ids) >= 2:
        non_noise = labels != -1
        try:
            silhouette = float(
                silhouette_score(X[non_noise], labels[non_noise])
            )
        except ValueError:
            pass

    return {
        "task": "clustering",
        "algorithm": algorithm,
        "rows": len(df),
        "feature_columns": features,
        "labels": labels.astype(int).tolist(),
        "metrics": {
            "clusters_found": len(cluster_ids),
            "noise_points": int((labels == -1).sum()),
            "silhouette_score": silhouette,
            "inertia": inertia,
        },
        "elapsed_seconds": round(time.perf_counter() - started, 4),
    }


def run_anomaly_detection(
    df,
    feature_columns=None,
    contamination=0.05,
):
    """Detect unusual observations using Isolation Forest.

    Label 1 means normal; label -1 means anomaly.
    """
    if not 0 < float(contamination) <= 0.5:
        raise ValueError("contamination must be greater than 0 and at most 0.5.")

    started = time.perf_counter()
    X, features = _prepare_features(df, feature_columns)

    model = IsolationForest(
        contamination=float(contamination),
        random_state=42,
        n_estimators=200,
    )

    labels = model.fit_predict(X)
    anomaly_mask = labels == -1

    return {
        "task": "anomaly_detection",
        "algorithm": "isolation_forest",
        "rows": len(df),
        "feature_columns": features,
        "labels": labels.astype(int).tolist(),
        "metrics": {
            "anomaly_count": int(anomaly_mask.sum()),
            "anomaly_percent": round(float(anomaly_mask.mean() * 100), 2),
            "contamination_requested": float(contamination),
        },
        "elapsed_seconds": round(time.perf_counter() - started, 4),
    }
