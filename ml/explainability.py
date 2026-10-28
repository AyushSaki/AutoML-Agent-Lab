"""Model-agnostic feature importance using permutation importance."""

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.utils.multiclass import type_of_target


def explain_model(
    estimator,
    X,
    y,
    scoring=None,
    n_repeats=5,
    random_state=42,
    n_jobs=-1,
):
    """Estimate how much model performance changes when each feature is shuffled.

    The estimator must already be fitted. X should contain the same feature
    columns and preprocessing expected by that estimator.
    """
    if len(X) != len(y):
        raise ValueError("X and y must contain the same number of rows.")

    if len(X) < 2:
        raise ValueError("At least two rows are required.")

    if n_repeats < 1:
        raise ValueError("n_repeats must be at least 1.")

    if hasattr(X, "columns"):
        names = [str(c) for c in X.columns]
    else:
        names = [f"feature_{i}" for i in range(np.asarray(X).shape[1])]

    if scoring is None:
        target_type = type_of_target(y)
        scoring = (
            "f1_weighted"
            if target_type in ("binary", "multiclass")
            else "neg_mean_absolute_error"
        )

    result = permutation_importance(
        estimator,
        X,
        y,
        scoring=scoring,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=n_jobs,
    )

    table = pd.DataFrame({
        "feature": names,
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std,
    })
    table = table.sort_values(
        "importance_mean", ascending=False
    ).reset_index(drop=True)

    return {
        "scoring": scoring,
        "n_repeats": n_repeats,
        "features": table.to_dict(orient="records"),
        "warning": (
            "Importance measures changes in the selected score when a "
            "feature is shuffled. It does not prove causation; correlated "
            "features can share or obscure importance."
        ),
    }
