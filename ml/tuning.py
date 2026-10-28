"""Hyperparameter tuning for classification and regression."""

from sklearn.base import clone, is_classifier
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    KFold,
    StratifiedKFold,
)


def tune_model(
    estimator,
    X,
    y,
    param_grid,
    search_type="grid",
    scoring=None,
    cv=3,
    n_iter=10,
    random_state=42,
    n_jobs=-1,
):
    """Tune an sklearn estimator without changing the supplied estimator.

    Returns the fitted search object and a JSON-friendly result dictionary.
    """
    if len(X) != len(y):
        raise ValueError("X and y must contain the same number of rows.")

    if len(X) < 2:
        raise ValueError("At least two rows are required for tuning.")

    if not param_grid:
        raise ValueError("param_grid cannot be empty.")

    if not isinstance(cv, int) or cv < 2:
        raise ValueError("cv must be an integer of at least 2.")

    classifier = is_classifier(estimator)

    if classifier:
        counts = y.value_counts() if hasattr(y, "value_counts") else None
        if counts is None:
            import pandas as pd
            counts = pd.Series(y).value_counts()

        smallest_class = int(counts.min())
        splits = min(cv, smallest_class)

        if splits < 2:
            raise ValueError(
                "Each classification class needs at least two samples."
            )

        splitter = StratifiedKFold(
            n_splits=splits, shuffle=True, random_state=random_state
        )
        if scoring is None:
            scoring = "f1_weighted"
    else:
        splits = min(cv, len(X))
        if splits < 2:
            raise ValueError("At least two CV splits are required.")

        splitter = KFold(
            n_splits=splits, shuffle=True, random_state=random_state
        )
        if scoring is None:
            scoring = "neg_mean_absolute_error"

    if search_type == "grid":
        search = GridSearchCV(
            estimator=clone(estimator),
            param_grid=param_grid,
            scoring=scoring,
            cv=splitter,
            n_jobs=n_jobs,
            refit=True,
            error_score="raise",
        )
    elif search_type == "random":
        search = RandomizedSearchCV(
            estimator=clone(estimator),
            param_distributions=param_grid,
            n_iter=max(1, int(n_iter)),
            scoring=scoring,
            cv=splitter,
            n_jobs=n_jobs,
            random_state=random_state,
            refit=True,
            error_score="raise",
        )
    else:
        raise ValueError("search_type must be 'grid' or 'random'.")

    search.fit(X, y)

    return search, {
        "task": "classification" if classifier else "regression",
        "search_type": search_type,
        "scoring": scoring,
        "cv_splits": splits,
        "best_params": search.best_params_,
        "best_score": float(search.best_score_),
        "candidates_tested": len(search.cv_results_["params"]),
    }
