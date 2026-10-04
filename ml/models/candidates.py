"""Candidate model families and their hyper-parameter search spaces."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:  # XGBoost is optional so the project still runs where it cannot be installed.
    from xgboost import XGBClassifier

    HAS_XGBOOST = True
except Exception:  # pragma: no cover - depends on environment
    XGBClassifier = None
    HAS_XGBOOST = False

DISPLAY_NAMES = {
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
    "hist_gradient_boosting": "Histogram Gradient Boosting",
    "xgboost": "XGBoost",
}


def default_model(name: str, seed: int) -> Any:
    if name == "logistic_regression":
        return Pipeline(
            [("scale", StandardScaler()), ("clf", LogisticRegression(max_iter=5000, random_state=seed))]
        )
    if name == "random_forest":
        return RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=seed)
    if name == "hist_gradient_boosting":
        return HistGradientBoostingClassifier(random_state=seed)
    if name == "xgboost":
        if not HAS_XGBOOST:
            raise RuntimeError("xgboost is not installed")
        return XGBClassifier(
            tree_method="hist", n_estimators=300, eval_metric="logloss", n_jobs=2, random_state=seed
        )
    raise KeyError(name)


def search_space(name: str, imbalance_ratio: float) -> dict[str, list]:
    if name == "logistic_regression":
        return {"clf__C": [0.01, 0.1, 1.0, 10.0, 100.0], "clf__class_weight": [None, "balanced"]}
    if name == "random_forest":
        return {
            "n_estimators": [300, 500],
            "max_depth": [None, 8, 12, 16],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt", 0.5, None],
            "class_weight": [None, "balanced_subsample"],
        }
    if name == "hist_gradient_boosting":
        return {
            "learning_rate": [0.03, 0.06, 0.1],
            "max_iter": [200, 400],
            "max_leaf_nodes": [15, 31, 63],
            "min_samples_leaf": [10, 20, 40],
            "l2_regularization": [0.0, 0.1, 1.0],
            "class_weight": [None, "balanced"],
        }
    if name == "xgboost":
        return {
            "n_estimators": [200, 400, 600],
            "max_depth": [3, 4, 6],
            "learning_rate": [0.03, 0.06, 0.1],
            "subsample": [0.8, 1.0],
            "colsample_bytree": [0.8, 1.0],
            "min_child_weight": [1, 3],
            "scale_pos_weight": [1.0, round(float(np.sqrt(imbalance_ratio)), 2), round(imbalance_ratio, 2)],
        }
    raise KeyError(name)


def available_families() -> list[str]:
    names = ["logistic_regression", "random_forest", "hist_gradient_boosting"]
    if HAS_XGBOOST:
        names.append("xgboost")
    return names
