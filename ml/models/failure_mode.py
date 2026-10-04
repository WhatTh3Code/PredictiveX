"""Conditional failure-mode classifier: "if this machine fails, which mechanism is most likely?"

Scope decisions (documented in reports/failure_mode.json and the README):
* Classes: TWF, HDF, PWF, OSF. RNF (random failure) is excluded: by the dataset's own
  definition it occurs with 0.1% probability regardless of process parameters, so no
  feature can predict it.
* Training rows: failed records in the TRAINING split with exactly one of these modes.
  Multi-mode records are excluded from training and scored as correct on validation/test
  when the predicted mode is any of their true modes.
* Failures with no mode flag are excluded (no label).
* The model is conditional on failure. The API only reports it alongside the binary
  failure probability and labels it as conditional.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from ml import config
from ml.features.engineering import FeatureBuilder
from ml.preprocessing.data import feature_target

MODES = ["twf", "hdf", "pwf", "osf"]


def _rows(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    failed = df[df[config.TARGET] == 1]
    flags = failed[MODES]
    single = (flags.sum(axis=1) == 1) & (failed["rnf"] == 0)
    any_mode = flags.sum(axis=1) >= 1
    label = flags.idxmax(axis=1).where(single)
    sets = flags.apply(lambda r: {m for m in MODES if r[m] == 1}, axis=1)
    return failed[any_mode], label[any_mode], sets[any_mode]


def _candidates(seed: int) -> dict:
    return {
        "decision_tree": DecisionTreeClassifier(max_depth=5, class_weight="balanced", random_state=seed),
        "random_forest": RandomForestClassifier(
            n_estimators=400, min_samples_leaf=2, class_weight="balanced", random_state=seed, n_jobs=-1
        ),
        "logistic_regression": Pipeline(
            [("scale", StandardScaler()),
             ("clf", LogisticRegression(max_iter=5000, class_weight="balanced", C=1.0))]
        ),
    }


def _evaluate(model, fb: FeatureBuilder, df: pd.DataFrame) -> dict:
    rows, label, sets = _rows(df)
    X = fb.transform(feature_target(rows)[0])
    pred = model.predict(X)
    single = label.notna().to_numpy()
    y_single = label[single].to_numpy()
    p_single = pred[single]
    prec, rec, f1, sup = precision_recall_fscore_support(y_single, p_single, labels=MODES, zero_division=0)
    any_match = np.mean([p in s for p, s in zip(pred, sets, strict=True)])
    return {
        "records_scored": int(len(rows)),
        "single_mode_records": int(single.sum()),
        "multi_mode_records": int((~single).sum()),
        "macro_f1": round(float(f1_score(y_single, p_single, labels=MODES, average="macro", zero_division=0)), 4),
        "accuracy_single_mode": round(float(np.mean(y_single == p_single)), 4),
        "any_match_accuracy_all": round(float(any_match), 4),
        "per_class": {
            m: {"precision": round(float(p), 4), "recall": round(float(r), 4), "f1": round(float(f), 4), "support": int(s)}
            for m, p, r, f, s in zip(MODES, prec, rec, f1, sup, strict=True)
        },
        "confusion_matrix": {
            "labels": MODES,
            "matrix": confusion_matrix(y_single, p_single, labels=MODES).tolist(),
        },
    }


def train_failure_mode_model(
    train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, fb: FeatureBuilder, seed: int
) -> tuple[dict, dict]:
    rows, label, _ = _rows(train_df)
    keep = label.notna()
    X = fb.transform(feature_target(rows[keep])[0])
    y = label[keep].to_numpy()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    cv_scores = {}
    for name, mdl in _candidates(seed).items():
        pred = cross_val_predict(clone(mdl), X, y, cv=cv)
        cv_scores[name] = round(float(f1_score(y, pred, average="macro")), 4)
    best = max(cv_scores, key=cv_scores.get)
    model = clone(_candidates(seed)[best]).fit(X, y)
    val = _evaluate(model, fb, val_df)
    test = _evaluate(model, fb, test_df)
    report = {
        "scope": "Conditional on failure. Classes TWF/HDF/PWF/OSF; RNF excluded (random by construction).",
        "training_records": int(len(y)),
        "training_class_counts": {m: int((y == m).sum()) for m in MODES},
        "cv_macro_f1": cv_scores,
        "selected": best,
        "validation": val,
        "test": test,
    }
    bundle = {
        "model": model,
        "classes": list(model.classes_),
        "names": {m: config.FAILURE_MODE_NAMES[m] for m in MODES},
        "test_recall_per_class": {m: test["per_class"][m]["recall"] for m in MODES},
        "test_macro_f1": test["macro_f1"],
    }
    return bundle, report
