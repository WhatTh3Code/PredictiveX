"""TreeSHAP correctness: brute-force Shapley values and XGBoost's native implementation."""

from __future__ import annotations

import itertools
from math import factorial

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier

from ml.explainability import shap_values as SV


def _cond_exp(t: SV.Tree, x: np.ndarray, S: set, j: int = 0) -> float:
    if t.feature[j] < 0:
        return t.value[j]
    f = t.feature[j]
    if f in S:
        go_left = x[f] < t.threshold[j] if t.strict_less else x[f] <= t.threshold[j]
        return _cond_exp(t, x, S, t.left[j] if go_left else t.right[j])
    lft, rgt = t.left[j], t.right[j]
    return (t.cover[lft] * _cond_exp(t, x, S, lft) + t.cover[rgt] * _cond_exp(t, x, S, rgt)) / t.cover[j]


def _brute_force(t: SV.Tree, x: np.ndarray, m: int) -> np.ndarray:
    phi = np.zeros(m)
    for i in range(m):
        others = [k for k in range(m) if k != i]
        for s in range(m):
            for S in itertools.combinations(others, s):
                w = factorial(s) * factorial(m - s - 1) / factorial(m)
                phi[i] += w * (_cond_exp(t, x, set(S) | {i}) - _cond_exp(t, x, set(S)))
    return phi


@pytest.fixture(scope="module")
def toy():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(400, 4)), columns=list("abcd"))
    y = ((X.a * X.b + X.c) > 0.3).astype(int)
    return X, y


def test_tree_shap_equals_brute_force_shapley(toy):
    X, y = toy
    rf = RandomForestClassifier(n_estimators=4, max_depth=6, random_state=0).fit(X, y)
    trees = SV.trees_from_random_forest(rf)
    for r in range(8):
        x = X.iloc[r].to_numpy()
        for t in trees:
            assert np.allclose(SV.tree_shap_single(t, x, 4), _brute_force(t, x, 4), atol=1e-12)


def test_tree_shap_matches_xgboost_native(toy):
    xgb = pytest.importorskip("xgboost")
    X, y = toy
    m = xgb.XGBClassifier(n_estimators=30, max_depth=4).fit(X, y)
    booster = m.get_booster()
    native = booster.predict(xgb.DMatrix(X.iloc[:30]), pred_contribs=True)[:, :-1]
    trees = SV.trees_from_xgboost(booster, 4)
    mine = np.array([sum(SV.tree_shap_single(t, X.iloc[i].to_numpy(), 4) for t in trees) for i in range(30)])
    assert np.abs(mine - native).max() < 1e-4  # float32 precision in XGBoost


@pytest.mark.parametrize("model", [
    RandomForestClassifier(n_estimators=10, max_depth=5, random_state=0),
    HistGradientBoostingClassifier(max_iter=30, random_state=0),
])
def test_local_accuracy(toy, model):
    X, y = toy
    model.fit(X, y)
    exp = SV.make_explainer(model, X)
    sv = exp.shap_values(X.iloc[:20])
    assert np.allclose(exp.expected_value + sv.sum(axis=1), exp.margin(X.iloc[:20]), atol=1e-9)


def test_feature_never_used_gets_zero(toy):
    X, y = toy
    rf = RandomForestClassifier(n_estimators=5, max_depth=4, random_state=0).fit(X[["a", "b"]].assign(c=0.0), y)
    exp = SV.make_explainer(rf, X[["a", "b"]].assign(c=0.0))
    sv = exp.shap_values(X[["a", "b"]].assign(c=0.0).iloc[:10])
    assert np.allclose(sv[:, 2], 0.0)
