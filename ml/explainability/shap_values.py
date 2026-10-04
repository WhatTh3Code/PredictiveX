"""SHAP explanations for every candidate model family.

The `shap` package could not be installed in the build environment, so values are computed
with exact algorithms instead of approximations:

* XGBoost              - XGBoost's built-in TreeSHAP (`pred_contribs=True`), the same
                         algorithm `shap.TreeExplainer` calls for XGBoost models.
                         Output space: log-odds.
* Random forest /      - `tree_shap_single`, a direct implementation of path-dependent
  hist. gradient boost   TreeSHAP (Lundberg et al., 2018, Algorithm 2). Verified in
                         tests/ml/test_shap.py against XGBoost's native implementation and
                         against brute-force Shapley values.
* Logistic regression  - exact linear SHAP: coef_j * (x_j - mean_j) in log-odds space
                         (assumes feature independence, as `shap.LinearExplainer` does).

Every explainer satisfies local accuracy: expected_value + sum(shap) == model output
(margin), which the training script checks and reports.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------------------------
# Generic tree representation + TreeSHAP
# --------------------------------------------------------------------------------------------


@dataclass
class Tree:
    left: np.ndarray  # child index or -1
    right: np.ndarray
    feature: np.ndarray  # -1 for leaves
    threshold: np.ndarray
    value: np.ndarray  # leaf output (in the explained output space)
    cover: np.ndarray  # training weight reaching each node
    strict_less: bool = False  # XGBoost: go left if x < t; sklearn: x <= t
    float32_inputs: bool = False  # XGBoost and sklearn forests route on float32-cast inputs

    def expected_value(self) -> float:
        leaves = self.feature < 0
        return float(np.sum(self.value[leaves] * self.cover[leaves]) / self.cover[0])


@dataclass
class _Path:
    d: list[int] = field(default_factory=list)
    z: list[float] = field(default_factory=list)
    o: list[float] = field(default_factory=list)
    w: list[float] = field(default_factory=list)

    def copy(self) -> _Path:
        return _Path(self.d.copy(), self.z.copy(), self.o.copy(), self.w.copy())


def _extend(m: _Path, pz: float, po: float, pi: int) -> _Path:
    m = m.copy()
    ln = len(m.d)
    m.d.append(pi)
    m.z.append(pz)
    m.o.append(po)
    m.w.append(1.0 if ln == 0 else 0.0)
    for i in range(ln - 1, -1, -1):
        m.w[i + 1] += po * m.w[i] * (i + 1) / (ln + 1)
        m.w[i] = pz * m.w[i] * (ln - i) / (ln + 1)
    return m


def _unwind(m: _Path, i: int) -> _Path:
    m = m.copy()
    ln = len(m.d) - 1
    n = m.w[ln]
    o, z = m.o[i], m.z[i]
    for j in range(ln - 1, -1, -1):
        if o != 0:
            t = m.w[j]
            m.w[j] = n * (ln + 1) / ((j + 1) * o)
            n = t - m.w[j] * z * (ln - j) / (ln + 1)
        else:
            m.w[j] = m.w[j] * (ln + 1) / (z * (ln - j))
    for j in range(i, ln):
        m.d[j], m.z[j], m.o[j] = m.d[j + 1], m.z[j + 1], m.o[j + 1]
    m.d.pop()
    m.z.pop()
    m.o.pop()
    m.w.pop()
    return m


def _unwound_sum(m: _Path, i: int) -> float:
    ln = len(m.d) - 1
    o, z = m.o[i], m.z[i]
    total = 0.0
    if o != 0:
        n = m.w[ln]
        for j in range(ln - 1, -1, -1):
            t = n / ((j + 1) * o)
            total += t
            n = m.w[j] - t * z * (ln - j)
    else:
        for j in range(ln - 1, -1, -1):
            total += m.w[j] / (z * (ln - j))
    return total * (ln + 1)


def _as_lists(tree: Tree) -> tuple:
    """Plain Python lists are ~5x faster than numpy scalar indexing inside the recursion."""
    cached = getattr(tree, "_lists", None)
    if cached is None:
        cached = (
            tree.left.tolist(), tree.right.tolist(), tree.feature.tolist(),
            (tree.threshold.astype(np.float32) if tree.strict_less else tree.threshold).astype(float).tolist(),
            tree.value.tolist(), tree.cover.tolist(),
        )
        object.__setattr__(tree, "_lists", cached)
    return cached


def tree_shap_single(tree: Tree, x: np.ndarray, n_features: int) -> np.ndarray:
    """Exact path-dependent TreeSHAP for one row and one tree."""
    left, right, feature, threshold, value, cover = _as_lists(tree)
    xs = (x.astype(np.float32) if tree.float32_inputs else x).astype(float).tolist()
    strict = tree.strict_less
    phi = [0.0] * n_features

    def recurse(j: int, m: _Path, pz: float, po: float, pi: int) -> None:
        m = _extend(m, pz, po, pi)
        f = feature[j]
        if f < 0:
            v = value[j]
            for i in range(1, len(m.d)):
                phi[m.d[i]] += _unwound_sum(m, i) * (m.o[i] - m.z[i]) * v
            return
        xv = xs[f]
        go_left = xv < threshold[j] if strict else xv <= threshold[j]
        hot, cold = (left[j], right[j]) if go_left else (right[j], left[j])
        iz, io = 1.0, 1.0
        k = next((i for i in range(1, len(m.d)) if m.d[i] == f), None)
        if k is not None:
            iz, io = m.z[k], m.o[k]
            m = _unwind(m, k)
        rj = cover[j]
        recurse(hot, m, iz * cover[hot] / rj, io, f)
        recurse(cold, m, iz * cover[cold] / rj, 0.0, f)

    recurse(0, _Path(), 1.0, 1.0, -1)
    return np.array(phi)


# --------------------------------------------------------------------------------------------
# Converters
# --------------------------------------------------------------------------------------------


def trees_from_random_forest(model) -> list[Tree]:
    trees = []
    n = len(model.estimators_)
    for est in model.estimators_:
        t = est.tree_
        # sklearn >= 1.4 stores class fractions in `value`; the forest averages them.
        frac = t.value[:, 0, :] / t.value[:, 0, :].sum(axis=1, keepdims=True)
        trees.append(
            Tree(
                left=t.children_left.astype(int),
                right=t.children_right.astype(int),
                feature=np.where(t.children_left < 0, -1, t.feature).astype(int),
                threshold=t.threshold.astype(float),
                value=frac[:, 1] / n,
                cover=t.weighted_n_node_samples.astype(float),
                float32_inputs=True,
            )
        )
    return trees


def trees_from_hist_gb(model) -> list[Tree]:
    trees = []
    for stage in model._predictors:  # one predictor per iteration for binary classification
        nodes = stage[0].nodes
        leaf = nodes["is_leaf"].astype(bool)
        trees.append(
            Tree(
                left=np.where(leaf, -1, nodes["left"].astype(np.int64)),
                right=np.where(leaf, -1, nodes["right"].astype(np.int64)),
                feature=np.where(leaf, -1, nodes["feature_idx"].astype(np.int64)),
                threshold=nodes["num_threshold"].astype(float),
                value=nodes["value"].astype(float),
                cover=nodes["count"].astype(float),
            )
        )
    return trees


def trees_from_xgboost(booster, n_features: int) -> list[Tree]:
    """Used by tests to cross-check the Python TreeSHAP against XGBoost's C++ implementation."""
    df = booster.trees_to_dataframe()
    names = booster.feature_names or [f"f{i}" for i in range(n_features)]
    trees = []
    for _, g in df.groupby("Tree", sort=True):
        g = g.sort_values("Node")
        idx = {nid: i for i, nid in enumerate(g["ID"])}
        is_leaf = g["Feature"].to_numpy() == "Leaf"
        trees.append(
            Tree(
                left=np.array([-1 if lf else idx[y] for lf, y in zip(is_leaf, g["Yes"], strict=True)]),
                right=np.array([-1 if lf else idx[n] for lf, n in zip(is_leaf, g["No"], strict=True)]),
                feature=np.array([-1 if lf else names.index(f) for lf, f in zip(is_leaf, g["Feature"], strict=True)]),
                threshold=np.nan_to_num(g["Split"].to_numpy(dtype=float)),
                value=np.where(is_leaf, g["Gain"].to_numpy(dtype=float), 0.0),
                cover=g["Cover"].to_numpy(dtype=float),
                strict_less=True,
                float32_inputs=True,
            )
        )
    return trees


# --------------------------------------------------------------------------------------------
# Explainers (picklable; stored inside the model bundle)
# --------------------------------------------------------------------------------------------


class XGBoostExplainer:
    description = "Exact TreeSHAP via XGBoost native pred_contribs (log-odds space)"
    output_space = "log-odds"

    def __init__(self, model) -> None:
        self.model = model
        self.expected_value: float = 0.0  # set from the bias column on first call

    def _contribs(self, X: pd.DataFrame) -> np.ndarray:
        import xgboost as xgb

        booster = self.model.get_booster()
        return booster.predict(xgb.DMatrix(X, feature_names=list(X.columns)), pred_contribs=True)

    def shap_values(self, X: pd.DataFrame) -> np.ndarray:
        c = self._contribs(X)
        self.expected_value = float(c[0, -1])
        return c[:, :-1]

    def margin(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X, output_margin=True)


class TreeEnsembleExplainer:
    def __init__(self, model, trees: list[Tree], base: float, output_space: str, description: str) -> None:
        self.model = model
        self.trees = trees
        self.expected_value = base + sum(t.expected_value() for t in trees)
        self.output_space = output_space
        self.description = description

    def shap_values(self, X: pd.DataFrame) -> np.ndarray:
        A = X.to_numpy(dtype=float)
        out = np.zeros_like(A)
        for r in range(len(A)):
            for t in self.trees:
                out[r] += tree_shap_single(t, A[r], A.shape[1])
        return out

    def margin(self, X: pd.DataFrame) -> np.ndarray:
        if self.output_space == "probability":
            return self.model.predict_proba(X)[:, 1]
        return self.model.decision_function(X)


class LinearExplainer:
    description = "Exact linear SHAP for logistic regression, coef*(x-mean) in log-odds space"
    output_space = "log-odds"

    def __init__(self, pipeline, X_background: pd.DataFrame) -> None:
        self.model = pipeline
        self.scaler = pipeline.named_steps["scale"]
        self.clf = pipeline.named_steps["clf"]
        Z = self.scaler.transform(X_background)
        self.mean_z = Z.mean(axis=0)
        self.expected_value = float(self.clf.intercept_[0] + self.clf.coef_[0] @ self.mean_z)

    def shap_values(self, X: pd.DataFrame) -> np.ndarray:
        Z = self.scaler.transform(X)
        return (Z - self.mean_z) * self.clf.coef_[0]

    def margin(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.decision_function(X)


def make_explainer(model, X_background: pd.DataFrame):
    name = type(model).__name__
    if name == "XGBClassifier":
        exp = XGBoostExplainer(model)
        exp.shap_values(X_background.iloc[:1])  # sets expected_value
        return exp
    if name == "RandomForestClassifier":
        return TreeEnsembleExplainer(
            model, trees_from_random_forest(model), 0.0, "probability",
            "Exact path-dependent TreeSHAP (Python implementation), probability space",
        )
    if name == "HistGradientBoostingClassifier":
        base = float(np.ravel(model._baseline_prediction)[0])
        return TreeEnsembleExplainer(
            model, trees_from_hist_gb(model), base, "log-odds",
            "Exact path-dependent TreeSHAP (Python implementation), log-odds space",
        )
    if name == "Pipeline":
        return LinearExplainer(model, X_background)
    raise TypeError(f"No SHAP explainer for {name}")
