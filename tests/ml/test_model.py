"""Model bundle tests, including a check that reported test metrics are reproducible."""

from __future__ import annotations

import json

import numpy as np
import pytest
from sklearn.metrics import average_precision_score, roc_auc_score

from ml import config
from ml.inference import ModelUnavailableError, Predictor
from ml.models.risk import RiskPolicy
from ml.preprocessing.data import feature_target, load_split
from tests.conftest import VALID_MACHINE


def test_bundle_contains_full_inference_pipeline(predictor):
    for key in ("model", "feature_builder", "risk_policy", "explainer", "metadata", "training_ranges"):
        assert key in predictor.bundle


def test_feature_consistency_between_builder_and_model(predictor):
    model = predictor.bundle["model"]
    assert model.n_features_in_ == len(predictor.feature_names)
    if hasattr(model, "feature_names_in_"):
        assert list(model.feature_names_in_) == predictor.feature_names


def test_reported_test_metrics_are_reproducible(predictor):
    """Every headline number must be regenerable from the saved model and the test split."""
    reported = json.loads((config.REPORTS_DIR / "metrics.json").read_text())["test"]
    X, y = feature_target(load_split("test"))
    p = predictor.bundle["model"].predict_proba(predictor.bundle["feature_builder"].transform(X))[:, 1]
    assert average_precision_score(y, p) == pytest.approx(reported["pr_auc"], abs=1e-4)
    assert roc_auc_score(y, p) == pytest.approx(reported["roc_auc"], abs=1e-4)
    pred = p >= predictor.policy.high_threshold
    cm = reported["confusion_matrix"]
    assert int(((pred == 1) & (y == 1)).sum()) == cm["tp"]
    assert int(((pred == 1) & (y == 0)).sum()) == cm["fp"]
    assert int(((pred == 0) & (y == 1)).sum()) == cm["fn"]


def test_predictions_are_deterministic(predictor):
    a = predictor.predict([VALID_MACHINE])[0]
    b = predictor.predict([VALID_MACHINE])[0]
    assert a["failure_probability"] == b["failure_probability"]
    assert a["contributions"] == b["contributions"]


def test_unseen_record_prediction(predictor):
    # A record that is not in the dataset (values chosen inside the training ranges).
    rec = {**VALID_MACHINE, "air_temp_k": 299.37, "rotational_speed_rpm": 1433, "torque_nm": 47.3}
    res = predictor.predict([rec])[0]
    assert 0.0 <= res["failure_probability"] <= 1.0
    assert res["risk_level"] in {"LOW", "MEDIUM", "HIGH"}
    assert res["warnings"] == []


def test_probability_responds_to_inputs(predictor):
    """High wear x high torque (overstrain region) must score above a nominal machine."""
    nominal = predictor.predict([VALID_MACHINE])[0]["failure_probability"]
    stressed = predictor.predict([{**VALID_MACHINE, "tool_wear_min": 230, "torque_nm": 65}])[0]["failure_probability"]
    assert stressed > nominal


def test_shap_local_accuracy_on_bundle(predictor):
    X, _ = feature_target(load_split("test").head(5))
    Xf = predictor.bundle["feature_builder"].transform(X)
    exp = predictor.bundle["explainer"]
    sv = exp.shap_values(Xf)
    assert np.allclose(exp.expected_value + sv.sum(axis=1), exp.margin(Xf), atol=1e-6)


def test_explanation_uses_non_causal_language(predictor):
    text = predictor.predict([VALID_MACHINE])[0]["explanation"].lower()
    assert "cause" not in text.replace("not proof of cause", "")
    assert "contribut" in text


def test_out_of_range_input_warns(predictor):
    res = predictor.predict([{**VALID_MACHINE, "torque_nm": 120.0}])[0]
    assert any("outside the training range" in w for w in res["warnings"])


def test_failure_mode_is_conditional_distribution(predictor):
    fm = predictor.predict([VALID_MACHINE])[0]["failure_mode"]
    assert fm is not None
    assert sum(p["probability"] for p in fm["probabilities"]) == pytest.approx(1.0, abs=1e-3)
    assert "rnf" not in {p["mode"] for p in fm["probabilities"]}


def test_risk_policy_validation():
    with pytest.raises(ValueError):
        RiskPolicy(0.6, 0.4)
    p = RiskPolicy(0.1, 0.5)
    assert (p.classify(0.05), p.classify(0.1), p.classify(0.5)) == ("LOW", "MEDIUM", "HIGH")


def test_thresholds_were_derived_not_hand_picked(predictor):
    pol = predictor.policy
    assert "out-of-fold" in pol.method
    assert 0 < pol.medium_threshold <= pol.high_threshold < 1


def test_missing_bundle_raises(tmp_path):
    with pytest.raises(ModelUnavailableError):
        Predictor.load(tmp_path / "nope.joblib")


def test_corrupt_bundle_raises(tmp_path):
    bad = tmp_path / "bad.joblib"
    bad.write_bytes(b"not a model")
    with pytest.raises(ModelUnavailableError):
        Predictor.load(bad)
