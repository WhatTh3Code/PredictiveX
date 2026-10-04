from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from backend.app.settings import Settings
from backend.main import create_app
from tests.conftest import VALID_MACHINE


# ---------------------------------------------------------------- happy path ---------------
def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "model_loaded": True, "detail": None}


def test_model_info(client):
    body = client.get("/model/info").json()
    assert body["model_family"]
    assert body["risk_policy"]["medium_threshold"] <= body["risk_policy"]["high_threshold"]
    assert body["features"]


def test_predict_valid_request(client):
    r = client.post("/predict", json={"machine": VALID_MACHINE})
    assert r.status_code == 200
    body = r.json()
    assert 0 <= body["failure_probability"] <= 1
    assert body["risk_level"] in {"LOW", "MEDIUM", "HIGH"}
    assert isinstance(body["predicted_failure"], bool)
    assert len(body["contributions"]) == 9
    assert body["explanation"]
    assert body["failure_mode"]["note"].startswith("Conditional")


def test_predict_threshold_override(client):
    lo = client.post("/predict", json={"machine": VALID_MACHINE, "thresholds": {"medium": 0.0001, "high": 0.0002}})
    assert lo.json()["risk_level"] == "HIGH" or lo.json()["failure_probability"] < 0.0001
    bad = client.post("/predict", json={"machine": VALID_MACHINE, "thresholds": {"medium": 0.7, "high": 0.2}})
    assert bad.status_code == 422


def test_batch(client):
    r = client.post("/predict/batch", json={"machines": [VALID_MACHINE, {**VALID_MACHINE, "tool_wear_min": 220}]})
    assert r.status_code == 200
    body = r.json()
    assert body["count"] == 2
    assert sum(body["risk_counts"].values()) == 2


def test_batch_empty_rejected(client):
    assert client.post("/predict/batch", json={"machines": []}).status_code == 422


@pytest.mark.parametrize("path", ["/metrics", "/model-comparison", "/feature-importance", "/dataset/summary",
                                  "/dataset/eda", "/failure-mode", "/demo-cases"])
def test_report_endpoints(client, path):
    r = client.get(path)
    assert r.status_code == 200
    assert isinstance(r.json(), dict)


# ---------------------------------------------------------------- bad input ----------------
def test_missing_parameter(client):
    m = {k: v for k, v in VALID_MACHINE.items() if k != "torque_nm"}
    r = client.post("/predict", json={"machine": m})
    assert r.status_code == 422
    assert r.json()["error"] == "validation_error"
    assert any(d["field"].endswith("torque_nm") for d in r.json()["detail"])


@pytest.mark.parametrize("field,value", [
    ("rotational_speed_rpm", -100),
    ("torque_nm", "heavy"),
    ("type", "X"),
    ("air_temp_k", 1000),
    ("tool_wear_min", None),
])
def test_invalid_parameter(client, field, value):
    r = client.post("/predict", json={"machine": {**VALID_MACHINE, field: value}})
    assert r.status_code == 422


def test_unknown_field_rejected(client):
    r = client.post("/predict", json={"machine": {**VALID_MACHINE, "machine_failure": 1}})
    assert r.status_code == 422  # a leaked label can never be smuggled in


def test_malformed_json(client):
    r = client.post("/predict", content=b'{"machine": {"type": "L",', headers={"Content-Type": "application/json"})
    assert r.status_code == 422
    assert "Traceback" not in r.text


def test_validate_input_endpoint(client):
    ok = client.post("/validate-input", json=VALID_MACHINE).json()
    assert ok == {"valid": True, "errors": [], "warnings": []}
    bad = client.post("/validate-input", json={**VALID_MACHINE, "torque_nm": -5}).json()
    assert bad["valid"] is False and bad["errors"][0]["field"] == "torque_nm"
    warn = client.post("/validate-input", json={**VALID_MACHINE, "tool_wear_min": 400}).json()
    assert warn["valid"] is True and warn["warnings"]


# ---------------------------------------------------------------- what-if ------------------
def test_simulate_valid_modification(client):
    base = {**VALID_MACHINE, "tool_wear_min": 220, "torque_nm": 62}
    r = client.post("/simulate", json={"base": base, "changes": {"tool_wear_min": 50}})
    assert r.status_code == 200
    body = r.json()
    assert body["label"].startswith("Model simulation")
    assert body["changed_parameters"] == [
        {"parameter": "tool_wear_min", "label": "Tool wear", "before": 220.0, "after": 50.0}
    ]
    assert body["probability_change"] == pytest.approx(
        body["after"]["failure_probability"] - body["before"]["failure_probability"], abs=1e-4)


def test_simulate_extreme_value_warns(client):
    r = client.post("/simulate", json={"base": VALID_MACHINE, "changes": {"torque_nm": 300}})
    assert r.status_code == 200
    assert r.json()["after"]["warnings"]


def test_simulate_invalid_value(client):
    r = client.post("/simulate", json={"base": VALID_MACHINE, "changes": {"torque_nm": -1}})
    assert r.status_code == 422


def test_simulate_requires_a_change(client):
    r = client.post("/simulate", json={"base": VALID_MACHINE, "changes": {}})
    assert r.status_code == 422


# ---------------------------------------------------------------- failure handling ---------
def test_model_unavailable_returns_503(tmp_path):
    app = create_app(Settings(model_path=tmp_path / "missing.joblib", frontend_dir=tmp_path / "none"))
    with TestClient(app) as c:
        assert c.get("/health").json()["model_loaded"] is False
        r = c.post("/predict", json={"machine": VALID_MACHINE})
        assert r.status_code == 503
        assert r.json()["error"] == "model_unavailable"


def test_internal_errors_hide_stack_traces(client, monkeypatch):
    from ml.inference import Predictor

    def boom(*a, **k):
        raise RuntimeError("secret internal detail")

    monkeypatch.setattr(Predictor, "predict", boom)
    with TestClient(client.app, raise_server_exceptions=False) as c:
        r = c.post("/predict", json={"machine": VALID_MACHINE})
    assert r.status_code == 500
    assert "secret" not in r.text and "Traceback" not in r.text


def test_cors_allows_configured_origin_only(client):
    ok = client.options("/predict", headers={"Origin": "http://localhost:8000", "Access-Control-Request-Method": "POST"})
    assert ok.headers.get("access-control-allow-origin") == "http://localhost:8000"
    bad = client.options("/predict", headers={"Origin": "http://evil.example", "Access-Control-Request-Method": "POST"})
    assert "access-control-allow-origin" not in bad.headers
