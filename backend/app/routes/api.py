"""HTTP endpoints. All responses are structured JSON; errors are handled in backend/main.py."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Body, HTTPException, Request
from pydantic import ValidationError

from backend.app.schemas.machine import (
    BatchPredictRequest,
    MachineInput,
    PredictRequest,
    SimulateRequest,
)
from backend.app.services.model_service import ModelService, report

router = APIRouter()


def service(request: Request) -> ModelService:
    return request.app.state.model_service


# ---------------------------------------------------------------- system -------------------
@router.get("/health", tags=["system"])
def health(request: Request) -> dict:
    svc = service(request)
    return {
        "status": "ok" if svc.ready else "degraded",
        "model_loaded": svc.ready,
        "detail": None if svc.ready else "Model unavailable; prediction endpoints return 503.",
    }


@router.get("/model/info", tags=["model"])
def model_info(request: Request) -> dict:
    pred = service(request).predictor
    policy = service(request).policy()
    fm = pred.bundle.get("failure_mode") or {}
    return {
        **pred.metadata,
        "risk_policy": policy.to_dict(),
        "training_ranges": pred.bundle["training_ranges"],
        "explainer": {
            "method": pred.bundle["explainer"].description,
            "output_space": pred.bundle["explainer"].output_space,
        },
        "failure_mode_model": {
            "classes": fm.get("classes"),
            "test_macro_f1": fm.get("test_macro_f1"),
            "test_recall_per_class": fm.get("test_recall_per_class"),
        } if fm else None,
    }


# ---------------------------------------------------------------- reports ------------------
@router.get("/dataset/summary", tags=["data"])
def dataset_summary() -> dict:
    return report("data_quality.json")


@router.get("/dataset/eda", tags=["data"])
def dataset_eda() -> dict:
    return report("eda_summary.json")


@router.get("/metrics", tags=["model"])
def metrics() -> dict:
    return report("metrics.json")


@router.get("/model-comparison", tags=["model"])
def model_comparison() -> dict:
    return {"comparison": report("model_comparison.json"), "feature_ablation": report("feature_ablation.json")}


@router.get("/feature-importance", tags=["model"])
def feature_importance() -> dict:
    return report("explainability.json")


@router.get("/failure-mode", tags=["model"])
def failure_mode() -> dict:
    return report("failure_mode.json")


@router.get("/demo-cases", tags=["demo"])
def demo_cases() -> dict:
    return report("demo_cases.json")


# ---------------------------------------------------------------- prediction ---------------
@router.post("/validate-input", tags=["prediction"])
def validate_input(request: Request, payload: Any = Body(...)) -> dict:
    """Validate a machine record without predicting. Always 200; lists errors and warnings."""
    try:
        machine = MachineInput.model_validate(payload)
    except ValidationError as exc:
        return {
            "valid": False,
            "errors": [
                {"field": ".".join(str(p) for p in e["loc"]) or "body", "message": e["msg"]} for e in exc.errors()
            ],
            "warnings": [],
        }
    svc = service(request)
    warnings = svc.predictor.range_warnings(machine.model_dump()) if svc.ready else []
    return {"valid": True, "errors": [], "warnings": warnings}


@router.post("/predict", tags=["prediction"])
def predict(request: Request, body: PredictRequest) -> dict:
    svc = service(request)
    policy = svc.policy(body.thresholds)
    result = svc.predictor.predict([body.machine.model_dump()], policy=policy)[0]
    return {"input": body.machine.model_dump(), **result}


@router.post("/predict/batch", tags=["prediction"])
def predict_batch(request: Request, body: BatchPredictRequest) -> dict:
    svc = service(request)
    if len(body.machines) > svc.settings.max_batch:
        raise HTTPException(status_code=422, detail=f"Batch size exceeds limit of {svc.settings.max_batch}")
    policy = svc.policy(body.thresholds)
    records = [m.model_dump() for m in body.machines]
    results = svc.predictor.predict(records, policy=policy, explain=body.explain)
    counts = {lvl: sum(r["risk_level"] == lvl for r in results) for lvl in ("LOW", "MEDIUM", "HIGH")}
    return {
        "count": len(results),
        "risk_counts": counts,
        "results": [{"input": rec, **res} for rec, res in zip(records, results, strict=True)],
    }


@router.post("/simulate", tags=["prediction"])
def simulate(request: Request, body: SimulateRequest) -> dict:
    svc = service(request)
    policy = svc.policy(body.thresholds)
    changes = body.changes.model_dump(exclude_none=True)
    return svc.predictor.simulate(body.base.model_dump(), changes, policy=policy)
