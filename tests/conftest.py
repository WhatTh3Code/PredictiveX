from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ml import config  # noqa: E402

VALID_MACHINE = {
    "type": "L",
    "air_temp_k": 300.1,
    "process_temp_k": 310.2,
    "rotational_speed_rpm": 1500,
    "torque_nm": 40.0,
    "tool_wear_min": 100,
}

requires_model = pytest.mark.skipif(
    not config.MODEL_BUNDLE_PATH.exists(), reason="model bundle not trained (run scripts/train_models.py)"
)


@pytest.fixture(scope="session")
def predictor():
    from ml.inference import Predictor

    if not config.MODEL_BUNDLE_PATH.exists():
        pytest.skip("model bundle not trained")
    return Predictor.load()


@pytest.fixture(scope="session")
def client():
    from fastapi.testclient import TestClient

    from backend.main import create_app

    if not config.MODEL_BUNDLE_PATH.exists():
        pytest.skip("model bundle not trained")
    with TestClient(create_app()) as c:
        yield c
