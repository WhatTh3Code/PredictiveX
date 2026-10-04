"""Loads the trained bundle once and serves predictions (no training at request time)."""

from __future__ import annotations

import json
import logging
import threading
from functools import lru_cache
from pathlib import Path

from backend.app.schemas.machine import ThresholdOverride
from backend.app.settings import Settings
from ml import config
from ml.inference import ModelUnavailableError, Predictor
from ml.models.risk import RiskPolicy

log = logging.getLogger("predictivex")


class ModelService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._predictor: Predictor | None = None
        self._error: str | None = None
        self._lock = threading.Lock()

    def load(self) -> None:
        with self._lock:
            try:
                self._predictor = Predictor.load(self.settings.model_path)
                self._error = None
                log.info("model loaded from %s", self.settings.model_path)
            except ModelUnavailableError as exc:
                self._predictor = None
                self._error = str(exc)
                log.error("model unavailable: %s", exc)

    @property
    def ready(self) -> bool:
        return self._predictor is not None

    @property
    def error(self) -> str | None:
        return self._error

    @property
    def predictor(self) -> Predictor:
        if self._predictor is None:
            raise ModelUnavailableError(self._error or "Model not loaded")
        return self._predictor

    def policy(self, override: ThresholdOverride | None = None) -> RiskPolicy:
        base = self.predictor.policy
        if override is not None:
            return RiskPolicy(override.medium, override.high, method="request override")
        s = self.settings
        if s.medium_threshold is not None or s.high_threshold is not None:
            return RiskPolicy(
                s.medium_threshold if s.medium_threshold is not None else base.medium_threshold,
                s.high_threshold if s.high_threshold is not None else base.high_threshold,
                method="environment override",
            )
        return base


class ReportUnavailableError(RuntimeError):
    pass


@lru_cache(maxsize=16)
def _read(path: Path, mtime: float) -> dict:
    return json.loads(path.read_text())


def report(name: str) -> dict:
    path = config.REPORTS_DIR / name
    if not path.exists():
        raise ReportUnavailableError(f"{name} has not been generated yet. Run the training pipeline.")
    return _read(path, path.stat().st_mtime)
