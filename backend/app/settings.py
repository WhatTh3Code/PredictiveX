"""Runtime settings, read from environment variables (see .env.example)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from ml import config


def _float_or_none(name: str) -> float | None:
    raw = os.getenv(name, "").strip()
    return float(raw) if raw else None


def _load_dotenv() -> None:
    """Minimal .env loader (no extra dependency). Real environment variables win."""
    path = config.ROOT / ".env"
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


@dataclass(frozen=True)
class Settings:
    model_path: Path = field(default_factory=lambda: Path(os.getenv("PREDICTIVEX_MODEL_PATH", config.MODEL_BUNDLE_PATH)))
    cors_origins: list[str] = field(
        default_factory=lambda: [
            o.strip()
            for o in os.getenv("PREDICTIVEX_CORS_ORIGINS", "http://localhost:8000,http://127.0.0.1:8000").split(",")
            if o.strip()
        ]
    )
    medium_threshold: float | None = field(default_factory=lambda: _float_or_none("PREDICTIVEX_RISK_MEDIUM"))
    high_threshold: float | None = field(default_factory=lambda: _float_or_none("PREDICTIVEX_RISK_HIGH"))
    max_batch: int = field(default_factory=lambda: int(os.getenv("PREDICTIVEX_MAX_BATCH", "1000")))
    frontend_dir: Path = field(default_factory=lambda: config.ROOT / "frontend" / "dist")


def get_settings() -> Settings:
    _load_dotenv()
    return Settings()
