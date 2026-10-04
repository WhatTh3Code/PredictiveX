"""PredictiveX API + dashboard server.

    uvicorn backend.main:app --port 8000          (from the repository root)

Serves the JSON API and, if `frontend/dist` exists, the built dashboard at `/`.
"""

from __future__ import annotations

import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi import FastAPI, Request  # noqa: E402
from fastapi.exceptions import RequestValidationError  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import JSONResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from starlette.exceptions import HTTPException as StarletteHTTPException  # noqa: E402

from backend.app.routes.api import router  # noqa: E402
from backend.app.services.model_service import ModelService, ReportUnavailableError  # noqa: E402
from backend.app.settings import Settings, get_settings  # noqa: E402
from ml.inference import ModelUnavailableError  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("predictivex")


def create_app(settings: Settings | None = None, load_model: bool = True) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if load_model:
            app.state.model_service.load()
        yield

    app = FastAPI(
        title="PredictiveX API",
        description="Industrial machine failure prediction (ALGOTHON'26 ALG-DATA-02). "
        "Trained offline on the AI4I 2020 synthetic predictive-maintenance dataset.",
        version="1.0.0",
        lifespan=lifespan,
    )
    app.state.model_service = ModelService(settings)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )

    @app.exception_handler(RequestValidationError)
    async def validation_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
        errors = []
        for e in exc.errors():
            loc = [str(p) for p in e.get("loc", []) if p != "body"]
            errors.append({"field": ".".join(loc) or "body", "message": e.get("msg", "invalid"), "type": e.get("type")})
        return JSONResponse(status_code=422, content={"error": "validation_error", "detail": errors})

    @app.exception_handler(ModelUnavailableError)
    async def model_handler(_: Request, exc: ModelUnavailableError) -> JSONResponse:
        return JSONResponse(status_code=503, content={"error": "model_unavailable", "message": str(exc)})

    @app.exception_handler(ReportUnavailableError)
    async def report_handler(_: Request, exc: ReportUnavailableError) -> JSONResponse:
        return JSONResponse(status_code=503, content={"error": "report_unavailable", "message": str(exc)})

    @app.exception_handler(StarletteHTTPException)
    async def http_handler(_: Request, exc: StarletteHTTPException) -> JSONResponse:
        return JSONResponse(status_code=exc.status_code, content={"error": "http_error", "message": str(exc.detail)})

    @app.exception_handler(Exception)
    async def unhandled(_: Request, exc: Exception) -> JSONResponse:
        log.exception("unhandled error")  # full trace goes to the server log only
        return JSONResponse(status_code=500, content={"error": "internal_error", "message": "Unexpected server error"})

    app.include_router(router)

    if settings.frontend_dir.exists():
        app.mount("/", StaticFiles(directory=settings.frontend_dir, html=True), name="dashboard")
    return app


app = create_app()
