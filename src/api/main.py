"""FastAPI application factory for the FEM Surrogate API.

Endpoints (mounted via routers)
--------------------------------
GET  /health       Availability probe + model status   (no auth)
GET  /version      API and model version info           (auth required)
POST /predict      Surrogate inference on a single case (auth required)
GET  /metrics      Prometheus metrics (if installed)

Configuration (environment variables)
--------------------------------------
MODEL_DIR          Path to the model version directory (overrides registry)
REGISTRY_DIR       Path to the registry root (default: artifacts/models)
MODEL_NAME         Model name in the registry (default: lgbm_surrogate)
API_LOG_LEVEL      Logging level (default: INFO)
API_USERNAME       Basic-auth username (default: admin)
API_PASSWORD       Basic-auth password (default: mdp123)

Local execution
---------------
    uvicorn src.api.main:app --reload --port 8000
"""
from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.api.model_loader import load_models
from src.api.routers import ops, predict

logging.basicConfig(
    level=os.environ.get("API_LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


# Lifecycle

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading surrogate models ...")
    model_dir = os.environ.get("MODEL_DIR")
    try:
        load_models(model_dir)
        logger.info("Models loaded successfully.")
    except Exception as exc:
        # Start in degraded mode — /health will report the problem
        logger.error("Model loading failed: %s", exc)
    yield
    logger.info("Shutting down.")


# Application factory

app = FastAPI(
    title="FEM Surrogate API",
    description=(
        "Surrogate ML model for FEM traction-plate simulations. "
        "Predicts max_displacement_m and max_von_mises_pa from plate geometry and loading."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8501",   # Streamlit dashboard
        "http://localhost:8000",   # API self (Swagger UI)
        "http://dashboard:8501",   # Docker internal
    ],
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "Content-Type"],
)

# Routers

app.include_router(ops.router)
app.include_router(predict.router)

# Optional Prometheus instrumentation

try:
    from prometheus_fastapi_instrumentator import Instrumentator
    Instrumentator().instrument(app).expose(app, endpoint="/metrics")
    logger.info("Prometheus instrumentation enabled at /metrics")
except ImportError:
    logger.warning(
        "prometheus-fastapi-instrumentator not installed — /metrics unavailable. "
        "Install with: pip install prometheus-fastapi-instrumentator"
    )


# Global error handler

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled exception: %s", exc)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})
