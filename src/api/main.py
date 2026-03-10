"""FEM Surrogate Model — FastAPI serving layer.

Endpoints
---------
GET  /health       Liveness + model readiness probe
GET  /version      API and model version info
POST /predict      Single-case surrogate inference
GET  /metrics      Prometheus metrics (via prometheus-fastapi-instrumentator)

Configuration (environment variables)
--------------------------------------
MODEL_DIR          Path to model version directory (overrides registry lookup)
REGISTRY_DIR       Path to model registry root (default: artifacts/models)
MODEL_NAME         Model name in registry (default: lgbm_surrogate)
API_LOG_LEVEL      Logging level (default: INFO)

Run locally
-----------
    uvicorn src.api.main:app --reload --port 8000
"""
from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.api.model_loader import ModelBundle, get_bundle, load_models
from src.api.schemas import (
    HealthResponse,
    PredictionRequest,
    PredictionResponse,
    PredictionResult,
    VersionResponse,
)
from src.processing.build_features import engineer_features
from src.training.train_advanced import CAT_COLS

_EPS = 1e-12
API_VERSION = "1.0.0"

logging.basicConfig(
    level=os.environ.get("API_LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


# ── Lifespan (startup / shutdown) ─────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading surrogate models ...")
    model_dir = os.environ.get("MODEL_DIR")
    try:
        load_models(model_dir)
        logger.info("Models loaded successfully.")
    except Exception as exc:
        logger.error("Model loading failed: %s", exc)
        # Do NOT raise — let the app start in degraded mode so /health reports the problem
    yield
    logger.info("Shutting down.")


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="FEM Surrogate API",
    description=(
        "Surrogate ML model for FEM traction-plate simulations. "
        "Predicts max_displacement_m and max_von_mises_pa from plate geometry and loading."
    ),
    version=API_VERSION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Prometheus instrumentation
try:
    from prometheus_fastapi_instrumentator import Instrumentator
    Instrumentator().instrument(app).expose(app, endpoint="/metrics")
    logger.info("Prometheus instrumentation enabled at /metrics")
except ImportError:
    logger.warning(
        "prometheus-fastapi-instrumentator not installed — /metrics endpoint unavailable. "
        "Install it with: pip install prometheus-fastapi-instrumentator"
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _infer(bundle: ModelBundle, request: PredictionRequest) -> dict[str, float]:
    """Run inference for all loaded targets. Returns {target: value_in_original_space}."""
    raw = request.model_dump()

    # Build a single-row DataFrame and auto-engineer features
    case_df = pd.DataFrame([raw])
    case_df = engineer_features(case_df, require_targets=False)

    results: dict[str, float] = {}
    for target, lm in bundle.models.items():
        cat_present = [c for c in CAT_COLS if c in lm.feature_cols]
        df = case_df.copy()
        if lm.encoder is not None and cat_present:
            df[cat_present] = lm.encoder.transform(df[cat_present].astype(str))

        X = df[lm.feature_cols].astype(float)
        y_pred_log = lm.model.predict(X)
        results[target] = float(10.0 ** y_pred_log[0])

    return results


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["ops"])
async def health() -> HealthResponse:
    bundle = get_bundle()
    return HealthResponse(
        status="ok" if bundle is not None else "degraded",
        model_loaded=bundle is not None,
        model_version=bundle.version if bundle else None,
    )


@app.get("/version", response_model=VersionResponse, tags=["ops"])
async def version() -> VersionResponse:
    bundle = get_bundle()
    return VersionResponse(
        api_version=API_VERSION,
        model_version=bundle.version if bundle else None,
        model_name=os.environ.get("MODEL_NAME", "lgbm_surrogate"),
        feature_count=bundle.feature_count if bundle else 0,
    )


@app.post("/predict", response_model=PredictionResponse, tags=["inference"])
async def predict(request: PredictionRequest) -> PredictionResponse:
    bundle = get_bundle()
    if bundle is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Check /health for details.",
        )

    t0 = time.perf_counter()
    try:
        preds = _infer(bundle, request)
    except KeyError as exc:
        raise HTTPException(status_code=422, detail=f"Missing feature: {exc}") from exc
    except Exception as exc:
        logger.exception("Inference error: %s", exc)
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc

    elapsed_ms = (time.perf_counter() - t0) * 1000
    logger.info(
        "predict | geometry=%s | disp=%.3e | vm=%.3e | %.1f ms",
        request.geometry_type,
        preds.get("max_displacement_m", float("nan")),
        preds.get("max_von_mises_pa", float("nan")),
        elapsed_ms,
    )

    return PredictionResponse(
        model_version=bundle.version,
        predictions=PredictionResult(
            max_displacement_m=preds.get("max_displacement_m", float("nan")),
            max_von_mises_pa=preds.get("max_von_mises_pa", float("nan")),
        ),
        input_summary={
            "geometry_type":   request.geometry_type,
            "length_m":        request.length_m,
            "height_m":        request.height_m,
            "young_modulus_pa": request.young_modulus_pa,
            "traction_pa":     request.traction_pa,
            "hole_radius_ratio": request.hole_radius_ratio,
        },
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled exception: %s", exc)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})
