"""POST /predict — surrogate inference router.

Example call (curl)
-------------------
    curl -s -u admin:mdp123 -X POST http://localhost:8000/predict \
         -H "Content-Type: application/json" \
         -d '{
               "geometry_type": "with_hole",
               "length_m": 0.2,
               "height_m": 0.05,
               "young_modulus_pa": 210000000000,
               "poisson_ratio": 0.3,
               "traction_pa": 50000000,
               "material_category": "steel",
               "dimension_category": "small",
               "hole_radius_ratio": 0.2
             }'

Example call (Python / requests)
---------------------------------
    import requests

    payload = {
        "geometry_type": "with_hole",
        "length_m": 0.2,
        "height_m": 0.05,
        "young_modulus_pa": 210_000_000_000,
        "poisson_ratio": 0.3,
        "traction_pa": 50_000_000,
        "material_category": "steel",
        "dimension_category": "small",
        "hole_radius_ratio": 0.2,
    }
    resp = requests.post(
        "http://localhost:8000/predict",
        json=payload,
        auth=("admin", "mdp123"),
    )
    resp.raise_for_status()
    print(resp.json())
    # {
    #   "model_version": "v20260519_...",
    #   "predictions": {
    #     "max_displacement_m": 1.23e-05,
    #     "max_von_mises_pa":   4.56e+07
    #   },
    #   "input_summary": { ... }
    # }
"""
from __future__ import annotations

import logging
import time

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException

from src.api.dependencies import require_auth
from src.api.model_loader import get_bundle
from src.api.schemas import CAT_COLS, PredictionRequest, PredictionResponse, PredictionResult
from src.processing.build_features import engineer_features

logger = logging.getLogger(__name__)

router = APIRouter(tags=["inference"])


# ── Inference helper ──────────────────────────────────────────────────────────

def _infer(request: PredictionRequest) -> dict[str, float]:
    """Run the surrogate model for all loaded targets.

    Returns a dict mapping target name → predicted value (original scale).
    """
    bundle = get_bundle()
    if bundle is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Check /health for details.",
        )

    raw = request.model_dump()
    case_df = pd.DataFrame([raw])
    case_df["simulation_id"] = "api-inference"
    case_df["timestamp"] = pd.Timestamp.utcnow()
    case_df = engineer_features(case_df, require_targets=False)

    results: dict[str, float] = {}
    for target, lm in bundle.models.items():
        cat_present = [c for c in CAT_COLS if c in lm.feature_cols]
        df = case_df.copy()
        if lm.encoder is not None and cat_present:
            df[cat_present] = lm.encoder.transform(df[cat_present].astype(str))
        X = df[lm.feature_cols].astype(float)
        y_log = lm.model.predict(X)
        results[target] = float(10.0 ** y_log[0])

    return results


# ── Endpoint ──────────────────────────────────────────────────────────────────

@router.post("/predict", response_model=PredictionResponse)
async def predict(
    request: PredictionRequest,
    _: None = Depends(require_auth),
) -> PredictionResponse:
    """Run the FEM surrogate model on a single plate geometry.

    Returns predicted ``max_displacement_m`` and ``max_von_mises_pa``.
    """
    bundle = get_bundle()
    t0 = time.perf_counter()
    try:
        preds = _infer(request)
    except HTTPException:
        raise
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
            "geometry_type":    request.geometry_type,
            "length_m":         request.length_m,
            "height_m":         request.height_m,
            "young_modulus_pa": request.young_modulus_pa,
            "traction_pa":      request.traction_pa,
            "hole_radius_ratio": request.hole_radius_ratio,
        },
    )
