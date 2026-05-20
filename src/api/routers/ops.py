"""Operational endpoints: /, /health and /version."""
from __future__ import annotations

import os

from fastapi import APIRouter, Depends
from fastapi.responses import HTMLResponse

from src.api.dependencies import require_auth
from src.api.model_loader import get_bundle
from src.api.schemas import HealthResponse, VersionResponse

API_VERSION = "1.0.0"

router = APIRouter(tags=["ops"])

_WELCOME_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>FEM Surrogate API</title>
  <style>
    body {{ font-family: system-ui, sans-serif; max-width: 860px; margin: 48px auto; padding: 0 24px; color: #1a1a2e; }}
    h1   {{ font-size: 1.8rem; margin-bottom: 4px; }}
    h2   {{ font-size: 1.1rem; margin-top: 36px; color: #16213e; border-bottom: 2px solid #e2e8f0; padding-bottom: 6px; }}
    p    {{ line-height: 1.6; color: #444; }}
    a    {{ color: #3b82f6; text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
    .badge {{ display: inline-block; background: #e0f2fe; color: #0369a1; border-radius: 6px; padding: 2px 10px; font-size: 0.82rem; margin-left: 8px; }}
    .links {{ display: flex; gap: 16px; flex-wrap: wrap; margin-top: 12px; }}
    .links a {{ background: #3b82f6; color: #fff; padding: 8px 18px; border-radius: 8px; font-weight: 600; }}
    .links a.secondary {{ background: #f1f5f9; color: #334155; }}
    pre  {{ background: #1e293b; color: #e2e8f0; padding: 20px; border-radius: 10px; overflow-x: auto; font-size: 0.88rem; line-height: 1.5; }}
    table {{ border-collapse: collapse; width: 100%; margin-top: 8px; }}
    th   {{ background: #f1f5f9; text-align: left; padding: 8px 12px; font-size: 0.85rem; }}
    td   {{ padding: 7px 12px; border-top: 1px solid #e2e8f0; font-size: 0.88rem; }}
    code {{ background: #f1f5f9; padding: 1px 6px; border-radius: 4px; font-size: 0.88rem; }}
    .note {{ background: #fefce8; border-left: 4px solid #facc15; padding: 10px 16px; border-radius: 4px; font-size: 0.9rem; margin-top: 12px; }}
  </style>
</head>
<body>

<h1>FEM Surrogate API <span class="badge">v{version}</span></h1>
<p>
  Surrogate ML model for <strong>finite-element traction-plate simulations</strong>.<br>
  Predicts <code>max_displacement_m</code> and <code>max_von_mises_pa</code> from plate
  geometry and loading conditions — in milliseconds instead of minutes.
</p>

<div class="links">
  <a href="/docs">Interactive docs (Swagger)</a>
  <a href="/redoc" class="secondary">ReDoc</a>
  <a href="/health" class="secondary">Health</a>
  <a href="/version" class="secondary">Version</a>
</div>

<h2>Supported geometries</h2>
<table>
  <tr><th>geometry_type</th><th>Description</th><th>Extra fields required</th></tr>
  <tr><td><code>with_hole</code></td><td>Plate with centred circular hole</td><td><code>hole_radius_ratio</code></td></tr>
  <tr><td><code>without_hole</code></td><td>Solid plate</td><td>—</td></tr>
  <tr><td><code>with_hole_moving</code></td><td>Plate with off-centre hole</td><td><code>hole_radius_ratio</code>, <code>hole_cx_ratio</code>, <code>hole_cy_ratio</code></td></tr>
</table>

<h2>Quick example — with_hole</h2>
<p>Copy and run this <code>curl</code> command directly:</p>
<pre>curl -s -u $API_USERNAME:$API_PASSWORD \\
  -X POST http://localhost:8000/predict \\
  -H "Content-Type: application/json" \\
  -d '{{
    "geometry_type":      "with_hole",
    "length_m":           0.2,
    "height_m":           0.05,
    "young_modulus_pa":   210000000000,
    "poisson_ratio":      0.3,
    "traction_pa":        50000000,
    "material_category":  "steel",
    "dimension_category": "small",
    "hole_radius_ratio":  0.2
  }}'</pre>

<h2>Quick example — without_hole</h2>
<pre>curl -s -u $API_USERNAME:$API_PASSWORD \\
  -X POST http://localhost:8000/predict \\
  -H "Content-Type: application/json" \\
  -d '{{
    "geometry_type":      "without_hole",
    "length_m":           0.3,
    "height_m":           0.08,
    "young_modulus_pa":   70000000000,
    "poisson_ratio":      0.33,
    "traction_pa":        20000000,
    "material_category":  "aluminum",
    "dimension_category": "medium"
  }}'</pre>

<h2>Quick example — with_hole_moving</h2>
<pre>curl -s -u $API_USERNAME:$API_PASSWORD \\
  -X POST http://localhost:8000/predict \\
  -H "Content-Type: application/json" \\
  -d '{{
    "geometry_type":      "with_hole_moving",
    "length_m":           0.25,
    "height_m":           0.06,
    "young_modulus_pa":   116000000000,
    "poisson_ratio":      0.32,
    "traction_pa":        35000000,
    "material_category":  "titanium",
    "dimension_category": "small",
    "hole_radius_ratio":  0.15,
    "hole_cx_ratio":      0.4,
    "hole_cy_ratio":      0.5
  }}'</pre>

<h2>Authentication</h2>
<p>All endpoints except <code>/health</code> require <strong>HTTP Basic Auth</strong>.
Credentials are set via <code>API_USERNAME</code> / <code>API_PASSWORD</code> environment variables.</p>

<div class="note">
  Model status: <strong>{model_status}</strong>{model_version_line}
</div>

</body>
</html>
"""


@router.get("/", response_class=HTMLResponse, include_in_schema=False)
async def welcome() -> HTMLResponse:
    bundle = get_bundle()
    model_status = "loaded" if bundle else "not loaded — check /health"
    model_version_line = f" &nbsp;|&nbsp; version <code>{bundle.version}</code>" if bundle else ""
    return HTMLResponse(_WELCOME_HTML.format(
        version=API_VERSION,
        model_status=model_status,
        model_version_line=model_version_line,
    ))


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Availability probe — no auth required."""
    bundle = get_bundle()
    return HealthResponse(
        status="ok" if bundle is not None else "degraded",
        model_loaded=bundle is not None,
        model_version=bundle.version if bundle else None,
    )


@router.get("/version", response_model=VersionResponse)
async def version(_: None = Depends(require_auth)) -> VersionResponse:
    bundle = get_bundle()
    return VersionResponse(
        api_version=API_VERSION,
        model_version=bundle.version if bundle else None,
        model_name=os.environ.get("MODEL_NAME", "lgbm_surrogate"),
        feature_count=bundle.feature_count if bundle else 0,
    )
