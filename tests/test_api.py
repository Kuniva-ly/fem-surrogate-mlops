"""Tests for the FastAPI application — schemas, auth, ops, predict endpoints."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

import src.api.main as _main
import src.api.model_loader as _ml
from src.api.model_loader import LoadedModel, ModelBundle

# ── Helpers ───────────────────────────────────────────────────────────────────

def _feature_cols() -> list[str]:
    """Compute numeric feature columns from a sample row (geometry=with_hole).

    Categorical columns are excluded because the mock bundle has encoder=None;
    the real pipeline encodes them with OrdinalEncoder before predict().
    """
    import pandas as pd
    from src.processing.build_features import _feature_columns, engineer_features

    row = {
        "simulation_id": "test", "timestamp": "2026-01-01T00:00:00Z",
        "material_category": "steel", "dimension_category": "small",
        "length_m": 0.2, "height_m": 0.05,
        "young_modulus_pa": 2.1e11, "poisson_ratio": 0.3,
        "traction_pa": 5e7, "mesh_nx": 120, "mesh_ny": 24,
        "geometry_type": "with_hole", "hole_radius_ratio": 0.2,
    }
    df = engineer_features(pd.DataFrame([row]), require_targets=False)
    all_cols = _feature_columns(df)
    return df[all_cols].select_dtypes(include="number").columns.tolist()


_FEAT_COLS = None  # computed once on first call


def _get_feat_cols() -> list[str]:
    global _FEAT_COLS
    if _FEAT_COLS is None:
        _FEAT_COLS = _feature_cols()
    return _FEAT_COLS


def _make_bundle(version: str = "v_test") -> ModelBundle:
    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([6.0])
    feat_cols = _get_feat_cols()
    lm = LoadedModel(
        target="max_displacement_m", model=mock_model,
        feature_cols=feat_cols, encoder=None, normalize_by=None,
    )
    lm_vm = LoadedModel(
        target="max_von_mises_pa", model=mock_model,
        feature_cols=feat_cols, encoder=None, normalize_by=None,
    )
    bundle = ModelBundle(version=version, model_dir=Path("/tmp"))
    bundle.models["max_displacement_m"] = lm
    bundle.models["max_von_mises_pa"] = lm_vm
    return bundle


# ── Fixtures ──────────────────────────────────────────────────────────────────

try:
    from fastapi.testclient import TestClient
    _FASTAPI_AVAILABLE = True
except ImportError:
    _FASTAPI_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not _FASTAPI_AVAILABLE, reason="fastapi/httpx not installed"
)

from src.api.main import app  # noqa: E402 — import after availability check

AUTH = ("testuser", "testpass")
BAD_AUTH = ("wrong", "bad")

PAYLOAD_HOLE = {
    "geometry_type": "with_hole",
    "length_m": 0.2, "height_m": 0.05,
    "young_modulus_pa": 2.1e11, "poisson_ratio": 0.3,
    "traction_pa": 5e7,
    "material_category": "steel", "dimension_category": "small",
    "hole_radius_ratio": 0.2,
}
PAYLOAD_NO_HOLE = {
    "geometry_type": "without_hole",
    "length_m": 0.3, "height_m": 0.08,
    "young_modulus_pa": 7e10, "poisson_ratio": 0.33,
    "traction_pa": 2e7,
    "material_category": "aluminum", "dimension_category": "medium",
}
PAYLOAD_MOVING = {
    "geometry_type": "with_hole_moving",
    "length_m": 0.25, "height_m": 0.06,
    "young_modulus_pa": 1.16e11, "poisson_ratio": 0.32,
    "traction_pa": 3.5e7,
    "material_category": "titanium", "dimension_category": "small",
    "hole_radius_ratio": 0.15, "hole_cx_ratio": 0.4, "hole_cy_ratio": 0.5,
}


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("API_USERNAME", "testuser")
    monkeypatch.setenv("API_PASSWORD", "testpass")
    bundle = _make_bundle()
    monkeypatch.setattr(_ml, "_bundle", bundle)
    monkeypatch.setattr(_main, "load_models", lambda *a, **kw: None)
    with TestClient(app) as c:
        yield c


@pytest.fixture
def client_no_model(monkeypatch):
    monkeypatch.setenv("API_USERNAME", "testuser")
    monkeypatch.setenv("API_PASSWORD", "testpass")
    monkeypatch.setattr(_ml, "_bundle", None)
    monkeypatch.setattr(_main, "load_models", lambda *a, **kw: None)
    with TestClient(app) as c:
        yield c


# ── /health ───────────────────────────────────────────────────────────────────

def test_health_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["model_loaded"] is True
    assert body["status"] == "ok"
    assert body["model_version"] == "v_test"


def test_health_degraded(client_no_model):
    r = client_no_model.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["model_loaded"] is False
    assert body["status"] == "degraded"
    assert body["model_version"] is None


# ── / (welcome page) ─────────────────────────────────────────────────────────

def test_welcome_page_returns_html(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    assert "FEM Surrogate API" in r.text


def test_welcome_page_no_model(client_no_model):
    r = client_no_model.get("/")
    assert r.status_code == 200


# ── /version ──────────────────────────────────────────────────────────────────

def test_version_no_auth_returns_401(client):
    r = client.get("/version")
    assert r.status_code == 401


def test_version_bad_auth_returns_401(client):
    r = client.get("/version", auth=BAD_AUTH)
    assert r.status_code == 401


def test_version_ok(client):
    r = client.get("/version", auth=AUTH)
    assert r.status_code == 200
    body = r.json()
    assert body["api_version"] == "1.0.0"
    assert body["model_version"] == "v_test"
    assert "feature_count" in body


def test_version_no_model_returns_503(client_no_model):
    r = client_no_model.get("/version", auth=AUTH)
    assert r.status_code == 503


# ── /predict ─────────────────────────────────────────────────────────────────

def test_predict_no_auth_returns_401(client):
    r = client.post("/predict", json=PAYLOAD_HOLE)
    assert r.status_code == 401


def test_predict_bad_auth_returns_401(client):
    r = client.post("/predict", json=PAYLOAD_HOLE, auth=BAD_AUTH)
    assert r.status_code == 401


def test_predict_with_hole(client):
    r = client.post("/predict", json=PAYLOAD_HOLE, auth=AUTH)
    assert r.status_code == 200
    body = r.json()
    assert "predictions" in body
    assert "max_displacement_m" in body["predictions"]
    assert "max_von_mises_pa" in body["predictions"]
    assert body["model_version"] == "v_test"


def test_predict_without_hole(client):
    r = client.post("/predict", json=PAYLOAD_NO_HOLE, auth=AUTH)
    assert r.status_code == 200


def test_predict_with_hole_moving(client):
    r = client.post("/predict", json=PAYLOAD_MOVING, auth=AUTH)
    assert r.status_code == 200


def test_predict_no_model_returns_503(client_no_model):
    r = client_no_model.post("/predict", json=PAYLOAD_HOLE, auth=AUTH)
    assert r.status_code == 503


# ── Schema validation ─────────────────────────────────────────────────────────

def test_predict_missing_length_returns_422(client):
    bad = {k: v for k, v in PAYLOAD_HOLE.items() if k != "length_m"}
    r = client.post("/predict", json=bad, auth=AUTH)
    assert r.status_code == 422


def test_predict_negative_length_returns_422(client):
    bad = {**PAYLOAD_HOLE, "length_m": -1.0}
    r = client.post("/predict", json=bad, auth=AUTH)
    assert r.status_code == 422


def test_predict_with_hole_missing_radius_returns_422(client):
    bad = {k: v for k, v in PAYLOAD_HOLE.items() if k != "hole_radius_ratio"}
    r = client.post("/predict", json=bad, auth=AUTH)
    assert r.status_code == 422


def test_predict_without_hole_with_radius_returns_422(client):
    bad = {**PAYLOAD_NO_HOLE, "hole_radius_ratio": 0.1}
    r = client.post("/predict", json=bad, auth=AUTH)
    assert r.status_code == 422


def test_predict_moving_hole_missing_cx_returns_422(client):
    bad = {k: v for k, v in PAYLOAD_MOVING.items() if k != "hole_cx_ratio"}
    r = client.post("/predict", json=bad, auth=AUTH)
    assert r.status_code == 422


def test_predict_invalid_poisson_returns_422(client):
    bad = {**PAYLOAD_HOLE, "poisson_ratio": 0.6}
    r = client.post("/predict", json=bad, auth=AUTH)
    assert r.status_code == 422


def test_predict_empty_material_category_returns_422(client):
    bad = {**PAYLOAD_HOLE, "material_category": "   "}
    r = client.post("/predict", json=bad, auth=AUTH)
    assert r.status_code == 422


def test_predict_result_contains_input_summary(client):
    r = client.post("/predict", json=PAYLOAD_HOLE, auth=AUTH)
    assert r.status_code == 200
    body = r.json()
    assert "input_summary" in body
    assert body["input_summary"]["geometry_type"] == "with_hole"
