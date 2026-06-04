"""Tests for src/api/model_loader.py."""
from pathlib import Path

import joblib
import numpy as np
import pytest

import src.api.model_loader as _ml
from src.api.model_loader import LoadedModel, ModelBundle, get_bundle, load_models


class _FakeModel:
    """Picklable fake model for joblib serialisation tests."""
    def predict(self, X):
        return np.ones(len(X)) * 6.0


def _write_fake_artifact(model_dir: Path, target: str, feature_cols=None) -> None:
    artifact = {
        "model": _FakeModel(),
        "feature_cols": feature_cols or ["length_m", "height_m"],
        "encoder": None,
        "normalize_by": None,
    }
    joblib.dump(artifact, model_dir / f"lgbm_{target}.joblib")


# ── get_bundle ────────────────────────────────────────────────────────────────

def test_get_bundle_returns_none_or_bundle(monkeypatch):
    monkeypatch.setattr(_ml, "_bundle", None)
    assert get_bundle() is None


def test_get_bundle_returns_set_bundle(monkeypatch):
    fake = ModelBundle(version="v_fake", model_dir=Path("/tmp"))
    monkeypatch.setattr(_ml, "_bundle", fake)
    assert get_bundle() is fake


# ── load_models ───────────────────────────────────────────────────────────────

def test_load_models_both_targets(tmp_path):
    model_dir = tmp_path / "v_test"
    model_dir.mkdir()
    _write_fake_artifact(model_dir, "max_displacement_m", ["a", "b", "c"])
    _write_fake_artifact(model_dir, "max_von_mises_pa", ["a", "b", "c"])

    bundle = load_models(model_dir)
    assert bundle.version == "v_test"
    assert "max_displacement_m" in bundle.models
    assert "max_von_mises_pa" in bundle.models


def test_load_models_returns_model_bundle(tmp_path):
    model_dir = tmp_path / "v1"
    model_dir.mkdir()
    _write_fake_artifact(model_dir, "max_displacement_m")
    _write_fake_artifact(model_dir, "max_von_mises_pa")
    bundle = load_models(model_dir)
    assert isinstance(bundle, ModelBundle)
    assert isinstance(bundle.models["max_displacement_m"], LoadedModel)


def test_load_models_sets_global_bundle(tmp_path):
    model_dir = tmp_path / "v_global"
    model_dir.mkdir()
    _write_fake_artifact(model_dir, "max_displacement_m")
    _write_fake_artifact(model_dir, "max_von_mises_pa")
    load_models(model_dir)
    assert _ml._bundle is not None
    assert _ml._bundle.version == "v_global"


def test_load_models_dir_not_found(tmp_path):
    with pytest.raises(RuntimeError, match="not found"):
        load_models(tmp_path / "nonexistent")


def test_load_models_no_artifacts_raises(tmp_path):
    empty = tmp_path / "v_empty"
    empty.mkdir()
    with pytest.raises(RuntimeError, match="No model artifacts"):
        load_models(empty)


def test_load_models_accepts_string_path(tmp_path):
    model_dir = tmp_path / "v_str"
    model_dir.mkdir()
    _write_fake_artifact(model_dir, "max_displacement_m")
    _write_fake_artifact(model_dir, "max_von_mises_pa")
    bundle = load_models(str(model_dir))
    assert bundle.version == "v_str"


def test_load_models_feature_count(tmp_path):
    model_dir = tmp_path / "v_fc"
    model_dir.mkdir()
    _write_fake_artifact(model_dir, "max_displacement_m", ["a", "b", "c"])
    _write_fake_artifact(model_dir, "max_von_mises_pa", ["a", "b", "c"])
    bundle = load_models(model_dir)
    assert bundle.feature_count == 3


def test_load_models_with_normalize_by(tmp_path):
    model_dir = tmp_path / "v_norm"
    model_dir.mkdir()
    artifact = {
        "model": _FakeModel(),
        "feature_cols": ["x"],
        "encoder": None,
        "normalize_by": "traction_pa",
    }
    for target in ["max_displacement_m", "max_von_mises_pa"]:
        joblib.dump(artifact, model_dir / f"lgbm_{target}.joblib")
    bundle = load_models(model_dir)
    assert bundle.models["max_displacement_m"].normalize_by == "traction_pa"


# ── _resolve_model_dir ────────────────────────────────────────────────────────

def test_resolve_model_dir_uses_env_var(monkeypatch, tmp_path):
    from src.api.model_loader import _resolve_model_dir
    monkeypatch.setenv("MODEL_DIR", str(tmp_path))
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    result = _resolve_model_dir()
    assert result == tmp_path


def test_resolve_model_dir_uses_registry_latest(monkeypatch, tmp_path):
    from src.api.model_loader import _resolve_model_dir
    monkeypatch.delenv("MODEL_DIR", raising=False)
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.setenv("REGISTRY_DIR", str(tmp_path))
    monkeypatch.setenv("MODEL_NAME", "my_model")

    latest_dir = tmp_path / "my_model" / "v_latest"
    latest_dir.mkdir(parents=True)
    (tmp_path / "my_model" / "latest.txt").write_text("v_latest", encoding="utf-8")

    result = _resolve_model_dir()
    assert result == latest_dir


def test_resolve_model_dir_raises_when_nothing_found(monkeypatch, tmp_path):
    from src.api.model_loader import _resolve_model_dir
    monkeypatch.delenv("MODEL_DIR", raising=False)
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.setenv("REGISTRY_DIR", str(tmp_path / "empty"))
    monkeypatch.setenv("MODEL_NAME", "no_model")
    with pytest.raises(RuntimeError):
        _resolve_model_dir()
