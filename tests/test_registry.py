"""Tests for the model registry (src/registry.py).

Verifies:
  - Version creation
  - Artifact registration (copy)
  - Latest pointer
  - Version listing
  - Save → registry → load round-trip for a real model
"""
import json
import time

import joblib
import numpy as np
import pytest
from sklearn.dummy import DummyRegressor

from src.registry import ModelRegistry, make_version


# ── make_version ──────────────────────────────────────────────────────────────

def test_format_starts_with_v():
    assert make_version().startswith("v")


def test_format_length():
    # "v" + YYYYMMDD + "_" + HHMMSS = 1 + 8 + 1 + 6 = 16
    assert len(make_version()) == 16


def test_two_calls_differ():
    v1 = make_version()
    time.sleep(1.01)
    v2 = make_version()
    assert v1 != v2


# ── ModelRegistry ─────────────────────────────────────────────────────────────

@pytest.fixture
def registry(tmp_path):
    return ModelRegistry(tmp_path / "registry", "test_model")


@pytest.fixture
def artifacts(tmp_path):
    """Small directory with fake artifacts."""
    src = tmp_path / "artifacts"
    src.mkdir()
    (src / "model.joblib").write_bytes(b"fake_model_bytes")
    (src / "metrics.csv").write_text("target,split,r2\nfoo,val,0.99\n")
    (src / "feature_columns.txt").write_text("feat_a\nfeat_b\n")
    return src


def test_initially_no_versions(registry):
    assert registry.list_versions() == []
    assert registry.latest_version() is None
    assert registry.latest_path() is None


def test_register_creates_version_dir(registry, artifacts):
    version, path = registry.register(artifacts, version="v20260310_120000")
    assert version == "v20260310_120000"
    assert path.exists()


def test_register_copies_files(registry, artifacts):
    _, path = registry.register(artifacts, version="v20260310_120000")
    assert (path / "model.joblib").exists()
    assert (path / "metrics.csv").exists()
    assert (path / "feature_columns.txt").exists()


def test_latest_txt_written(registry, artifacts):
    registry.register(artifacts, version="v20260310_120000")
    assert registry.latest_version() == "v20260310_120000"


def test_latest_path_resolves(registry, artifacts):
    _, expected_path = registry.register(artifacts, version="v20260310_120000")
    assert registry.latest_path() == expected_path


def test_list_versions_sorted(registry, artifacts):
    registry.register(artifacts, version="v20260310_110000")
    registry.register(artifacts, version="v20260310_120000")
    assert registry.list_versions() == ["v20260310_110000", "v20260310_120000"]


def test_latest_points_to_last_registered(registry, artifacts):
    registry.register(artifacts, version="v20260310_110000")
    registry.register(artifacts, version="v20260310_120000")
    assert registry.latest_version() == "v20260310_120000"


def test_register_with_metrics_dict(registry, artifacts):
    metrics = {"max_displacement_m": {"val": {"r2_log": 0.99}}}
    _, path = registry.register(artifacts, version="v20260310_120000", metrics=metrics)
    loaded = json.loads((path / "metrics.json").read_text())
    assert abs(loaded["max_displacement_m"]["val"]["r2_log"] - 0.99) < 1e-9


def test_get_version_path_missing_raises(registry):
    with pytest.raises(FileNotFoundError):
        registry.get_version_path("v99999999_000000")


def test_load_metrics_latest(registry, artifacts):
    metrics = {"r2_log": 0.95}
    registry.register(artifacts, version="v20260310_120000", metrics=metrics)
    loaded = registry.load_metrics()
    assert abs(loaded["r2_log"] - 0.95) < 1e-9


# ── Save → registry → load round-trip ────────────────────────────────────────

def test_model_roundtrip(tmp_path):
    model = DummyRegressor(strategy="mean")
    X = np.array([[1.0], [2.0], [3.0]])
    y = np.array([10.0, 20.0, 30.0])
    model.fit(X, y)
    pred_before = model.predict(X)

    src_dir = tmp_path / "source"
    src_dir.mkdir()
    artifact = {"model": model, "feature_cols": ["feat_a"], "encoder": None}
    joblib.dump(artifact, src_dir / "lgbm_max_displacement_m.joblib")

    reg = ModelRegistry(tmp_path / "registry", "test_model")
    _, reg_path = reg.register(src_dir, version="v20260310_120000")

    loaded = joblib.load(reg_path / "lgbm_max_displacement_m.joblib")
    np.testing.assert_array_equal(loaded["model"].predict(X), pred_before)
    assert loaded["feature_cols"] == ["feat_a"]
