"""Tests for feature engineering (src/processing/build_features.py).

Verifies:
  - Schema consistency (feature columns are stable across runs)
  - Feature completeness (all expected columns are produced)
  - Value correctness (physics formulas)
  - Deterministic splitting (same seed → same assignment)
  - No null values after feature computation
"""
import uuid

import numpy as np
import pandas as pd
import pytest

from src.processing.build_features import (
    MESH_COLS,
    TARGET_COLS,
    build_features,
    engineer_features,
    _feature_columns,
    _read_raw,
    _split,
)


def _make_row(geometry_type="with_hole", **kwargs) -> dict:
    """Return a minimal valid raw simulation row."""
    base = {
        "simulation_id":       str(uuid.uuid4()),
        "timestamp":           "2026-03-10T00:00:00Z",
        "material_category":   "steel",
        "dimension_category":  "medium",
        "length_m":            1.0,
        "height_m":            0.3,
        "young_modulus_pa":    2.1e11,
        "poisson_ratio":       0.3,
        "traction_pa":         1_500_000.0,
        "mesh_nx":             120,
        "mesh_ny":             24,
        "max_displacement_m":  5e-6,
        "max_von_mises_pa":    2e7,
        "solver_name":         "proxy",
        "solver_version":      "1.0",
        "data_version":        "sim_v1",
        "geometry_type":       geometry_type,
    }
    if geometry_type in ("with_hole", "with_hole_moving"):
        base["hole_radius_ratio"] = 0.1
    if geometry_type == "with_hole_moving":
        base["hole_cx_ratio"] = 0.5
        base["hole_cy_ratio"] = 0.5
    base.update(kwargs)
    return base


def _make_df(n: int = 10, geometry_type: str = "with_hole") -> pd.DataFrame:
    rows = [_make_row(geometry_type=geometry_type) for _ in range(n)]
    for i, r in enumerate(rows):
        r["simulation_id"] = f"sim-{i:06d}"
    return pd.DataFrame(rows)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def df_hole():
    return _make_df(20, "with_hole")


@pytest.fixture
def df_plain():
    return _make_df(20, "without_hole")


@pytest.fixture
def df_moving():
    return _make_df(20, "with_hole_moving")


@pytest.fixture
def df_split():
    return engineer_features(_make_df(200, "with_hole"))


# ── engineer_features tests ───────────────────────────────────────────────────

def test_returns_dataframe(df_hole):
    assert isinstance(engineer_features(df_hole), pd.DataFrame)


def test_row_count_preserved(df_hole):
    assert len(engineer_features(df_hole)) == len(df_hole)


@pytest.mark.parametrize("geom", ["with_hole", "without_hole", "with_hole_moving"])
def test_no_nulls_in_numeric_features(geom):
    df = _make_df(20, geom)
    out = engineer_features(df)
    feat_cols = _feature_columns(out)
    numeric = out[feat_cols].select_dtypes(include="number")
    nulls = numeric.isnull().sum().sum()
    bad = numeric.isnull().sum()
    assert nulls == 0, f"Unexpected nulls in numeric features: {bad[bad > 0]}"


def test_expected_derived_columns_present(df_hole):
    out = engineer_features(df_hole)
    expected = [
        "area_m2", "aspect_ratio", "d_over_W", "Kt_theory",
        "net_section_ratio", "sigma_net", "epsilon", "delta_theory",
        "biaxial_factor", "lig_min", "edge_ratio",
        "eccentricity", "stress_amp_proxy",
        "logE", "logS", "log_epsilon", "log_delta_th",
        "log_sigma_net", "log_Kt", "log_lig_min", "log_edge_ratio",
        "has_hole", "has_moving_hole", "traction_over_E",
    ]
    for col in expected:
        assert col in out.columns, f"Missing expected column: {col}"


def test_mesh_columns_present_but_excluded_from_features(df_hole):
    """Mesh columns pass through engineer_features but are excluded from model features."""
    out = engineer_features(df_hole)
    feat_cols = _feature_columns(out)
    for col in MESH_COLS:
        assert col in out.columns, f"Mesh column {col} should be in DataFrame"
        assert col not in feat_cols, f"Mesh column {col} should NOT be in feature list"


def test_target_cols_excluded_from_features(df_hole):
    out = engineer_features(df_hole)
    feat_cols = _feature_columns(out)
    for col in TARGET_COLS:
        assert col not in feat_cols


def test_physics_epsilon_formula(df_hole):
    """epsilon = traction / young_modulus."""
    out = engineer_features(df_hole)
    expected = df_hole["traction_pa"] / df_hole["young_modulus_pa"]
    np.testing.assert_allclose(out["epsilon"].values, expected.values, rtol=1e-9)


def test_delta_theory_formula(df_hole):
    """delta_theory = epsilon * length."""
    out = engineer_features(df_hole)
    expected = out["epsilon"] * df_hole["length_m"]
    np.testing.assert_allclose(out["delta_theory"].values, expected.values, rtol=1e-9)


def test_has_hole_flag(df_hole):
    out = engineer_features(df_hole)
    assert (out["has_hole"] == 1).all()


def test_no_hole_flag_for_plain_plate(df_plain):
    out = engineer_features(df_plain)
    assert (out["has_hole"] == 0).all()


def test_Kt_theory_equals_one_for_plain_plate(df_plain):
    out = engineer_features(df_plain)
    np.testing.assert_array_equal(out["Kt_theory"].values, 1.0)


def test_feature_schema_stable_across_runs(df_hole):
    """Running engineer_features twice with the same data yields the same column list."""
    out1 = engineer_features(df_hole)
    out2 = engineer_features(df_hole.copy())
    assert _feature_columns(out1) == _feature_columns(out2)


def test_require_targets_false_allows_missing_targets(df_hole):
    df_no_targets = df_hole.drop(columns=TARGET_COLS)
    out = engineer_features(df_no_targets, require_targets=False)
    assert isinstance(out, pd.DataFrame)


def test_missing_required_column_raises(df_hole):
    df_bad = df_hole.drop(columns=["young_modulus_pa"])
    with pytest.raises(ValueError):
        engineer_features(df_bad)


# ── _split tests ──────────────────────────────────────────────────────────────

def test_split_sizes_approx_correct(df_split):
    train, val, test = _split(df_split, 0.7, 0.15, seed=42)
    n = len(df_split)
    assert len(train) > 0
    assert len(val) > 0
    assert len(test) > 0
    assert len(train) + len(val) + len(test) == n


def test_hash_split_is_deterministic(df_split):
    t1, v1, _ = _split(df_split, 0.7, 0.15, seed=42, strategy="hash")
    t2, v2, _ = _split(df_split, 0.7, 0.15, seed=42, strategy="hash")
    assert list(t1["simulation_id"]) == list(t2["simulation_id"])
    assert list(v1["simulation_id"]) == list(v2["simulation_id"])


def test_different_seeds_give_different_splits(df_split):
    t1, _, _ = _split(df_split, 0.7, 0.15, seed=42, strategy="hash")
    t2, _, _ = _split(df_split, 0.7, 0.15, seed=99, strategy="hash")
    assert set(t1["simulation_id"]) != set(t2["simulation_id"])


def test_no_overlap_between_splits(df_split):
    train, val, test = _split(df_split, 0.7, 0.15, seed=42)
    ids_train = set(train["simulation_id"])
    ids_val   = set(val["simulation_id"])
    ids_test  = set(test["simulation_id"])
    assert len(ids_train & ids_val) == 0
    assert len(ids_train & ids_test) == 0
    assert len(ids_val & ids_test) == 0


def test_invalid_ratios_raise(df_split):
    with pytest.raises(ValueError):
        _split(df_split, 0.7, 0.4, seed=42)


def test_invalid_strategy_raises(df_split):
    with pytest.raises(ValueError):
        _split(df_split, 0.7, 0.15, seed=42, strategy="unknown")


def test_random_strategy_split(df_split):
    train, val, test = _split(df_split, 0.7, 0.15, seed=42, strategy="random")
    assert len(train) + len(val) + len(test) == len(df_split)
    assert len(train) > 0 and len(val) > 0 and len(test) > 0


def test_split_missing_simulation_id_raises(df_split):
    bad = df_split.drop(columns=["simulation_id"])
    with pytest.raises(ValueError, match="simulation_id"):
        _split(bad, 0.7, 0.15, seed=42)


def test_split_train_ratio_not_in_range(df_split):
    with pytest.raises(ValueError, match="train_ratio"):
        _split(df_split, 1.1, 0.15, seed=42)


def test_split_val_ratio_negative(df_split):
    with pytest.raises(ValueError, match="val_ratio"):
        _split(df_split, 0.7, -0.1, seed=42)


# ── _read_raw ─────────────────────────────────────────────────────────────────

def test_read_raw_no_parquet_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        _read_raw(tmp_path)


def test_read_raw_reads_parquet(tmp_path):
    df = _make_df(10)
    df.to_parquet(tmp_path / "data.parquet", index=False)
    result = _read_raw(tmp_path)
    assert len(result) == 10


# ── build_features ────────────────────────────────────────────────────────────

def _write_raw(input_dir, n: int = 200, geometry: str = "with_hole"):
    rows = [_make_row(geometry_type=geometry) for _ in range(n)]
    for i, r in enumerate(rows):
        r["simulation_id"] = f"sim-{i:06d}"
    pd.DataFrame(rows).to_parquet(input_dir / "raw.parquet", index=False)


def test_build_features_creates_split_files(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    _write_raw(raw_dir)
    out_dir = tmp_path / "processed"
    build_features(raw_dir, out_dir)
    assert (out_dir / "train.parquet").exists()
    assert (out_dir / "val.parquet").exists()
    assert (out_dir / "test.parquet").exists()
    assert (out_dir / "feature_columns.txt").exists()


def test_build_features_random_strategy(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    _write_raw(raw_dir)
    out_dir = tmp_path / "out"
    build_features(raw_dir, out_dir, split_strategy="random", seed=0)
    assert (out_dir / "train.parquet").exists()


def test_build_features_separate_features_dir(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    _write_raw(raw_dir)
    out_dir = tmp_path / "splits"
    feat_dir = tmp_path / "features"
    build_features(raw_dir, out_dir, features_out_dir=feat_dir)
    assert (feat_dir / "features.parquet").exists()
    assert (feat_dir / "feature_columns.txt").exists()


def test_build_features_feature_columns_nonempty(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    _write_raw(raw_dir)
    out_dir = tmp_path / "out"
    build_features(raw_dir, out_dir)
    cols = (out_dir / "feature_columns.txt").read_text().splitlines()
    assert len(cols) > 0
