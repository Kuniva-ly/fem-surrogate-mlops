"""Tests for src/processing/validate.py."""
import pandas as pd
import pytest

from src.processing.validate import (
    REQUIRED_COLUMNS,
    _assert_ranges,
    _coerce_optional_columns,
    validate_folder,
)


def _make_df(n: int = 5, geometry: str = "with_hole") -> pd.DataFrame:
    base = {
        "simulation_id":      [f"sim-{i:04d}" for i in range(n)],
        "timestamp":          ["2026-01-01T00:00:00Z"] * n,
        "material_category":  ["steel"] * n,
        "dimension_category": ["medium"] * n,
        "length_m":           [1.0] * n,
        "height_m":           [0.3] * n,
        "young_modulus_pa":   [2.1e11] * n,
        "poisson_ratio":      [0.3] * n,
        "traction_pa":        [1.5e6] * n,
        "mesh_nx":            [120] * n,
        "mesh_ny":            [24] * n,
        "max_displacement_m": [5e-6] * n,
        "max_von_mises_pa":   [2e7] * n,
        "solver_name":        ["proxy"] * n,
        "solver_version":     ["1.0"] * n,
        "data_version":       ["sim_v1"] * n,
        "geometry_type":      [geometry] * n,
    }
    if geometry in ("with_hole", "with_hole_moving"):
        base["hole_radius_ratio"] = [0.1] * n
    if geometry == "with_hole_moving":
        base["hole_cx_ratio"] = [0.5] * n
        base["hole_cy_ratio"] = [0.5] * n
    return pd.DataFrame(base)


# ── _coerce_optional_columns ──────────────────────────────────────────────────

def test_coerce_adds_missing_hole_columns():
    df = _make_df(3, "without_hole")
    out = _coerce_optional_columns(df)
    assert "hole_radius_ratio" in out.columns
    assert "hole_cx_ratio" in out.columns
    assert "hole_cy_ratio" in out.columns


def test_coerce_adds_geometry_type_if_absent():
    df = _make_df(3).drop(columns=["geometry_type"])
    out = _coerce_optional_columns(df)
    assert "geometry_type" in out.columns


def test_coerce_strips_geometry_type():
    df = _make_df(3)
    df["geometry_type"] = "  with_hole  "
    out = _coerce_optional_columns(df)
    assert (out["geometry_type"] == "with_hole").all()


# ── _assert_ranges ────────────────────────────────────────────────────────────

def test_assert_ranges_valid_with_hole():
    df = _coerce_optional_columns(_make_df(5, "with_hole"))
    _assert_ranges(df)  # no exception


def test_assert_ranges_valid_without_hole():
    df = _coerce_optional_columns(_make_df(5, "without_hole"))
    _assert_ranges(df)


def test_assert_ranges_valid_moving_hole():
    df = _coerce_optional_columns(_make_df(5, "with_hole_moving"))
    _assert_ranges(df)


def test_assert_ranges_negative_length():
    df = _coerce_optional_columns(_make_df())
    df["length_m"] = -1.0
    with pytest.raises(ValueError, match="length_m"):
        _assert_ranges(df)


def test_assert_ranges_zero_young_modulus():
    df = _coerce_optional_columns(_make_df())
    df["young_modulus_pa"] = 0.0
    with pytest.raises(ValueError):
        _assert_ranges(df)


def test_assert_ranges_invalid_poisson_too_high():
    df = _coerce_optional_columns(_make_df())
    df["poisson_ratio"] = 0.6
    with pytest.raises(ValueError):
        _assert_ranges(df)


def test_assert_ranges_invalid_poisson_zero():
    df = _coerce_optional_columns(_make_df())
    df["poisson_ratio"] = 0.0
    with pytest.raises(ValueError):
        _assert_ranges(df)


def test_assert_ranges_negative_displacement():
    df = _coerce_optional_columns(_make_df())
    df["max_displacement_m"] = -1e-6
    with pytest.raises(ValueError):
        _assert_ranges(df)


def test_assert_ranges_empty_material_category():
    df = _coerce_optional_columns(_make_df())
    df["material_category"] = "   "
    with pytest.raises(ValueError, match="material_category"):
        _assert_ranges(df)


def test_assert_ranges_empty_dimension_category():
    df = _coerce_optional_columns(_make_df())
    df["dimension_category"] = ""
    with pytest.raises(ValueError, match="dimension_category"):
        _assert_ranges(df)


def test_assert_ranges_invalid_geometry_type():
    df = _coerce_optional_columns(_make_df())
    df["geometry_type"] = df["geometry_type"].astype("string")
    df.loc[0, "geometry_type"] = "bad_geometry"
    with pytest.raises(ValueError, match="geometry_type"):
        _assert_ranges(df)


def test_assert_ranges_with_hole_missing_radius():
    df = _coerce_optional_columns(_make_df(5, "with_hole"))
    df["hole_radius_ratio"] = pd.NA
    with pytest.raises(ValueError, match="hole_radius_ratio"):
        _assert_ranges(df)


def test_assert_ranges_with_hole_has_cx_ratio():
    df = _coerce_optional_columns(_make_df(5, "with_hole"))
    df["hole_cx_ratio"] = 0.5
    with pytest.raises(ValueError, match="with_hole"):
        _assert_ranges(df)


def test_assert_ranges_without_hole_has_radius():
    df = _coerce_optional_columns(_make_df(5, "without_hole"))
    df["hole_radius_ratio"] = 0.1
    with pytest.raises(ValueError, match="without_hole"):
        _assert_ranges(df)


def test_assert_ranges_moving_hole_missing_cx():
    df = _coerce_optional_columns(_make_df(5, "with_hole_moving"))
    df["hole_cx_ratio"] = pd.NA
    with pytest.raises(ValueError, match="hole_cx_ratio"):
        _assert_ranges(df)


def test_assert_ranges_hole_radius_out_of_range():
    df = _coerce_optional_columns(_make_df(5, "with_hole"))
    df["hole_radius_ratio"] = 0.6  # must be < 0.5
    with pytest.raises(ValueError, match="hole_radius_ratio"):
        _assert_ranges(df)


def test_assert_ranges_cx_ratio_out_of_range():
    df = _coerce_optional_columns(_make_df(5, "with_hole_moving"))
    df["hole_cx_ratio"] = 1.5  # must be < 1
    with pytest.raises(ValueError, match="hole_cx_ratio"):
        _assert_ranges(df)


def test_assert_ranges_cy_ratio_out_of_range():
    df = _coerce_optional_columns(_make_df(5, "with_hole_moving"))
    df["hole_cy_ratio"] = 0.0  # must be > 0
    with pytest.raises(ValueError, match="hole_cy_ratio"):
        _assert_ranges(df)


# ── validate_folder ───────────────────────────────────────────────────────────

def test_validate_folder_parquet(tmp_path):
    _make_df(20, "with_hole").to_parquet(tmp_path / "data.parquet", index=False)
    validate_folder(tmp_path)  # no exception


def test_validate_folder_csv(tmp_path):
    _make_df(20, "without_hole").to_csv(tmp_path / "data.csv", index=False)
    validate_folder(tmp_path)


def test_validate_folder_prefers_parquet_over_csv(tmp_path):
    _make_df(10, "with_hole").to_parquet(tmp_path / "a.parquet", index=False)
    _make_df(10, "without_hole").to_csv(tmp_path / "b.csv", index=False)
    validate_folder(tmp_path)  # should not raise (parquet wins)


def test_validate_folder_no_files_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        validate_folder(tmp_path)


def test_validate_folder_missing_column_raises(tmp_path):
    df = _make_df(5).drop(columns=["traction_pa"])
    df.to_parquet(tmp_path / "data.parquet", index=False)
    with pytest.raises(ValueError, match="Missing columns"):
        validate_folder(tmp_path)


def test_validate_folder_duplicate_ids_raises(tmp_path):
    df = _make_df(5)
    df["simulation_id"] = "dup-0000"
    df.to_parquet(tmp_path / "data.parquet", index=False)
    with pytest.raises(ValueError, match="Duplicate"):
        validate_folder(tmp_path)


def test_validate_folder_null_in_required_column_raises(tmp_path):
    df = _make_df(5)
    df.loc[0, "length_m"] = None
    df.to_parquet(tmp_path / "data.parquet", index=False)
    with pytest.raises(ValueError, match="Null"):
        validate_folder(tmp_path)


def test_validate_folder_moving_hole(tmp_path):
    _make_df(10, "with_hole_moving").to_parquet(tmp_path / "data.parquet", index=False)
    validate_folder(tmp_path)
