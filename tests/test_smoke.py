"""End-to-end smoke test for the ML pipeline.

Uses a small synthetic dataset (~120 rows) to verify that the full pipeline
runs without error in a temporary directory:

  engineer_features → split → train_advanced (n_trials=0) → predict

No real FEM data or pre-trained model is required.
"""
import uuid

import joblib
import numpy as np
import pandas as pd
import pytest

from src.processing.build_features import build_features, engineer_features
from src.training.train_advanced import train_advanced, CAT_COLS


def _make_synthetic_dataset(n: int = 120, seed: int = 42) -> pd.DataFrame:
    """Generate a small synthetic simulation dataset."""
    rng = np.random.default_rng(seed)

    length_m          = rng.uniform(0.5, 2.0, n)
    height_m          = rng.uniform(0.1, 0.5, n)
    young_modulus_pa  = rng.uniform(7e10, 2.1e11, n)
    poisson_ratio     = rng.uniform(0.25, 0.35, n)
    traction_pa       = rng.uniform(5e5, 2e6, n)
    hole_radius_ratio = rng.uniform(0.05, 0.20, n)

    epsilon   = traction_pa / young_modulus_pa
    delta     = epsilon * length_m * rng.uniform(0.95, 1.05, n)
    d_over_W  = (2.0 * hole_radius_ratio * np.minimum(length_m, height_m)) / height_m
    Kt        = np.clip(3.0 - 3.13 * d_over_W + 3.66 * d_over_W**2, 1.0, 10.0)
    sigma_net = traction_pa / np.clip(1.0 - d_over_W, 0.05, 1.0)
    von_mises = Kt * sigma_net * rng.uniform(0.9, 1.1, n)

    geometry_type = rng.choice(
        ["with_hole", "without_hole", "with_hole_moving"], n, p=[0.5, 0.3, 0.2]
    )
    hole_cx = np.where(geometry_type == "with_hole_moving", rng.uniform(0.3, 0.7, n), np.nan)
    hole_cy = np.where(geometry_type == "with_hole_moving", rng.uniform(0.3, 0.7, n), np.nan)
    hole_r  = np.where(geometry_type == "without_hole", np.nan, hole_radius_ratio)

    return pd.DataFrame({
        "simulation_id":      [str(uuid.uuid4()) for _ in range(n)],
        "timestamp":          "2026-03-10T00:00:00Z",
        "material_category":  rng.choice(["steel", "aluminum", "titanium"], n),
        "dimension_category": rng.choice(["small", "medium", "large"], n),
        "length_m":           length_m,
        "height_m":           height_m,
        "young_modulus_pa":   young_modulus_pa,
        "poisson_ratio":      poisson_ratio,
        "traction_pa":        traction_pa,
        "hole_radius_ratio":  hole_r,
        "hole_cx_ratio":      hole_cx,
        "hole_cy_ratio":      hole_cy,
        "geometry_type":      geometry_type,
        "mesh_nx":            120,
        "mesh_ny":            24,
        "max_displacement_m": np.clip(delta, 1e-9, None),
        "max_von_mises_pa":   np.clip(von_mises, 1e3, None),
        "solver_name":        "proxy",
        "solver_version":     "smoke_test",
        "data_version":       "smoke_v0",
    })


# ── Shared fixtures ───────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def dirs(tmp_path_factory):
    base      = tmp_path_factory.mktemp("smoke")
    raw_dir   = base / "raw"
    proc_dir  = base / "processed"
    model_dir = base / "models"
    raw_dir.mkdir()
    _make_synthetic_dataset(n=120, seed=42).to_parquet(
        raw_dir / "part-00000.parquet", index=False
    )
    return {"raw": raw_dir, "proc": proc_dir, "model": model_dir, "base": base}


@pytest.fixture(scope="module")
def built(dirs):
    build_features(
        input_dir=dirs["raw"],
        out_dir=dirs["proc"],
        train_ratio=0.70,
        val_ratio=0.15,
        seed=42,
        split_strategy="hash",
        features_out_dir=dirs["proc"],
    )
    return dirs


@pytest.fixture(scope="module")
def trained(built):
    train_advanced(
        data_dir=built["proc"],
        out_dir=built["model"],
        n_trials=0,
        cv_folds=2,
        random_state=42,
    )
    return built


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_build_features_runs(built):
    proc = built["proc"]
    assert (proc / "train.parquet").exists()
    assert (proc / "val.parquet").exists()
    assert (proc / "test.parquet").exists()
    assert (proc / "feature_columns.txt").exists()


def test_train_runs_no_optuna(trained):
    """Training with n_trials=0 (skip Optuna) must complete without error."""
    model_dir = trained["model"]
    assert (model_dir / "lgbm_max_displacement_m.joblib").exists()
    assert (model_dir / "lgbm_max_von_mises_pa.joblib").exists()
    assert (model_dir / "advanced_metrics.csv").exists()


@pytest.mark.parametrize("target", ["max_displacement_m", "max_von_mises_pa"])
def test_predict_smoke(trained, target):
    """After training, inference on a single case returns plausible values."""
    raw_case = {
        "simulation_id":      "00000000-0000-0000-0000-000000000001",
        "timestamp":          "2026-03-10T00:00:00Z",
        "material_category":  "steel",
        "dimension_category": "medium",
        "length_m":           1.0,
        "height_m":           0.3,
        "young_modulus_pa":   2.1e11,
        "poisson_ratio":      0.3,
        "traction_pa":        1_500_000.0,
        "mesh_nx":            120,
        "mesh_ny":            24,
        "hole_radius_ratio":  0.1,
        "geometry_type":      "with_hole",
    }

    artifact     = joblib.load(trained["model"] / f"lgbm_{target}.joblib")
    model        = artifact["model"]
    feature_cols = artifact["feature_cols"]
    encoder      = artifact.get("encoder")

    case_df = engineer_features(pd.DataFrame([raw_case]), require_targets=False)

    cat_present = [c for c in CAT_COLS if c in feature_cols]
    if encoder is not None and cat_present:
        case_df[cat_present] = encoder.transform(case_df[cat_present].astype(str))

    pred = float(10.0 ** model.predict(case_df[feature_cols].astype(float))[0])
    assert pred > 0.0, f"{target} prediction must be > 0"
    assert not np.isnan(pred), f"{target} prediction is NaN"


def test_metrics_csv_has_expected_columns(trained):
    metrics = pd.read_csv(trained["model"] / "advanced_metrics.csv")
    for col in ("target", "split", "r2_log", "rmse_log", "mae_log",
                "r2_orig", "rmse_orig", "mape"):
        assert col in metrics.columns


def test_train_reproducibility(built):
    """Same seed → identical metrics across two runs."""
    base      = built["base"]
    model_dir1 = base / "models_run1"
    model_dir2 = base / "models_run2"

    train_advanced(data_dir=built["proc"], out_dir=model_dir1, n_trials=0, cv_folds=2, random_state=42)
    train_advanced(data_dir=built["proc"], out_dir=model_dir2, n_trials=0, cv_folds=2, random_state=42)

    m1 = pd.read_csv(model_dir1 / "advanced_metrics.csv")
    m2 = pd.read_csv(model_dir2 / "advanced_metrics.csv")
    pd.testing.assert_frame_equal(
        m1.reset_index(drop=True),
        m2.reset_index(drop=True),
        check_exact=False,
        rtol=1e-6,
    )
