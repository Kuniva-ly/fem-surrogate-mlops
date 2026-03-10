"""End-to-end smoke test for the ML pipeline.

Uses a tiny synthetic dataset (~100 rows) to verify the complete pipeline
runs without errors in a temporary directory:

  engineer_features → split → train_advanced (n_trials=0) → predict

No real FEM data or pre-trained models are required.
"""
import json
import tempfile
import unittest
import uuid
from pathlib import Path

import numpy as np
import pandas as pd

from src.processing.build_features import build_features, engineer_features, _feature_columns
from src.training.train_advanced import train_advanced


def _make_synthetic_dataset(n: int = 120, seed: int = 42) -> pd.DataFrame:
    """Generate a tiny synthetic simulation dataset."""
    rng = np.random.default_rng(seed)

    length_m          = rng.uniform(0.5, 2.0, n)
    height_m          = rng.uniform(0.1, 0.5, n)
    young_modulus_pa  = rng.uniform(7e10, 2.1e11, n)
    poisson_ratio     = rng.uniform(0.25, 0.35, n)
    traction_pa       = rng.uniform(5e5, 2e6, n)
    hole_radius_ratio = rng.uniform(0.05, 0.20, n)

    # Physics-based proxy targets (deterministic, no FEM)
    epsilon          = traction_pa / young_modulus_pa
    delta            = epsilon * length_m * rng.uniform(0.95, 1.05, n)
    d_over_W         = (2.0 * hole_radius_ratio * np.minimum(length_m, height_m)) / height_m
    Kt               = np.clip(3.0 - 3.13 * d_over_W + 3.66 * d_over_W**2, 1.0, 10.0)
    sigma_net        = traction_pa / np.clip(1.0 - d_over_W, 0.05, 1.0)
    von_mises        = Kt * sigma_net * rng.uniform(0.9, 1.1, n)

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


class TestEndToEndSmoke(unittest.TestCase):

    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.base = Path(self._tmpdir.name)
        self.raw_dir  = self.base / "raw"
        self.proc_dir = self.base / "processed"
        self.model_dir = self.base / "models"
        self.raw_dir.mkdir()

        # Write synthetic raw data as parquet
        df = _make_synthetic_dataset(n=120, seed=42)
        df.to_parquet(self.raw_dir / "part-00000.parquet", index=False)

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def test_build_features_runs(self) -> None:
        build_features(
            input_dir=self.raw_dir,
            out_dir=self.proc_dir,
            train_ratio=0.70,
            val_ratio=0.15,
            seed=42,
            split_strategy="hash",
            features_out_dir=self.proc_dir,
        )
        self.assertTrue((self.proc_dir / "train.parquet").exists())
        self.assertTrue((self.proc_dir / "val.parquet").exists())
        self.assertTrue((self.proc_dir / "test.parquet").exists())
        self.assertTrue((self.proc_dir / "feature_columns.txt").exists())

    def test_train_runs_no_optuna(self) -> None:
        """Training with n_trials=0 (skip Optuna) must complete without error."""
        build_features(
            input_dir=self.raw_dir,
            out_dir=self.proc_dir,
            train_ratio=0.70,
            val_ratio=0.15,
            seed=42,
            split_strategy="hash",
            features_out_dir=self.proc_dir,
        )
        train_advanced(
            data_dir=self.proc_dir,
            out_dir=self.model_dir,
            n_trials=0,
            cv_folds=2,
            random_state=42,
        )
        self.assertTrue((self.model_dir / "lgbm_max_displacement_m.joblib").exists())
        self.assertTrue((self.model_dir / "lgbm_max_von_mises_pa.joblib").exists())
        self.assertTrue((self.model_dir / "advanced_metrics.csv").exists())

    def test_predict_smoke(self) -> None:
        """After training, single-case inference returns plausible values."""
        import joblib
        from src.processing.build_features import engineer_features
        from src.training.train_advanced import CAT_COLS

        build_features(
            input_dir=self.raw_dir,
            out_dir=self.proc_dir,
            train_ratio=0.70,
            val_ratio=0.15,
            seed=42,
            split_strategy="hash",
            features_out_dir=self.proc_dir,
        )
        train_advanced(
            data_dir=self.proc_dir,
            out_dir=self.model_dir,
            n_trials=0,
            cv_folds=2,
            random_state=42,
        )

        raw_case = {
            "simulation_id":     "00000000-0000-0000-0000-000000000001",
            "timestamp":         "2026-03-10T00:00:00Z",
            "material_category": "steel",
            "dimension_category": "medium",
            "length_m":          1.0,
            "height_m":          0.3,
            "young_modulus_pa":  2.1e11,
            "poisson_ratio":     0.3,
            "traction_pa":       1_500_000.0,
            "mesh_nx":           120,
            "mesh_ny":           24,
            "hole_radius_ratio": 0.1,
            "geometry_type":     "with_hole",
        }

        _EPS = 1e-12
        for target in ["max_displacement_m", "max_von_mises_pa"]:
            model_file = self.model_dir / f"lgbm_{target}.joblib"
            artifact = joblib.load(model_file)
            model        = artifact["model"]
            feature_cols = artifact["feature_cols"]
            encoder      = artifact.get("encoder")

            case_df = pd.DataFrame([raw_case])
            case_df = engineer_features(case_df, require_targets=False)

            cat_present = [c for c in CAT_COLS if c in feature_cols]
            if encoder is not None and cat_present:
                case_df[cat_present] = encoder.transform(case_df[cat_present].astype(str))

            X = case_df[feature_cols].astype(float)
            y_pred_log = model.predict(X)
            pred = float(10.0 ** y_pred_log[0])

            self.assertGreater(pred, 0.0, f"{target} prediction must be > 0")
            self.assertFalse(np.isnan(pred), f"{target} prediction is NaN")

    def test_metrics_csv_has_expected_columns(self) -> None:
        build_features(
            input_dir=self.raw_dir,
            out_dir=self.proc_dir,
            train_ratio=0.70,
            val_ratio=0.15,
            seed=42,
            split_strategy="hash",
            features_out_dir=self.proc_dir,
        )
        train_advanced(
            data_dir=self.proc_dir,
            out_dir=self.model_dir,
            n_trials=0,
            cv_folds=2,
            random_state=42,
        )
        metrics = pd.read_csv(self.model_dir / "advanced_metrics.csv")
        for col in ("target", "split", "r2_log", "rmse_log", "mae_log",
                    "r2_orig", "rmse_orig", "mape"):
            self.assertIn(col, metrics.columns)

    def test_train_reproducibility(self) -> None:
        """Same seed → identical metrics on both runs."""
        build_features(
            input_dir=self.raw_dir,
            out_dir=self.proc_dir,
            train_ratio=0.70,
            val_ratio=0.15,
            seed=42,
            split_strategy="hash",
            features_out_dir=self.proc_dir,
        )
        model_dir1 = self.base / "models_run1"
        model_dir2 = self.base / "models_run2"

        train_advanced(data_dir=self.proc_dir, out_dir=model_dir1, n_trials=0, cv_folds=2, random_state=42)
        train_advanced(data_dir=self.proc_dir, out_dir=model_dir2, n_trials=0, cv_folds=2, random_state=42)

        m1 = pd.read_csv(model_dir1 / "advanced_metrics.csv")
        m2 = pd.read_csv(model_dir2 / "advanced_metrics.csv")

        pd.testing.assert_frame_equal(
            m1.reset_index(drop=True),
            m2.reset_index(drop=True),
            check_exact=False,
            rtol=1e-6,
        )


if __name__ == "__main__":
    unittest.main()
