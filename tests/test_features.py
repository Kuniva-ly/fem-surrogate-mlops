"""Tests pour l'ingénierie des features (src/processing/build_features.py).

Vérifie :
  - Cohérence du schéma (les colonnes de features sont stables entre les runs)
  - Complétude des features (toutes les colonnes attendues sont produites)
  - Exactitude des valeurs (formules physiques)
  - Division déterministe (même graine → même assignation)
  - Pas de valeurs nulles après le calcul des features
"""
import unittest
import uuid

import numpy as np
import pandas as pd

from src.processing.build_features import (
    MESH_COLS,
    TARGET_COLS,
    engineer_features,
    _feature_columns,
    _split,
)


def _make_row(geometry_type="with_hole", **kwargs) -> dict:
    """Retourne une ligne de simulation brute minimale valide."""
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
    # Rendre les simulation_ids uniques
    for i, r in enumerate(rows):
        r["simulation_id"] = f"sim-{i:06d}"
    return pd.DataFrame(rows)


class TestEngineerFeatures(unittest.TestCase):

    def setUp(self) -> None:
        self.df_hole = _make_df(20, "with_hole")
        self.df_plain = _make_df(20, "without_hole")
        self.df_moving = _make_df(20, "with_hole_moving")

    def test_returns_dataframe(self) -> None:
        out = engineer_features(self.df_hole)
        self.assertIsInstance(out, pd.DataFrame)

    def test_row_count_preserved(self) -> None:
        out = engineer_features(self.df_hole)
        self.assertEqual(len(out), len(self.df_hole))

    def test_no_nulls_in_numeric_features(self) -> None:
        for df in (self.df_hole, self.df_plain, self.df_moving):
            out = engineer_features(df)
            feat_cols = _feature_columns(out)
            numeric = out[feat_cols].select_dtypes(include="number")
            nulls = numeric.isnull().sum().sum()
            self.assertEqual(nulls, 0, f"Unexpected nulls in numeric features: {numeric.isnull().sum()[numeric.isnull().sum()>0]}")

    def test_expected_derived_columns_present(self) -> None:
        out = engineer_features(self.df_hole)
        for col in [
            "area_m2", "aspect_ratio", "d_over_W", "Kt_theory",
            "net_section_ratio", "sigma_net", "epsilon", "delta_theory",
            "biaxial_factor", "lig_min", "edge_ratio",
            "eccentricity", "stress_amp_proxy",
            "logE", "logS", "log_epsilon", "log_delta_th",
            "log_sigma_net", "log_Kt", "log_lig_min", "log_edge_ratio",
            "has_hole", "has_moving_hole", "traction_over_E",
        ]:
            self.assertIn(col, out.columns, f"Missing expected column: {col}")

    def test_mesh_columns_present_but_excluded_from_features(self) -> None:
        """Les colonnes de maillage transitent par engineer_features mais sont exclues des features du modèle."""
        out = engineer_features(self.df_hole)
        for col in MESH_COLS:
            self.assertIn(col, out.columns, f"Mesh column {col} should be in DataFrame")
        feat_cols = _feature_columns(out)
        for col in MESH_COLS:
            self.assertNotIn(col, feat_cols, f"Mesh column {col} should NOT be in feature list")

    def test_target_cols_excluded_from_features(self) -> None:
        out = engineer_features(self.df_hole)
        feat_cols = _feature_columns(out)
        for col in TARGET_COLS:
            self.assertNotIn(col, feat_cols)

    def test_physics_epsilon_formula(self) -> None:
        """epsilon = traction / module_de_young."""
        out = engineer_features(self.df_hole)
        expected = self.df_hole["traction_pa"] / self.df_hole["young_modulus_pa"]
        np.testing.assert_allclose(out["epsilon"].values, expected.values, rtol=1e-9)

    def test_delta_theory_formula(self) -> None:
        """delta_theory = epsilon * longueur."""
        out = engineer_features(self.df_hole)
        expected = out["epsilon"] * self.df_hole["length_m"]
        np.testing.assert_allclose(out["delta_theory"].values, expected.values, rtol=1e-9)

    def test_has_hole_flag(self) -> None:
        out = engineer_features(self.df_hole)
        self.assertTrue((out["has_hole"] == 1).all())

    def test_no_hole_flag_for_plain_plate(self) -> None:
        out = engineer_features(self.df_plain)
        self.assertTrue((out["has_hole"] == 0).all())

    def test_Kt_theory_equals_one_for_plain_plate(self) -> None:
        out = engineer_features(self.df_plain)
        np.testing.assert_array_equal(out["Kt_theory"].values, 1.0)

    def test_feature_schema_stable_across_runs(self) -> None:
        """Exécuter engineer_features deux fois avec les mêmes données donne la même liste de colonnes."""
        out1 = engineer_features(self.df_hole)
        out2 = engineer_features(self.df_hole.copy())
        self.assertEqual(_feature_columns(out1), _feature_columns(out2))

    def test_require_targets_false_allows_missing_targets(self) -> None:
        df_no_targets = self.df_hole.drop(columns=TARGET_COLS)
        out = engineer_features(df_no_targets, require_targets=False)
        self.assertIsInstance(out, pd.DataFrame)

    def test_missing_required_column_raises(self) -> None:
        df_bad = self.df_hole.drop(columns=["young_modulus_pa"])
        with self.assertRaises(ValueError):
            engineer_features(df_bad)


class TestSplit(unittest.TestCase):

    def setUp(self) -> None:
        df = _make_df(200, "with_hole")
        self.df = engineer_features(df)

    def test_split_sizes_approx_correct(self) -> None:
        train, val, test = _split(self.df, 0.7, 0.15, seed=42)
        n = len(self.df)
        self.assertGreater(len(train), 0)
        self.assertGreater(len(val), 0)
        self.assertGreater(len(test), 0)
        self.assertEqual(len(train) + len(val) + len(test), n)

    def test_hash_split_is_deterministic(self) -> None:
        t1, v1, _ = _split(self.df, 0.7, 0.15, seed=42, strategy="hash")
        t2, v2, _ = _split(self.df, 0.7, 0.15, seed=42, strategy="hash")
        self.assertEqual(list(t1["simulation_id"]), list(t2["simulation_id"]))
        self.assertEqual(list(v1["simulation_id"]), list(v2["simulation_id"]))

    def test_different_seeds_give_different_splits(self) -> None:
        t1, _, _ = _split(self.df, 0.7, 0.15, seed=42, strategy="hash")
        t2, _, _ = _split(self.df, 0.7, 0.15, seed=99, strategy="hash")
        self.assertNotEqual(set(t1["simulation_id"]), set(t2["simulation_id"]))

    def test_no_overlap_between_splits(self) -> None:
        train, val, test = _split(self.df, 0.7, 0.15, seed=42)
        ids_train = set(train["simulation_id"])
        ids_val   = set(val["simulation_id"])
        ids_test  = set(test["simulation_id"])
        self.assertEqual(len(ids_train & ids_val), 0)
        self.assertEqual(len(ids_train & ids_test), 0)
        self.assertEqual(len(ids_val & ids_test), 0)

    def test_invalid_ratios_raise(self) -> None:
        with self.assertRaises(ValueError):
            _split(self.df, 0.7, 0.4, seed=42)

    def test_invalid_strategy_raises(self) -> None:
        with self.assertRaises(ValueError):
            _split(self.df, 0.7, 0.15, seed=42, strategy="unknown")


if __name__ == "__main__":
    unittest.main()
