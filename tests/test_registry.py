"""Tests for the model registry (src/registry.py).

Verifies:
  - Version creation
  - Artifact registration (copy)
  - Latest pointer
  - Version listing
  - Save → registry → load round-trip for a real model
"""
import json
import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
from sklearn.dummy import DummyRegressor

from src.registry import ModelRegistry, make_version


class TestMakeVersion(unittest.TestCase):
    def test_format_starts_with_v(self) -> None:
        v = make_version()
        self.assertTrue(v.startswith("v"))

    def test_format_length(self) -> None:
        v = make_version()
        # "v" + YYYYMMDD + "_" + HHMMSS = 1 + 8 + 1 + 6 = 16
        self.assertEqual(len(v), 16)

    def test_two_calls_differ(self) -> None:
        import time
        v1 = make_version()
        time.sleep(1.01)
        v2 = make_version()
        self.assertNotEqual(v1, v2)


class TestModelRegistry(unittest.TestCase):

    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.registry_dir = Path(self._tmpdir.name) / "registry"
        self.model_name   = "test_model"
        self.registry     = ModelRegistry(self.registry_dir, self.model_name)

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def _create_artifacts(self) -> Path:
        """Create a small temporary directory with fake artifacts."""
        src = Path(self._tmpdir.name) / "artifacts"
        src.mkdir(exist_ok=True)
        (src / "model.joblib").write_bytes(b"fake_model_bytes")
        (src / "metrics.csv").write_text("target,split,r2\nfoo,val,0.99\n")
        (src / "feature_columns.txt").write_text("feat_a\nfeat_b\n")
        return src

    def test_initially_no_versions(self) -> None:
        self.assertEqual(self.registry.list_versions(), [])
        self.assertIsNone(self.registry.latest_version())
        self.assertIsNone(self.registry.latest_path())

    def test_register_creates_version_dir(self) -> None:
        src = self._create_artifacts()
        version, path = self.registry.register(src, version="v20260310_120000")
        self.assertEqual(version, "v20260310_120000")
        self.assertTrue(path.exists())

    def test_register_copies_files(self) -> None:
        src = self._create_artifacts()
        _, path = self.registry.register(src, version="v20260310_120000")
        self.assertTrue((path / "model.joblib").exists())
        self.assertTrue((path / "metrics.csv").exists())
        self.assertTrue((path / "feature_columns.txt").exists())

    def test_latest_txt_written(self) -> None:
        src = self._create_artifacts()
        version, _ = self.registry.register(src, version="v20260310_120000")
        self.assertEqual(self.registry.latest_version(), "v20260310_120000")

    def test_latest_path_resolves(self) -> None:
        src = self._create_artifacts()
        version, expected_path = self.registry.register(src, version="v20260310_120000")
        self.assertEqual(self.registry.latest_path(), expected_path)

    def test_list_versions_sorted(self) -> None:
        src = self._create_artifacts()
        self.registry.register(src, version="v20260310_110000")
        self.registry.register(src, version="v20260310_120000")
        versions = self.registry.list_versions()
        self.assertEqual(versions, ["v20260310_110000", "v20260310_120000"])

    def test_latest_points_to_last_registered(self) -> None:
        src = self._create_artifacts()
        self.registry.register(src, version="v20260310_110000")
        self.registry.register(src, version="v20260310_120000")
        self.assertEqual(self.registry.latest_version(), "v20260310_120000")

    def test_register_with_metrics_dict(self) -> None:
        src = self._create_artifacts()
        metrics = {"max_displacement_m": {"val": {"r2_log": 0.99}}}
        _, path = self.registry.register(src, version="v20260310_120000", metrics=metrics)
        loaded = json.loads((path / "metrics.json").read_text())
        self.assertAlmostEqual(loaded["max_displacement_m"]["val"]["r2_log"], 0.99)

    def test_get_version_path_missing_raises(self) -> None:
        with self.assertRaises(FileNotFoundError):
            self.registry.get_version_path("v99999999_000000")

    def test_load_metrics_latest(self) -> None:
        src = self._create_artifacts()
        metrics = {"r2_log": 0.95}
        self.registry.register(src, version="v20260310_120000", metrics=metrics)
        loaded = self.registry.load_metrics()
        self.assertAlmostEqual(loaded["r2_log"], 0.95)


class TestModelSaveLoadRoundtrip(unittest.TestCase):
    """Verify that a real sklearn model survives the joblib save → registry → load cycle."""

    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.registry_dir = Path(self._tmpdir.name) / "registry"
        self.src_dir      = Path(self._tmpdir.name) / "source"
        self.src_dir.mkdir()

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def test_model_roundtrip(self) -> None:
        # Train a trivial model
        model = DummyRegressor(strategy="mean")
        X = np.array([[1.0], [2.0], [3.0]])
        y = np.array([10.0, 20.0, 30.0])
        model.fit(X, y)
        pred_before = model.predict(X)

        # Save the artifact bundle
        artifact = {"model": model, "feature_cols": ["feat_a"], "encoder": None}
        model_path = self.src_dir / "lgbm_max_displacement_m.joblib"
        joblib.dump(artifact, model_path)

        # Register
        registry = ModelRegistry(self.registry_dir, "test_model")
        _, reg_path = registry.register(self.src_dir, version="v20260310_120000")

        # Load from the registry
        loaded = joblib.load(reg_path / "lgbm_max_displacement_m.joblib")
        pred_after = loaded["model"].predict(X)
        np.testing.assert_array_equal(pred_before, pred_after)
        self.assertEqual(loaded["feature_cols"], ["feat_a"])


if __name__ == "__main__":
    unittest.main()
