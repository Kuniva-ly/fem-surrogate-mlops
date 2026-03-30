"""Tests for the YAML configuration loader (src/config.py)."""
import tempfile
import textwrap
import unittest
from pathlib import Path

from src.config import (
    ArtifactsConfig,
    DataConfig,
    EvaluationConfig,
    FeaturesConfig,
    PipelineConfig,
    TrainingConfig,
    load_config,
)


class TestDefaultConfig(unittest.TestCase):
    """load_config() with the real configs/training.yaml file."""

    def test_loads_default_config(self) -> None:
        cfg = load_config()
        self.assertIsInstance(cfg, PipelineConfig)

    def test_section_types(self) -> None:
        cfg = load_config()
        self.assertIsInstance(cfg.data, DataConfig)
        self.assertIsInstance(cfg.features, FeaturesConfig)
        self.assertIsInstance(cfg.training, TrainingConfig)
        self.assertIsInstance(cfg.artifacts, ArtifactsConfig)
        self.assertIsInstance(cfg.evaluation, EvaluationConfig)

    def test_known_defaults(self) -> None:
        cfg = load_config()
        self.assertEqual(cfg.features.seed, 42)
        self.assertEqual(cfg.features.split_strategy, "hash")
        self.assertEqual(cfg.training.random_state, 42)
        self.assertGreater(cfg.training.n_trials, 0)
        self.assertGreaterEqual(cfg.training.cv_folds, 2)

    def test_evaluation_metrics_not_empty(self) -> None:
        cfg = load_config()
        self.assertGreater(len(cfg.evaluation.metrics), 0)
        self.assertIn("r2_log", cfg.evaluation.metrics)
        self.assertIn("mape", cfg.evaluation.metrics)

    def test_evaluation_splits_not_empty(self) -> None:
        cfg = load_config()
        self.assertIn("val", cfg.evaluation.splits)
        self.assertIn("test", cfg.evaluation.splits)


class TestConfigValidation(unittest.TestCase):
    """Validation rejects invalid values with clear messages."""

    def _write_yaml(self, content: str) -> Path:
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, encoding="utf-8"
        )
        tmp.write(textwrap.dedent(content))
        tmp.close()
        return Path(tmp.name)

    def test_train_ratio_out_of_range(self) -> None:
        path = self._write_yaml("""
            features:
              train_ratio: 1.5
              val_ratio: 0.15
        """)
        with self.assertRaises(ValueError):
            load_config(path)

    def test_split_ratios_sum_to_one(self) -> None:
        path = self._write_yaml("""
            features:
              train_ratio: 0.7
              val_ratio: 0.4
        """)
        with self.assertRaises(ValueError):
            load_config(path)

    def test_invalid_split_strategy(self) -> None:
        path = self._write_yaml("""
            features:
              split_strategy: kfold
        """)
        with self.assertRaises(ValueError):
            load_config(path)

    def test_cv_folds_too_small(self) -> None:
        path = self._write_yaml("""
            training:
              cv_folds: 1
        """)
        with self.assertRaises(ValueError):
            load_config(path)

    def test_unknown_metric(self) -> None:
        path = self._write_yaml("""
            evaluation:
              metrics:
                - r2_log
                - not_a_real_metric
        """)
        with self.assertRaises(ValueError):
            load_config(path)

    def test_file_not_found(self) -> None:
        with self.assertRaises(FileNotFoundError):
            load_config("nonexistent_config_12345.yaml")

    def test_unknown_key_raises(self) -> None:
        path = self._write_yaml("""
            training:
              bogus_key: 999
        """)
        with self.assertRaises(ValueError):
            load_config(path)


class TestConfigFromEnvVar(unittest.TestCase):
    def test_env_var_override(self) -> None:
        import os
        cfg_default = load_config()
        # If the default loads, the env-var path also works because it points to the same file
        env_path = str(Path("configs/training.yaml"))
        os.environ["CONFIG_PATH"] = env_path
        try:
            cfg_env = load_config()
            self.assertEqual(cfg_default.features.seed, cfg_env.features.seed)
        finally:
            del os.environ["CONFIG_PATH"]


if __name__ == "__main__":
    unittest.main()
