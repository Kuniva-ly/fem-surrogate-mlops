"""Tests for the YAML configuration loader (src/config.py)."""
import os
import textwrap
from pathlib import Path

import pytest

from src.config import (
    ArtifactsConfig,
    DataConfig,
    EvaluationConfig,
    FeaturesConfig,
    PipelineConfig,
    TrainingConfig,
    load_config,
)


@pytest.fixture
def write_yaml(tmp_path):
    """Return a helper that writes a YAML snippet to a temp file."""
    def _write(content: str) -> Path:
        p = tmp_path / "config.yaml"
        p.write_text(textwrap.dedent(content), encoding="utf-8")
        return p
    return _write


# ── Default config (configs/training.yaml) ────────────────────────────────────

def test_loads_default_config():
    assert isinstance(load_config(), PipelineConfig)


def test_section_types():
    cfg = load_config()
    assert isinstance(cfg.data, DataConfig)
    assert isinstance(cfg.features, FeaturesConfig)
    assert isinstance(cfg.training, TrainingConfig)
    assert isinstance(cfg.artifacts, ArtifactsConfig)
    assert isinstance(cfg.evaluation, EvaluationConfig)


def test_known_defaults():
    cfg = load_config()
    assert cfg.features.seed == 42
    assert cfg.features.split_strategy == "hash"
    assert cfg.training.random_state == 42
    assert cfg.training.n_trials > 0
    assert cfg.training.cv_folds >= 2


def test_evaluation_metrics_not_empty():
    cfg = load_config()
    assert len(cfg.evaluation.metrics) > 0
    assert "r2_log" in cfg.evaluation.metrics
    assert "mape" in cfg.evaluation.metrics


def test_evaluation_splits_not_empty():
    cfg = load_config()
    assert "val" in cfg.evaluation.splits
    assert "test" in cfg.evaluation.splits


# ── Validation rejects invalid values ─────────────────────────────────────────

def test_train_ratio_out_of_range(write_yaml):
    path = write_yaml("""
        features:
          train_ratio: 1.5
          val_ratio: 0.15
    """)
    with pytest.raises(ValueError):
        load_config(path)


def test_split_ratios_sum_to_one(write_yaml):
    path = write_yaml("""
        features:
          train_ratio: 0.7
          val_ratio: 0.4
    """)
    with pytest.raises(ValueError):
        load_config(path)


def test_invalid_split_strategy(write_yaml):
    path = write_yaml("""
        features:
          split_strategy: kfold
    """)
    with pytest.raises(ValueError):
        load_config(path)


def test_cv_folds_too_small(write_yaml):
    path = write_yaml("""
        training:
          cv_folds: 1
    """)
    with pytest.raises(ValueError):
        load_config(path)


def test_unknown_metric(write_yaml):
    path = write_yaml("""
        evaluation:
          metrics:
            - r2_log
            - not_a_real_metric
    """)
    with pytest.raises(ValueError):
        load_config(path)


def test_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_config("nonexistent_config_12345.yaml")


def test_unknown_key_raises(write_yaml):
    path = write_yaml("""
        training:
          bogus_key: 999
    """)
    with pytest.raises(ValueError):
        load_config(path)


# ── Env-var override ──────────────────────────────────────────────────────────

def test_env_var_override():
    cfg_default = load_config()
    os.environ["CONFIG_PATH"] = str(Path("configs/training.yaml"))
    try:
        cfg_env = load_config()
        assert cfg_default.features.seed == cfg_env.features.seed
    finally:
        del os.environ["CONFIG_PATH"]
