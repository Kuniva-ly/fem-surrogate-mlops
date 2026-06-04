"""Tests for src/evaluation.py."""
import json

import numpy as np
import pytest

from src.evaluation import (
    METRIC_KEYS,
    build_metrics_df,
    compute_metrics,
    print_metrics_table,
    save_metrics,
)


def _y(n: int = 50, noise: float = 0.05, seed: int = 42):
    rng = np.random.default_rng(seed)
    y_true_log = np.log10(rng.uniform(1e4, 1e8, n))
    y_pred_log = y_true_log + rng.normal(0, noise, n)
    return y_true_log, y_pred_log, 10**y_true_log


# ── compute_metrics ───────────────────────────────────────────────────────────

def test_compute_metrics_returns_all_keys():
    yt, yp, yo = _y()
    m = compute_metrics(yt, yp, yo)
    assert set(m.keys()) == set(METRIC_KEYS)


def test_compute_metrics_all_floats():
    yt, yp, yo = _y()
    m = compute_metrics(yt, yp, yo)
    assert all(isinstance(v, float) for v in m.values())


def test_compute_metrics_perfect_predictions():
    yt, _, yo = _y()
    m = compute_metrics(yt, yt, yo)
    assert m["r2_log"] == pytest.approx(1.0)
    assert m["rmse_log"] == pytest.approx(0.0, abs=1e-10)
    assert m["mae_log"] == pytest.approx(0.0, abs=1e-10)


def test_compute_metrics_with_denorm_factors():
    yt, yp, yo = _y()
    denorm = np.ones(len(yt)) * 1e6
    m = compute_metrics(yt, yp, yo, denorm_factors=denorm)
    assert all(k in m for k in METRIC_KEYS)


def test_compute_metrics_without_denorm():
    yt, yp, yo = _y()
    m1 = compute_metrics(yt, yp, yo, denorm_factors=None)
    m2 = compute_metrics(yt, yp, yo)
    assert m1["r2_log"] == pytest.approx(m2["r2_log"])


# ── build_metrics_df ──────────────────────────────────────────────────────────

def _sample_metrics():
    yt, yp, yo = _y()
    return compute_metrics(yt, yp, yo)


def test_build_metrics_df_row_count():
    m = _sample_metrics()
    df = build_metrics_df({"disp": {"val": m, "test": m}, "vm": {"val": m}})
    assert len(df) == 3


def test_build_metrics_df_has_required_columns():
    m = _sample_metrics()
    df = build_metrics_df({"disp": {"test": m}})
    assert "target" in df.columns
    assert "split" in df.columns
    for k in METRIC_KEYS:
        assert k in df.columns


def test_build_metrics_df_target_values():
    m = _sample_metrics()
    df = build_metrics_df({"max_displacement_m": {"test": m}})
    assert df["target"].iloc[0] == "max_displacement_m"
    assert df["split"].iloc[0] == "test"


# ── save_metrics ──────────────────────────────────────────────────────────────

def test_save_metrics_creates_json_and_csv(tmp_path):
    m = _sample_metrics()
    df = build_metrics_df({"disp": {"test": m}})
    json_path, csv_path = save_metrics(df, tmp_path, prefix="run")
    assert json_path.exists()
    assert csv_path.exists()


def test_save_metrics_json_structure(tmp_path):
    m = _sample_metrics()
    df = build_metrics_df({"disp": {"test": m}})
    json_path, _ = save_metrics(df, tmp_path)
    data = json.loads(json_path.read_text())
    assert "disp" in data
    assert "test" in data["disp"]
    assert "r2_log" in data["disp"]["test"]


def test_save_metrics_creates_parent_dirs(tmp_path):
    m = _sample_metrics()
    df = build_metrics_df({"disp": {"test": m}})
    deep_dir = tmp_path / "a" / "b" / "c"
    save_metrics(df, deep_dir)
    assert deep_dir.exists()


# ── print_metrics_table ───────────────────────────────────────────────────────

def test_print_metrics_table_outputs_header(capsys):
    m = _sample_metrics()
    df = build_metrics_df({"max_displacement_m": {"test": m}})
    print_metrics_table(df)
    out = capsys.readouterr().out
    assert "EVALUATION RESULTS" in out
    assert "max_displacement_m" in out
