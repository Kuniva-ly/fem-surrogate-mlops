"""Frozen evaluation protocol for FEM surrogate models.

All models are evaluated with the same set of metrics and the same split policy,
ensuring comparability of results across runs and versions.

Metrics (computed in log space and original space):
    r2_log    — R² on log10(target)             [optimisation objective]
    rmse_log  — RMSE on log10(target)
    mae_log   — MAE on log10(target)
    r2_orig   — R² on raw values                [practical interpretation]
    rmse_orig — RMSE on raw values
    mape      — Mean absolute percentage error on raw values

Output files:
    <prefix>_metrics.json   — nested JSON {target: {split: {metric: value}}}
    <prefix>_metrics.csv    — tidy table (target, split, metric...)

Usage
-----
    from src.evaluation import compute_metrics, build_metrics_df, save_metrics

    m = compute_metrics(y_true_log, y_pred_log, y_true_orig)
    df = build_metrics_df({"max_displacement_m": {"val": m, "test": m2}})
    save_metrics(df, out_dir=Path("artifacts/models/.../"), prefix="advanced")
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

_EPS = 1e-12

# Canonically ordered metric keys — stable order across all outputs.
METRIC_KEYS = ("r2_log", "rmse_log", "mae_log", "r2_orig", "rmse_orig", "mape")


# Core metric computation

def compute_metrics(
    y_true_log: np.ndarray,
    y_pred_log: np.ndarray,
    y_true_orig: np.ndarray,
    denorm_factors: np.ndarray | None = None,
) -> dict[str, float]:
    """Compute canonical evaluation metrics.

    Args:
        y_true_log:     Ground truth in log10 space (normalised if applicable).
        y_pred_log:     Model predictions in log10 space (normalised if applicable).
        y_true_orig:    Ground truth in original physical space.
        denorm_factors: Per-row normalisation denominators (e.g. traction_pa or
                        delta_theory). When provided, predictions are back-transformed
                        via ``10^pred * denorm_factors`` to recover physical units.
                        When None, ``10^pred`` is used directly.

    Returns:
        Dict with keys: r2_log, rmse_log, mae_log, r2_orig, rmse_orig, mape.
    """
    if denorm_factors is not None:
        y_pred_orig = (10.0 ** y_pred_log) * denorm_factors
    else:
        y_pred_orig = 10.0 ** y_pred_log
    return {
        "r2_log":    float(r2_score(y_true_log,  y_pred_log)),
        "rmse_log":  float(np.sqrt(mean_squared_error(y_true_log,  y_pred_log))),
        "mae_log":   float(mean_absolute_error(y_true_log,  y_pred_log)),
        "r2_orig":   float(r2_score(y_true_orig,  y_pred_orig)),
        "rmse_orig": float(np.sqrt(mean_squared_error(y_true_orig, y_pred_orig))),
        "mape":      float(np.mean(
            np.abs((y_true_orig - y_pred_orig) / (np.abs(y_true_orig) + _EPS))
        )),
    }


# DataFrame utilities

def build_metrics_df(
    all_metrics: dict[str, dict[str, dict[str, float]]],
) -> pd.DataFrame:
    """Convert nested metrics dict to a tidy DataFrame.

    Args:
        all_metrics: ``{target: {split: {metric_key: value}}}``

    Returns:
        DataFrame with columns: target, split, r2_log, rmse_log, ...
    """
    rows = []
    for target, splits in all_metrics.items():
        for split_name, m in splits.items():
            rows.append({"target": target, "split": split_name, **m})
    return pd.DataFrame(rows)


# Persistence

def save_metrics(
    metrics_df: pd.DataFrame,
    out_dir: Path,
    prefix: str = "advanced",
) -> tuple[Path, Path]:
    """Save metrics in JSON and CSV formats.

    Args:
        metrics_df: Tidy DataFrame from :func:`build_metrics_df`.
        out_dir:    Output directory.
        prefix:     File prefix (default ``advanced``).

    Returns:
        ``(json_path, csv_path)``
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / f"{prefix}_metrics.json"
    csv_path  = out_dir / f"{prefix}_metrics.csv"

    # Nested JSON: {target: {split: {metric: value}}}
    nested: dict = {}
    for _, row in metrics_df.iterrows():
        t, s = row["target"], row["split"]
        nested.setdefault(t, {})[s] = {
            k: v for k, v in row.items() if k not in ("target", "split")
        }
    json_path.write_text(json.dumps(nested, indent=2), encoding="utf-8")

    # Tidy CSV
    metrics_df.to_csv(csv_path, index=False)

    print(f"Metrics JSON : {json_path}")
    print(f"Metrics CSV  : {csv_path}")
    return json_path, csv_path


# Console output

def print_metrics_table(metrics_df: pd.DataFrame) -> None:
    """Print a formatted metrics table to stdout."""
    print("\n" + "=" * 84)
    print("EVALUATION RESULTS")
    print("=" * 84)
    print(
        f"  {'target':28s} {'split':5s}  "
        f"{'R²(log)':>8} {'RMSE(log)':>10} {'MAE(log)':>9} "
        f"{'R²(orig)':>9} {'MAPE':>7}"
    )
    print("-" * 84)
    for _, row in metrics_df.iterrows():
        print(
            f"  {row['target']:28s} {row['split']:5s}  "
            f"{row.get('r2_log',    float('nan')):8.4f} "
            f"{row.get('rmse_log',  float('nan')):10.5f} "
            f"{row.get('mae_log',   float('nan')):9.5f} "
            f"{row.get('r2_orig',   float('nan')):9.4f} "
            f"{row.get('mape',      float('nan')):7.4f}"
        )
    print("=" * 84)
