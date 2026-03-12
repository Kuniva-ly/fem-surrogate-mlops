"""Protocole d'évaluation figé pour les modèles surrogate FEM.

Tous les modèles sont évalués avec le même ensemble de métriques et la même
politique de split, assurant la comparabilité des résultats entre les runs et versions.

Métriques (calculées dans l'espace log et l'espace original) :
    r2_log    — R² sur log10(cible)             [objectif d'optimisation]
    rmse_log  — RMSE sur log10(cible)
    mae_log   — MAE sur log10(cible)
    r2_orig   — R² sur les valeurs brutes        [interprétation pratique]
    rmse_orig — RMSE sur les valeurs brutes
    mape      — Erreur absolue moyenne en pourcentage sur les valeurs brutes

Fichiers de sortie :
    <prefix>_metrics.json   — JSON imbriqué {cible: {split: {métrique: valeur}}}
    <prefix>_metrics.csv    — tableau tidy (cible, split, métrique...)

Utilisation
-----------
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

# Clés de métriques ordonnées canoniquement — ordre stable sur toutes les sorties.
METRIC_KEYS = ("r2_log", "rmse_log", "mae_log", "r2_orig", "rmse_orig", "mape")


# ── Calcul des métriques de base ──────────────────────────────────────────────

def compute_metrics(
    y_true_log: np.ndarray,
    y_pred_log: np.ndarray,
    y_true_orig: np.ndarray,
) -> dict[str, float]:
    """Calcule les métriques d'évaluation canoniques.

    Args:
        y_true_log:  Vérité terrain dans l'espace log10.
        y_pred_log:  Prédictions du modèle dans l'espace log10.
        y_true_orig: Vérité terrain dans l'espace physique original.

    Returns:
        Dict avec les clés : r2_log, rmse_log, mae_log, r2_orig, rmse_orig, mape.
    """
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


# ── Utilitaires DataFrame ─────────────────────────────────────────────────────

def build_metrics_df(
    all_metrics: dict[str, dict[str, dict[str, float]]],
) -> pd.DataFrame:
    """Convertit le dict de métriques imbriqué en un DataFrame tidy.

    Args:
        all_metrics: ``{cible: {split: {clé_métrique: valeur}}}``

    Returns:
        DataFrame avec les colonnes : target, split, r2_log, rmse_log, ...
    """
    rows = []
    for target, splits in all_metrics.items():
        for split_name, m in splits.items():
            rows.append({"target": target, "split": split_name, **m})
    return pd.DataFrame(rows)


# ── Persistance ───────────────────────────────────────────────────────────────

def save_metrics(
    metrics_df: pd.DataFrame,
    out_dir: Path,
    prefix: str = "advanced",
) -> tuple[Path, Path]:
    """Sauvegarde les métriques aux formats JSON et CSV.

    Args:
        metrics_df: DataFrame tidy de :func:`build_metrics_df`.
        out_dir:    Répertoire de sortie.
        prefix:     Préfixe de fichier (défaut ``advanced``).

    Returns:
        ``(json_path, csv_path)``
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / f"{prefix}_metrics.json"
    csv_path  = out_dir / f"{prefix}_metrics.csv"

    # JSON imbriqué : {cible: {split: {métrique: valeur}}}
    nested: dict = {}
    for _, row in metrics_df.iterrows():
        t, s = row["target"], row["split"]
        nested.setdefault(t, {})[s] = {
            k: v for k, v in row.items() if k not in ("target", "split")
        }
    json_path.write_text(json.dumps(nested, indent=2), encoding="utf-8")

    # CSV tidy
    metrics_df.to_csv(csv_path, index=False)

    print(f"Metrics JSON : {json_path}")
    print(f"Metrics CSV  : {csv_path}")
    return json_path, csv_path


# ── Affichage console ─────────────────────────────────────────────────────────

def print_metrics_table(metrics_df: pd.DataFrame) -> None:
    """Affiche un tableau de métriques formaté sur stdout."""
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
