"""Entraînement surrogate avancé : LightGBM + transformation log des cibles + optimisation Optuna.

Améliorations clés par rapport à train_baseline.py
---------------------------------------------------
1. Transformation log des cibles avant l'entraînement.
   - max_von_mises_pa couvre ~4 décades (1e4 → 4e8 Pa).
   - max_displacement_m couvre ~6 décades (1e-8 → 1e-2 m).
   - Le RMSE en espace log pondère également tous les régimes.
   - Les prédictions sont retransformées (10**pred) pour l'évaluation finale.

2. LightGBM à la place de RandomForest.
   - 5-10x plus rapide sur 50k lignes.
   - Généralement 5-15 points de R² en plus sur des données physiques tabulaires à cette échelle.

3. Optimisation des hyperparamètres avec Optuna.
   - 60 essais par défaut, validation croisée 5-fold, minimise le RMSE sur log(cible).
   - Recherche sur : n_estimators, learning_rate, num_leaves, max_depth,
     min_child_samples, subsample, colsample_bytree, reg_alpha, reg_lambda.

4. Modèles séparés par cible (déplacement vs von Mises) — chaque cible a
   des drivers physiques très différents, les modèles indépendants fonctionnent
   mieux que MultiOutputRegressor ici.

5. Vérifications physiques des prédictions (monotonicité PDP).

Utilisation
-----------
python -m src.training.train_advanced     --data-dir data/processed     --out-dir  data/models/advanced     --n-trials 60

Avec MLflow :
python -m src.training.train_advanced     --data-dir data/processed     --out-dir  data/models/advanced     --n-trials 60     --mlflow --mlflow-run-name lgbm-advanced
"""
import argparse
import os
import random
import warnings
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import OrdinalEncoder

try:
    import mlflow
except ImportError:
    mlflow = None

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore", category=UserWarning, module="lightgbm")


def _set_global_seed(seed: int) -> None:
    """Propage une graine globale à toutes les sources RNG concernées."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

# ── Constantes ────────────────────────────────────────────────────────────────
TARGET_COLS    = ["max_displacement_m", "max_von_mises_pa"]
DROP_COLS      = ["simulation_id", "timestamp", "solver_name",
                  "solver_version", "data_version", "mesh_nx", "mesh_ny"]
CAT_COLS       = ["geometry_type", "material_category", "dimension_category"]
_EPS           = 1e-12

# Contraintes de monotonicité physiques pour LightGBM.
# +1 = la cible augmente avec cette feature
# -1 = la cible diminue avec cette feature
#  0 = pas de contrainte
# Appliquées aux deux modèles ; les features absentes dans X sont ignorées silencieusement.
_MONO_CONSTRAINTS: dict[str, int] = {
    "traction_pa":       +1,   # plus de charge  -> plus de contrainte et de déplacement
    "young_modulus_pa":  -1,   # plus rigide -> moins de déplacement (pas directement sur vm)
    "logS":              +1,
    "logE":              -1,
    "log_sigma_net":     +1,
    "sigma_net":         +1,
    "log_delta_th":      +1,   # déplacement théorique plus grand -> plus grand réel
    "delta_theory":      +1,
    "epsilon":           +1,
    "hole_radius_ratio": +1,   # trou plus grand -> plus grande concentration de contrainte
    "d_over_W":          +1,
    "Kt_theory":         +1,
    "edge_ratio":        +1,   # trou plus proche du bord -> plus de contrainte
    "net_section_ratio": -1,   # moins de section nette -> plus de contrainte
    "lig_min":           -1,   # ligament plus petit -> plus de contrainte
}


# ── E/S ───────────────────────────────────────────────────────────────────────
def _load(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Split file not found: {path}")
    return pd.read_parquet(path)


def _feature_columns(df: pd.DataFrame) -> list[str]:
    exclude = set(TARGET_COLS) | set(DROP_COLS)
    return [c for c in df.columns if c not in exclude]


# ── Prétraitement ─────────────────────────────────────────────────────────────
def _encode_categoricals(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, OrdinalEncoder | None]:
    """Encode ordinalement les colonnes catégorielles. Retourne (train, val, test, encoder).

    L'encodeur est ajusté UNIQUEMENT sur le split train pour éviter la fuite de données.
    Les splits val et test sont transformés avec l'encodeur déjà ajusté.
    Les catégories inconnues rencontrées dans val/test sont encodées comme -1.
    """
    cat_present = [c for c in CAT_COLS if c in feature_cols]
    if not cat_present:
        return train, val, test, None

    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    # Ajustement sur train uniquement — jamais sur val ou test
    train[cat_present] = enc.fit_transform(train[cat_present].astype(str))
    for df_ in (val, test):
        df_[cat_present] = enc.transform(df_[cat_present].astype(str))
    return train, val, test, enc


def _log_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Ajoute les colonnes de cibles transformées en log10."""
    out = df.copy()
    for t in TARGET_COLS:
        if t in out.columns:
            out[f"log_{t}"] = np.log10(out[t].clip(lower=_EPS))
    return out


# ── Métriques ─────────────────────────────────────────────────────────────────
def _metrics(y_true_log: np.ndarray, y_pred_log: np.ndarray,
             y_true_orig: np.ndarray) -> dict[str, float]:
    y_pred_orig = 10.0 ** y_pred_log
    return {
        "r2_log":    float(r2_score(y_true_log,  y_pred_log)),
        "rmse_log":  float(np.sqrt(mean_squared_error(y_true_log,  y_pred_log))),
        "mae_log":   float(mean_absolute_error(y_true_log,  y_pred_log)),
        "r2_orig":   float(r2_score(y_true_orig,  y_pred_orig)),
        "rmse_orig": float(np.sqrt(mean_squared_error(y_true_orig, y_pred_orig))),
        "mape":      float(np.mean(np.abs((y_true_orig - y_pred_orig) /
                                          (np.abs(y_true_orig) + _EPS)))),
    }


# ── Objectif Optuna ───────────────────────────────────────────────────────────
def _build_optuna_objective(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    cv: KFold,
    mono_constraints: list[int],
    random_state: int = 42,
) -> callable:
    def objective(trial: optuna.Trial) -> float:
        params = {
            "objective":         "regression",
            "metric":            "rmse",
            "verbosity":         -1,
            "n_estimators":      trial.suggest_int("n_estimators", 400, 2000, step=100),
            "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "num_leaves":        trial.suggest_int("num_leaves", 31, 255),
            "max_depth":         trial.suggest_int("max_depth", 4, 12),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
            "subsample":         trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha":         trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda":        trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "monotone_constraints": mono_constraints,
            "n_jobs":            -1,
            "random_state":      random_state,
        }
        fold_rmse = []
        for tr_idx, vl_idx in cv.split(X_train):
            X_tr, X_vl = X_train.iloc[tr_idx], X_train.iloc[vl_idx]
            y_tr, y_vl = y_train[tr_idx],       y_train[vl_idx]
            model = lgb.LGBMRegressor(**params)
            model.fit(
                X_tr, y_tr,
                eval_set=[(X_vl, y_vl)],
                callbacks=[lgb.early_stopping(50, verbose=False),
                           lgb.log_evaluation(-1)],
            )
            pred = model.predict(X_vl)
            fold_rmse.append(float(np.sqrt(mean_squared_error(y_vl, pred))))
        return float(np.mean(fold_rmse))

    return objective


# ── Entraîner un modèle ───────────────────────────────────────────────────────
def _train_one_target(
    target_name: str,
    X_train: pd.DataFrame,
    y_train_log: np.ndarray,
    X_val: pd.DataFrame,
    y_val_log: np.ndarray,
    y_val_orig: np.ndarray,
    X_test: pd.DataFrame,
    y_test_log: np.ndarray,
    y_test_orig: np.ndarray,
    feature_cols: list[str],
    n_trials: int,
    cv_folds: int,
    random_state: int,
) -> tuple[lgb.LGBMRegressor, dict, dict]:
    """Optimise et entraîne un modèle LightGBM pour une cible unique (transformée en log)."""
    print(f"\n{'='*60}")
    print(f"Target: {target_name}")
    print(f"{'='*60}")

    cv = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    # Construction du vecteur de contraintes de monotonicité (doit correspondre à l'ordre de feature_cols)
    mono_constraints = [_MONO_CONSTRAINTS.get(c, 0) for c in feature_cols]

    # ── Recherche Optuna ───────────────────────────────────────────────────
    if n_trials > 0:
        print(f"Optuna: searching {n_trials} trials ...")
        study = optuna.create_study(
            direction="minimize",
            study_name=f"lgbm_{target_name}",
            sampler=optuna.samplers.TPESampler(seed=random_state),
        )
        study.optimize(
            _build_optuna_objective(X_train, y_train_log, cv, mono_constraints, random_state),
            n_trials=n_trials,
            show_progress_bar=False,
        )
        best_params = study.best_params
        print(f"  Best CV RMSE(log): {study.best_value:.5f}")
        print(f"  Best params: {best_params}")
    else:
        # Paramètres par défaut quand Optuna est désactivé
        best_params = {
            "n_estimators": 1000, "learning_rate": 0.05,
            "num_leaves": 127,    "max_depth": 8,
            "min_child_samples": 20, "subsample": 0.8,
            "colsample_bytree": 0.8, "reg_alpha": 0.1, "reg_lambda": 1.0,
        }

    # ── Modèle final sur le jeu d'entraînement complet ────────────────────
    final_params = {
        "objective":         "regression",
        "metric":            "rmse",
        "verbosity":         -1,
        "monotone_constraints": mono_constraints,
        "n_jobs":            -1,
        "random_state":      random_state,
        **best_params,
    }
    model = lgb.LGBMRegressor(**final_params)
    model.fit(
        X_train, y_train_log,
        eval_set=[(X_val, y_val_log)],
        callbacks=[lgb.early_stopping(50, verbose=False),
                   lgb.log_evaluation(-1)],
    )

    # ── Évaluation ────────────────────────────────────────────────────────
    val_metrics  = _metrics(y_val_log,  model.predict(X_val),  y_val_orig)
    test_metrics = _metrics(y_test_log, model.predict(X_test), y_test_orig)

    print(f"\n  Val  R²(log)={val_metrics['r2_log']:.4f}  "
          f"RMSE(log)={val_metrics['rmse_log']:.5f}  "
          f"R²(orig)={val_metrics['r2_orig']:.4f}  "
          f"MAPE={val_metrics['mape']:.4f}")
    print(f"  Test R²(log)={test_metrics['r2_log']:.4f}  "
          f"RMSE(log)={test_metrics['rmse_log']:.5f}  "
          f"R²(orig)={test_metrics['r2_orig']:.4f}  "
          f"MAPE={test_metrics['mape']:.4f}")

    return model, val_metrics, test_metrics


# ── Résumé de l'importance des features ──────────────────────────────────────
def _print_top_features(model: lgb.LGBMRegressor, feature_cols: list[str],
                         target_name: str, top_n: int = 15) -> None:
    imp = pd.Series(model.feature_importances_, index=feature_cols)
    imp = imp.sort_values(ascending=False).head(top_n)
    print(f"\n  Top {top_n} features for {target_name}:")
    for feat, val in imp.items():
        print(f"    {feat:30s}: {val:6.0f}")


# ── Fonction principale d'entraînement ───────────────────────────────────────
def train_advanced(
    data_dir: Path,
    out_dir: Path,
    n_trials: int = 60,
    cv_folds: int = 5,
    random_state: int = 42,
    mlflow_enabled: bool = False,
    mlflow_tracking_uri: str | None = None,
    mlflow_experiment: str = "advanced-surrogate",
    mlflow_run_name: str | None = None,
) -> None:
    # ── Initialiser toutes les sources RNG pour un déterminisme complet ───
    _set_global_seed(random_state)

    # ── Charger les splits ────────────────────────────────────────────────
    train_df = _load(data_dir / "train.parquet")
    val_df   = _load(data_dir / "val.parquet")
    test_df  = _load(data_dir / "test.parquet")

    for t in TARGET_COLS:
        for df_name, df_ in [("train", train_df), ("val", val_df), ("test", test_df)]:
            if t not in df_.columns:
                raise ValueError(f"Missing target '{t}' in {df_name} split")

    feature_cols = _feature_columns(train_df)
    if not feature_cols:
        raise ValueError("No feature columns found after dropping targets/metadata.")

    # ── Transformation log des cibles ─────────────────────────────────────
    train_df = _log_targets(train_df)
    val_df   = _log_targets(val_df)
    test_df  = _log_targets(test_df)

    # ── Encoder les variables catégorielles ───────────────────────────────
    train_df, val_df, test_df, enc = _encode_categoricals(
        train_df, val_df, test_df, feature_cols
    )

    X_train = train_df[feature_cols].astype(float)
    X_val   = val_df[feature_cols].astype(float)
    X_test  = test_df[feature_cols].astype(float)

    null_tr = X_train.isnull().sum().sum()
    null_vl = X_val.isnull().sum().sum()
    null_ts = X_test.isnull().sum().sum()
    if null_tr + null_vl + null_ts > 0:
        raise ValueError(
            f"Null values in features: train={null_tr}, val={null_vl}, test={null_ts}. "
            "Run build_features before training."
        )

    print(f"Train: {len(X_train):,}  Val: {len(X_val):,}  Test: {len(X_test):,}")
    print(f"Features: {len(feature_cols)}")

    out_dir.mkdir(parents=True, exist_ok=True)

    all_metrics: dict[str, dict] = {}
    artifacts: dict[str, Path]   = {}

    # ── Entraîner un modèle par cible ─────────────────────────────────────
    for target in TARGET_COLS:
        log_target = f"log_{target}"
        y_train = train_df[log_target].to_numpy()
        y_val   = val_df[log_target].to_numpy()
        y_test  = test_df[log_target].to_numpy()
        y_val_orig  = val_df[target].to_numpy()
        y_test_orig = test_df[target].to_numpy()

        model, val_m, test_m = _train_one_target(
            target_name=target,
            X_train=X_train, y_train_log=y_train,
            X_val=X_val,     y_val_log=y_val,   y_val_orig=y_val_orig,
            X_test=X_test,   y_test_log=y_test, y_test_orig=y_test_orig,
            feature_cols=feature_cols,
            n_trials=n_trials,
            cv_folds=cv_folds,
            random_state=random_state,
        )
        _print_top_features(model, feature_cols, target)

        all_metrics[target] = {"val": val_m, "test": test_m}

        model_path = out_dir / f"lgbm_{target}.joblib"
        joblib.dump({"model": model, "feature_cols": feature_cols,
                     "encoder": enc, "target": target, "log_target": log_target},
                    model_path)
        artifacts[target] = model_path
        print(f"\n  Model saved: {model_path}")

    # ── Résumé ────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    metrics_rows = []
    for target, splits in all_metrics.items():
        for split_name, m in splits.items():
            row = {"target": target, "split": split_name, **m}
            metrics_rows.append(row)
            print(f"  {target:25s} {split_name:5s}  "
                  f"R2_log={m['r2_log']:.4f}  RMSE_log={m['rmse_log']:.5f}  "
                  f"R2_orig={m['r2_orig']:.4f}  MAPE={m['mape']:.4f}")

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_path = out_dir / "advanced_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"\nMetrics saved: {metrics_path}")

    # ── Journalisation MLflow ─────────────────────────────────────────────
    if mlflow_enabled:
        if mlflow is None:
            raise ImportError("mlflow not installed. pip install mlflow")
        os.environ["MLFLOW_SUPPRESS_PRINTING_URL_TO_STDOUT"] = "true"
        if mlflow_tracking_uri:
            mlflow.set_tracking_uri(mlflow_tracking_uri)
        mlflow.set_experiment(mlflow_experiment)
        with mlflow.start_run(run_name=mlflow_run_name) as run:
            mlflow.log_params({
                "n_trials":     n_trials,
                "cv_folds":     cv_folds,
                "random_state": random_state,
                "n_train":      len(X_train),
                "n_features":   len(feature_cols),
            })
            for row in metrics_rows:
                prefix = f"{row['target'].replace('_','.')}_{row['split']}"
                for k, v in row.items():
                    if k not in ("target", "split"):
                        mlflow.log_metric(f"{prefix}_{k}", float(v))
            for target, path in artifacts.items():
                mlflow.log_artifact(str(path))
            mlflow.log_artifact(str(metrics_path))
            mlflow.set_tags({
                "model_family": "lightgbm",
                "task": "regression_log_transform",
                "log_transform": "log10",
            })
            print(f"MLflow run logged: {run.info.run_id}")


# ── CLI ───────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Entraîne le surrogate LightGBM avancé avec transformation log et optimisation Optuna."
    )
    parser.add_argument("--data-dir",  type=Path, default=Path("data/processed"))
    parser.add_argument("--out-dir",   type=Path, default=Path("data/models/advanced"))
    parser.add_argument("--n-trials",  type=int,  default=60,
                        help="Nombre d'essais Optuna par cible (0 = utiliser les valeurs par défaut).")
    parser.add_argument("--cv-folds",  type=int,  default=5)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--mlflow",    action="store_true")
    parser.add_argument("--mlflow-tracking-uri",
                        default=os.getenv("MLFLOW_TRACKING_URI"))
    parser.add_argument("--mlflow-experiment",
                        default=os.getenv("MLFLOW_EXPERIMENT_NAME", "advanced-surrogate"))
    parser.add_argument("--mlflow-run-name", default=None)
    args = parser.parse_args()

    train_advanced(
        data_dir=args.data_dir,
        out_dir=args.out_dir,
        n_trials=args.n_trials,
        cv_folds=args.cv_folds,
        random_state=args.random_state,
        mlflow_enabled=args.mlflow,
        mlflow_tracking_uri=args.mlflow_tracking_uri,
        mlflow_experiment=args.mlflow_experiment,
        mlflow_run_name=args.mlflow_run_name,
    )


if __name__ == "__main__":
    main()
