"""Advanced surrogate training: LightGBM + log target transformation + Optuna optimisation.

Key improvements over train_baseline.py
----------------------------------------
1. Log transformation of targets before training.
   - max_von_mises_pa spans ~4 decades (1e4 → 4e8 Pa).
   - max_displacement_m spans ~6 decades (1e-8 → 1e-2 m).
   - Log-space RMSE weights all regimes equally.
   - Predictions are back-transformed (10**pred) for final evaluation.

2. LightGBM instead of RandomForest.
   - 5-10x faster on 50k rows.
   - Typically 5-15 R² points better on physical tabular data at this scale.

3. Hyperparameter optimisation with Optuna.
   - 60 trials by default, 5-fold cross-validation, minimises RMSE on log(target).
   - Search over: n_estimators, learning_rate, num_leaves, max_depth,
     min_child_samples, subsample, colsample_bytree, reg_alpha, reg_lambda.

4. Separate models per target (displacement vs von Mises) — each target has
   very different physical drivers, independent models work better than
   MultiOutputRegressor here.

5. Physical prediction checks (PDP monotonicity).

Usage
-----
python -m src.training.train_advanced     --data-dir data/processed     --out-dir  data/models/advanced     --n-trials 60

With MLflow:
python -m src.training.train_advanced     --data-dir data/processed     --out-dir  data/models/advanced     --n-trials 60     --mlflow --mlflow-run-name lgbm-advanced
"""
import argparse
import logging
import os
import random
import warnings
from pathlib import Path

logger = logging.getLogger(__name__)

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
    """Propagate a global seed to all relevant RNG sources."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

# Constants
TARGET_COLS    = ["max_displacement_m", "max_von_mises_pa"]
DROP_COLS      = ["simulation_id", "timestamp", "solver_name",
                  "solver_version", "data_version", "mesh_nx", "mesh_ny",
                  "simulation_source", "source"]
CAT_COLS       = ["geometry_type", "material_category", "dimension_category"]
_EPS           = 1e-12

# Normalisation des cibles avant log :
#   max_von_mises_pa   → divisé par traction_pa   → modèle prédit log10(Kt_eff)
#   max_displacement_m → divisé par delta_theory   → modèle prédit log10(C_disp)
# L'API restitue la grandeur physique en multipliant par le dénominateur à l'inférence.
# Cette normalisation supprime la dépendance linéaire en traction que LightGBM
# (modèle additif) ne peut pas apprendre fiablement quand la géométrie varie davantage.
NORMALIZE_BY: dict[str, str] = {
    "max_von_mises_pa":   "traction_pa",   # Kt_eff = σ_VM / σ_app
    "max_displacement_m": "delta_theory",  # C_disp = u / u_théorique
}

# Physical monotonicity constraints for LightGBM.
# +1 = target increases with this feature
# -1 = target decreases with this feature
#  0 = no constraint
# Applied to both models; features absent from X are silently ignored.
# Après normalisation, les features proportionnels à la traction n'ont plus
# de relation monotone avec la cible adimensionnelle → contrainte = 0.
_MONO_CONSTRAINTS: dict[str, int] = {
    "traction_pa":       0,    # Kt_eff et C_disp sont indépendants de la charge (élasticité linéaire)
    "young_modulus_pa":  -1,   # stiffer -> less displacement (not directly on vm)
    "logS":              0,    # log(traction) — normalisé hors cible
    "logE":              -1,
    "log_sigma_net":     0,    # ∝ traction — normalisé hors cible
    "sigma_net":         0,    # ∝ traction — normalisé hors cible
    "log_delta_th":      0,    # ∝ traction — normalisé hors cible (déplacement)
    "delta_theory":      0,    # normalisé hors cible (déplacement)
    "epsilon":           0,    # = traction/E — normalisé hors cible
    "hole_radius_ratio": +1,   # larger hole -> greater stress concentration
    "d_over_W":          +1,
    "Kt_theory":         +1,
    "edge_ratio":        +1,   # hole closer to edge -> more stress
    "net_section_ratio": -1,   # less net section -> more stress
    "lig_min":           -1,   # smaller ligament -> more stress
}


# I/O
def _load(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Split file not found: {path}")
    return pd.read_parquet(path)


def _download_splits_from_minio(local_dir: Path) -> None:
    """Télécharge train/val/test splits depuis le bucket MinIO features."""
    from src.utils.s3_client import get_s3_client, BUCKET_FEATURES

    s3 = get_s3_client()
    local_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Download splits depuis s3://%s/...", BUCKET_FEATURES)
    for filename in ("train.parquet", "val.parquet", "test.parquet"):
        s3.download_file(BUCKET_FEATURES, filename, str(local_dir / filename))
        logger.info("  ↓ %s", filename)


def _feature_columns(df: pd.DataFrame) -> list[str]:
    exclude = set(TARGET_COLS) | set(DROP_COLS)
    return [c for c in df.columns if c not in exclude]


# Pre-processing
def _encode_categoricals(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, OrdinalEncoder | None]:
    """Ordinally encode categorical columns. Returns (train, val, test, encoder).

    The encoder is fitted ONLY on the train split to prevent data leakage.
    Val and test splits are transformed with the already-fitted encoder.
    Unknown categories encountered in val/test are encoded as -1.
    """
    cat_present = [c for c in CAT_COLS if c in feature_cols]
    if not cat_present:
        return train, val, test, None

    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    # Fit on train only — never on val or test
    train[cat_present] = enc.fit_transform(train[cat_present].astype(str))
    for df_ in (val, test):
        df_[cat_present] = enc.transform(df_[cat_present].astype(str))
    return train, val, test, enc


def _normalize_and_log_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize each target by its physical reference then apply log10.

    Von Mises  : log10(σ_VM / traction_pa)   → model predicts log10(Kt_eff)
    Displacement: log10(u / delta_theory)      → model predicts log10(C_disp)

    This removes the linear-in-load component from the target so that tree
    splits capture only geometric/material effects (Kt, correction factor).
    The API multiplies the prediction back by the normaliser at inference time.
    """
    out = df.copy()
    for t in TARGET_COLS:
        if t not in out.columns:
            continue
        norm_col = NORMALIZE_BY[t]
        norm_vals = out[norm_col].clip(lower=_EPS)
        out[f"log_{t}"] = np.log10((out[t] / norm_vals).clip(lower=_EPS))
    return out


# Metrics
def _metrics(y_true_log: np.ndarray, y_pred_log: np.ndarray,
             y_true_orig: np.ndarray,
             denorm_factors: np.ndarray | None = None) -> dict[str, float]:
    # y_pred_log is in normalised log space; multiply back by denorm to get physical units
    if denorm_factors is not None:
        y_pred_orig = 10.0 ** y_pred_log * denorm_factors
    else:
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


# Optuna objective
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
            "max_depth":         trial.suggest_int("max_depth", 8, 16),
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


# Train one model
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
    denorm_val: np.ndarray | None = None,
    denorm_test: np.ndarray | None = None,
) -> tuple[lgb.LGBMRegressor, dict, dict]:
    """Optimise and train a LightGBM model for a single (normalised log) target."""
    print(f"\n{'='*60}")
    print(f"Target: {target_name}")
    print(f"{'='*60}")

    cv = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    # Build monotonicity constraint vector (must match feature_cols order)
    mono_constraints = [_MONO_CONSTRAINTS.get(c, 0) for c in feature_cols]

    # Optuna search
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
        # Default parameters when Optuna is disabled — loaded from YAML config
        import yaml as _yaml
        _default_params_file = Path(
            os.getenv("DEFAULT_PARAMS_PATH", "configs/default_hyperparameters.yaml")
        )
        if _default_params_file.exists():
            with open(_default_params_file, encoding="utf-8") as _f:
                best_params = _yaml.safe_load(_f)
            print(f"  Loaded default params from {_default_params_file}")
        else:
            best_params = {
                "n_estimators": 1000, "learning_rate": 0.05,
                "num_leaves": 127,    "max_depth": 10,
                "min_child_samples": 20, "subsample": 0.8,
                "colsample_bytree": 0.8, "reg_alpha": 0.1, "reg_lambda": 1.0,
            }

    # Final model on full training set
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

    # Evaluation (de-normalise predictions to original physical units)
    val_metrics  = _metrics(y_val_log,  model.predict(X_val),  y_val_orig,  denorm_val)
    test_metrics = _metrics(y_test_log, model.predict(X_test), y_test_orig, denorm_test)

    print(f"\n  Val  R²(log)={val_metrics['r2_log']:.4f}  "
          f"RMSE(log)={val_metrics['rmse_log']:.5f}  "
          f"R²(orig)={val_metrics['r2_orig']:.4f}  "
          f"MAPE={val_metrics['mape']:.4f}")
    print(f"  Test R²(log)={test_metrics['r2_log']:.4f}  "
          f"RMSE(log)={test_metrics['rmse_log']:.5f}  "
          f"R²(orig)={test_metrics['r2_orig']:.4f}  "
          f"MAPE={test_metrics['mape']:.4f}")

    return model, val_metrics, test_metrics


# Feature importance summary
def _print_top_features(model: lgb.LGBMRegressor, feature_cols: list[str],
                         target_name: str, top_n: int = 15) -> None:
    imp = pd.Series(model.feature_importances_, index=feature_cols)
    imp = imp.sort_values(ascending=False).head(top_n)
    print(f"\n  Top {top_n} features for {target_name}:")
    for feat, val in imp.items():
        print(f"    {feat:30s}: {val:6.0f}")


def _export_feature_importance(
    model: lgb.LGBMRegressor,
    feature_cols: list[str],
    target_name: str,
    out_dir: Path,
) -> tuple[Path, Path]:
    """Export feature importance (split and gain) to CSV files.

    Returns (split_path, gain_path).
    """
    booster    = model.booster_
    split_vals = booster.feature_importance(importance_type="split")
    gain_vals  = booster.feature_importance(importance_type="gain")

    split_df = pd.DataFrame(
        {"feature": feature_cols, "importance": split_vals}
    ).sort_values("importance", ascending=False)
    gain_df = pd.DataFrame(
        {"feature": feature_cols, "importance": gain_vals}
    ).sort_values("importance", ascending=False)

    split_path = out_dir / f"feature_importance_split_{target_name}.csv"
    gain_path  = out_dir / f"feature_importance_gain_{target_name}.csv"
    split_df.to_csv(split_path, index=False)
    gain_df.to_csv(gain_path, index=False)

    logger.info("Feature importance saved: %s, %s", split_path.name, gain_path.name)
    return split_path, gain_path


# Main training function
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
    use_minio: bool = False,
) -> None:
    # Initialise all RNG sources for full determinism
    _set_global_seed(random_state)

    # Load splits
    if use_minio:
        _download_splits_from_minio(data_dir)
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

    # Normalize and log-transform targets
    train_df = _normalize_and_log_targets(train_df)
    val_df   = _normalize_and_log_targets(val_df)
    test_df  = _normalize_and_log_targets(test_df)

    # Encode categorical variables
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

    logger.info("Train: %d  Val: %d  Test: %d  Features: %d",
                len(X_train), len(X_val), len(X_test), len(feature_cols))
    print(f"Train: {len(X_train):,}  Val: {len(X_val):,}  Test: {len(X_test):,}")
    print(f"Features: {len(feature_cols)}")

    out_dir.mkdir(parents=True, exist_ok=True)

    all_metrics: dict[str, dict] = {}
    artifacts: dict[str, Path]   = {}
    fi_paths:   list[Path]       = []  # feature importance CSVs (logged but not registered)

    # Train one model per target
    for target in TARGET_COLS:
        log_target   = f"log_{target}"
        normalize_by = NORMALIZE_BY[target]

        y_train = train_df[log_target].to_numpy()
        y_val   = val_df[log_target].to_numpy()
        y_test  = test_df[log_target].to_numpy()
        y_val_orig  = val_df[target].to_numpy()
        y_test_orig = test_df[target].to_numpy()

        # Denorm factors for original-scale evaluation (traction_pa or delta_theory)
        denorm_val  = val_df[normalize_by].to_numpy()
        denorm_test = test_df[normalize_by].to_numpy()

        model, val_m, test_m = _train_one_target(
            target_name=target,
            X_train=X_train, y_train_log=y_train,
            X_val=X_val,     y_val_log=y_val,   y_val_orig=y_val_orig,
            X_test=X_test,   y_test_log=y_test, y_test_orig=y_test_orig,
            feature_cols=feature_cols,
            n_trials=n_trials,
            cv_folds=cv_folds,
            random_state=random_state,
            denorm_val=denorm_val,
            denorm_test=denorm_test,
        )
        _print_top_features(model, feature_cols, target)
        split_p, gain_p = _export_feature_importance(model, feature_cols, target, out_dir)
        fi_paths.extend([split_p, gain_p])

        all_metrics[target] = {"val": val_m, "test": test_m}

        model_path = out_dir / f"lgbm_{target}.joblib"
        joblib.dump({"model": model, "feature_cols": feature_cols,
                     "encoder": enc, "target": target, "log_target": log_target,
                     "normalize_by": normalize_by},
                    model_path)
        artifacts[target] = model_path
        print(f"\n  Model saved: {model_path}")

    # Summary
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
    logger.info("Metrics saved: %s", metrics_path)
    print(f"\nMetrics saved: {metrics_path}")

    # MLflow logging
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
            for p in fi_paths:
                mlflow.log_artifact(str(p))
            mlflow.set_tags({
                "model_family":   "lightgbm",
                "task":           "regression_log_transform",
                "log_transform":  "log10",
                "normalize_by":   "traction_pa/delta_theory",
            })

            # Enregistrer dans le MLflow Model Registry → onglet "Models"
            run_id = run.info.run_id
            for target, path in artifacts.items():
                model_uri = f"runs:/{run_id}/{path.name}"
                reg_name  = f"fem-surrogate-{target.replace('_', '-')}"
                try:
                    mv = mlflow.register_model(model_uri, reg_name)
                    print(f"  Registered: {reg_name}  version={mv.version}")
                except Exception as e:
                    print(f"  [warn] Model Registry non disponible: {e}")

            logger.info("MLflow run logged: %s", run_id)
            print(f"MLflow run logged: {run_id}")


# CLI
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the advanced LightGBM surrogate with log transformation and Optuna optimisation."
    )
    parser.add_argument("--data-dir",  type=Path, default=Path("data/processed"))
    parser.add_argument("--out-dir",   type=Path, default=Path("data/models/advanced"))
    parser.add_argument("--n-trials",  type=int,  default=60,
                        help="Number of Optuna trials per target (0 = use default values).")
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
