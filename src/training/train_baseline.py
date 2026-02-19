import argparse
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

try:
    import mlflow
except ImportError: 
    mlflow = None


TARGET_COLS = ["max_displacement_m", "max_von_mises_pa"]
DROP_COLS = ["simulation_id", "timestamp", "solver_name", "solver_version", "data_version"]


def _load_split(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing split file: {path}")
    return pd.read_parquet(path)


def _feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in TARGET_COLS + DROP_COLS]


def _infer_model_variant(train_df: pd.DataFrame) -> str:
    if "geometry_type" in train_df.columns:
        gt = train_df["geometry_type"].astype("string").str.strip()
        values = sorted(v for v in gt.dropna().unique().tolist() if v)
        if len(values) == 1:
            return str(values[0])
        if len(values) > 1:
            return "mixed"

    has_radius = "hole_radius_ratio" in train_df.columns and (pd.to_numeric(train_df["hole_radius_ratio"], errors="coerce") > 0).any()
    has_moving = (
        "hole_cx_ratio" in train_df.columns
        and "hole_cy_ratio" in train_df.columns
        and pd.to_numeric(train_df["hole_cx_ratio"], errors="coerce").notna().any()
        and pd.to_numeric(train_df["hole_cy_ratio"], errors="coerce").notna().any()
    )
    if has_moving:
        return "with_hole_moving"
    if has_radius:
        return "with_hole"
    return "without_hole"


def _build_pipeline(X_train: pd.DataFrame, n_estimators: int, random_state: int) -> Pipeline:
    cat_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = [c for c in X_train.columns if c not in cat_cols]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", "passthrough", num_cols),
        ]
    )
    model = MultiOutputRegressor(
        RandomForestRegressor(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1,
        )
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


def _evaluate(name: str, y_true: pd.DataFrame, y_pred: np.ndarray) -> pd.DataFrame:
    rows = []
    for i, target in enumerate(TARGET_COLS):
        yt = y_true[target].to_numpy()
        yp = y_pred[:, i]
        rows.append(
            {
                "split": name,
                "target": target,
                "mae": float(mean_absolute_error(yt, yp)),
                "rmse": float(np.sqrt(mean_squared_error(yt, yp))),
                "r2": float(r2_score(yt, yp)),
            }
        )
    return pd.DataFrame(rows)


def _metrics_to_mlflow(metrics: pd.DataFrame) -> dict[str, float]:
    mlflow_metrics: dict[str, float] = {}
    for row in metrics.itertuples(index=False):
        target_name = row.target.replace(" ", "_")
        mlflow_metrics[f"{row.split}_{target_name}_mae"] = float(row.mae)
        mlflow_metrics[f"{row.split}_{target_name}_rmse"] = float(row.rmse)
        mlflow_metrics[f"{row.split}_{target_name}_r2"] = float(row.r2)
    return mlflow_metrics


def train_baseline(
    data_dir: Path,
    out_dir: Path,
    n_estimators: int = 300,
    random_state: int = 42,
    mlflow_enabled: bool = False,
    mlflow_tracking_uri: str | None = None,
    mlflow_experiment: str = "baseline-surrogate",
    mlflow_run_name: str | None = None,
    mlflow_log_artifacts: bool = True,
) -> None:
    train_df = _load_split(data_dir / "train.parquet")
    val_df = _load_split(data_dir / "val.parquet")
    test_df = _load_split(data_dir / "test.parquet")

    feature_cols = _feature_columns(train_df)
    if not feature_cols:
        raise ValueError("No feature columns available for training after dropping metadata/targets.")
    for target in TARGET_COLS:
        if target not in train_df.columns:
            raise ValueError(f"Missing target column '{target}' in train split")

    X_train, y_train = train_df[feature_cols], train_df[TARGET_COLS]
    X_val, y_val = val_df[feature_cols], val_df[TARGET_COLS]
    X_test, y_test = test_df[feature_cols], test_df[TARGET_COLS]
    model_variant = _infer_model_variant(train_df)

    if X_train.isnull().any().any() or X_val.isnull().any().any() or X_test.isnull().any().any():
        raise ValueError("Null values detected in feature columns. Clean/impute data before training.")
    if y_train.isnull().any().any() or y_val.isnull().any().any() or y_test.isnull().any().any():
        raise ValueError("Null values detected in target columns.")

    pipeline = _build_pipeline(X_train, n_estimators=n_estimators, random_state=random_state)
    pipeline.fit(X_train, y_train)

    val_pred = pipeline.predict(X_val)
    test_pred = pipeline.predict(X_test)
    metrics = pd.concat(
        [
            _evaluate("val", y_val, val_pred),
            _evaluate("test", y_test, test_pred),
        ],
        ignore_index=True,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "baseline_model.joblib"
    metrics_path = out_dir / "baseline_metrics.csv"
    features_path = out_dir / "baseline_features.txt"

    joblib.dump(pipeline, model_path)
    metrics.to_csv(metrics_path, index=False)
    features_path.write_text("\n".join(feature_cols), encoding="utf-8")
    print(f"Model saved: {model_path}")
    print(f"Metrics saved: {metrics_path}")
    print(f"Features saved: {features_path}")
    print("Metrics:")
    print(metrics.to_string(index=False))

    if mlflow_enabled:
        if mlflow is None:
            raise ImportError("MLflow is not installed. Install it with: pip install mlflow")

        # Hide only "View run / View experiment" stdout lines (with emoji),
        # while keeping MLflow logs enabled.
        os.environ["MLFLOW_SUPPRESS_PRINTING_URL_TO_STDOUT"] = "true"

        if mlflow_tracking_uri:
            mlflow.set_tracking_uri(mlflow_tracking_uri)
        mlflow.set_experiment(mlflow_experiment)

        with mlflow.start_run(run_name=mlflow_run_name) as run:
            mlflow.log_params(
                {
                    "n_estimators": int(n_estimators),
                    "random_state": int(random_state),
                    "n_train_rows": int(len(X_train)),
                    "n_val_rows": int(len(X_val)),
                    "n_test_rows": int(len(X_test)),
                    "n_features": int(len(feature_cols)),
                    "data_dir": str(data_dir),
                }
            )
            mlflow.log_metrics(_metrics_to_mlflow(metrics))
            if mlflow_log_artifacts:
                try:
                    import boto3  # noqa: F401  # required by S3 artifact repository
                except ImportError as exc:
                    raise ImportError(
                        "Missing dependency 'boto3' for MLflow artifact upload to S3. "
                        "Install with: pip install boto3 "
                        "or rerun with --no-mlflow-log-artifacts."
                    ) from exc
                mlflow.log_artifact(str(model_path))
                mlflow.log_artifact(str(metrics_path))
                mlflow.log_artifact(str(features_path))
            mlflow.set_tags(
                {
                    "model_family": "random_forest",
                    "task": "multioutput_regression",
                    "model_variant": model_variant,
                }
            )
            print(f"MLflow run logged: {run.info.run_id}")
            if mlflow_tracking_uri:
                print(f"MLflow tracking URI: {mlflow_tracking_uri}")
            print(f"MLflow experiment: {mlflow_experiment}")

def main() -> None:
    parser = argparse.ArgumentParser(description="Train baseline surrogate model on processed split parquet files.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"), help="Folder containing train/val/test parquet")
    parser.add_argument("--out-dir", type=Path, default=Path("data/processed"), help="Output folder for model and metrics")
    parser.add_argument("--n-estimators", type=int, default=300, help="Number of trees for RandomForest")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed")
    parser.add_argument("--mlflow", action="store_true", help="Enable MLflow logging")
    parser.add_argument(
        "--mlflow-tracking-uri",
        default=os.getenv("MLFLOW_TRACKING_URI"),
        help="MLflow tracking URI (default: env MLFLOW_TRACKING_URI)",
    )
    parser.add_argument(
        "--mlflow-experiment",
        default=os.getenv("MLFLOW_EXPERIMENT_NAME", "baseline-surrogate"),
        help="MLflow experiment name",
    )
    parser.add_argument("--mlflow-run-name", default=None, help="Optional MLflow run name")
    parser.add_argument(
        "--mlflow-log-artifacts",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Upload local model/metrics/features files as MLflow artifacts",
    )
    args = parser.parse_args()

    train_baseline(
        data_dir=args.data_dir,
        out_dir=args.out_dir,
        n_estimators=args.n_estimators,
        random_state=args.random_state,
        mlflow_enabled=args.mlflow,
        mlflow_tracking_uri=args.mlflow_tracking_uri,
        mlflow_experiment=args.mlflow_experiment,
        mlflow_run_name=args.mlflow_run_name,
        mlflow_log_artifacts=args.mlflow_log_artifacts,
    )


if __name__ == "__main__":
    main()


