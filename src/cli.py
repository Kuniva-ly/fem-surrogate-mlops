"""CLI entry point for the FEM surrogate ML pipeline.

Commands
--------
  build-features    Computes features and creates train/val/test splits.
  train             Trains the advanced LightGBM surrogate model.
  evaluate          Evaluates a trained model on val/test splits.
  predict           Runs inference on a single case.
  verify-artifacts  Verifies the SHA-256 integrity of registered artifacts.

Usage (Windows PowerShell)
--------------------------
  .venv\\Scripts\\python -m src.cli build-features --input data/raw
  .venv\\Scripts\\python -m src.cli train --n-trials 60
  .venv\\Scripts\\python -m src.cli evaluate
  .venv\\Scripts\\python -m src.cli predict --case-json '{"length_m": 1.2, ...}'
  .venv\\Scripts\\python -m src.cli verify-artifacts

Configuration
-------------
  All commands read configs/training.yaml by default.
  Override with --config <path> or the CONFIG_PATH environment variable.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


# ── Sub-command handlers ─────────────────────────────────────────────────────────

def _cmd_build_features(args: argparse.Namespace) -> int:
    from src.config import load_config
    from src.processing.build_features import build_features

    cfg = load_config(args.config)
    f, d = cfg.features, cfg.data

    build_features(
        input_dir=Path(args.input) if args.input else d.raw_dir,
        out_dir=Path(args.out_dir) if args.out_dir else d.processed_dir,
        train_ratio=args.train_ratio if args.train_ratio is not None else f.train_ratio,
        val_ratio=args.val_ratio if args.val_ratio is not None else f.val_ratio,
        seed=args.seed if args.seed is not None else f.seed,
        split_strategy=args.split_strategy or f.split_strategy,
        features_out_dir=Path(args.features_out_dir) if args.features_out_dir else d.features_dir,
        keep_ambiguous_as_without_hole=args.keep_ambiguous or f.keep_ambiguous,
    )
    return 0


def _cmd_train(args: argparse.Namespace) -> int:
    import os

    from src.config import load_config
    from src.training.train_advanced import train_advanced
    from src.utils.integrity import generate_checksums, save_checksums
    from src.utils.manifest import build_manifest, save_manifest
    from src.registry import ModelRegistry

    cfg = load_config(args.config)
    tr, art, d = cfg.training, cfg.artifacts, cfg.data

    n_trials     = args.n_trials     if args.n_trials     is not None else tr.n_trials
    cv_folds     = args.cv_folds     if args.cv_folds     is not None else tr.cv_folds
    random_state = args.random_state if args.random_state is not None else tr.random_state

    data_dir = Path(args.data_dir) if args.data_dir else d.processed_dir
    out_dir  = Path(args.out_dir)  if args.out_dir  else d.features_dir.parent / "models_run"

    # MLflow: enabled via --mlflow or the MLFLOW_TRACKING_URI environment variable
    # mlflow_enabled = cfg.get("mlflow", {}).get("enabled", False)
    # mlflow_uri     = args.mlflow_tracking_uri or cfg.get("mlflow", {}).get("tracking_uri")
    # mlflow_exp     = args.mlflow_experiment   or cfg.get("mlflow", {}).get("experiment", "fem-surrogate")

    mlflow_enabled = args.mlflow or bool(os.environ.get("MLFLOW_TRACKING_URI"))
    mlflow_uri     = args.mlflow_tracking_uri or os.environ.get("MLFLOW_TRACKING_URI")
    mlflow_exp     = args.mlflow_experiment or os.environ.get("MLFLOW_EXPERIMENT_NAME", "fem-surrogate")

    train_advanced(
        data_dir=data_dir,
        out_dir=out_dir,
        n_trials=n_trials,
        cv_folds=cv_folds,
        random_state=random_state,
        mlflow_enabled=mlflow_enabled,
        mlflow_tracking_uri=mlflow_uri,
        mlflow_experiment=mlflow_exp,
        mlflow_run_name=args.mlflow_run_name,
    )

    # ── Manifest ──────────────────────────────────────────────────────────────
    dataset_files = list(data_dir.glob("*.parquet"))
    manifest = build_manifest(
        config_snapshot={
            "data_dir":     str(data_dir),
            "n_trials":     n_trials,
            "cv_folds":     cv_folds,
            "random_state": random_state,
        },
        dataset_paths=dataset_files,
    )
    save_manifest(manifest, out_dir / "manifest.json")

    # ── Checksums ─────────────────────────────────────────────────────────────
    artifact_files = (
        list(out_dir.glob("*.joblib"))
        + list(out_dir.glob("*.csv"))
        + list(out_dir.glob("*.json"))
        + list(out_dir.glob("*.txt"))
    )
    checksums = generate_checksums(artifact_files)
    save_checksums(checksums, out_dir / "checksums")

    # ── Registry ──────────────────────────────────────────────────────────────
    if not args.no_registry:
        registry = ModelRegistry(art.registry_dir, art.model_name)
        version, reg_path = registry.register(source_dir=out_dir)
        print(f"Registered   : {version}  ->  {reg_path}")

    return 0


def _cmd_evaluate(args: argparse.Namespace) -> int:
    import joblib
    import numpy as np
    import pandas as pd

    from src.config import load_config
    from src.evaluation import (
        build_metrics_df, compute_metrics, print_metrics_table, save_metrics,
    )
    from src.registry import ModelRegistry
    from src.training.train_advanced import CAT_COLS

    cfg = load_config(args.config)
    _EPS = 1e-12
    TARGET_COLS = ["max_displacement_m", "max_von_mises_pa"]

    # Resolve the model directory
    if args.model_dir:
        model_dir = Path(args.model_dir)
    else:
        registry = ModelRegistry(cfg.artifacts.registry_dir, cfg.artifacts.model_name)
        model_dir = registry.latest_path()
        if model_dir is None:
            print("ERROR: No registered model found. Run 'train' first.", file=sys.stderr)
            return 1

    data_dir = Path(args.data_dir) if args.data_dir else cfg.data.processed_dir

    all_metrics: dict = {}
    for target in TARGET_COLS:
        model_file = model_dir / f"lgbm_{target}.joblib"
        if not model_file.exists():
            print(f"SKIP: model not found for {target} at {model_file}", file=sys.stderr)
            continue

        artifact = joblib.load(model_file)
        model       = artifact["model"]
        feature_cols = artifact["feature_cols"]
        encoder     = artifact.get("encoder")

        target_metrics: dict = {}
        for split in cfg.evaluation.splits:
            split_path = data_dir / f"{split}.parquet"
            if not split_path.exists():
                print(f"SKIP: {split_path} not found", file=sys.stderr)
                continue

            df = pd.read_parquet(split_path)
            cat_present = [c for c in CAT_COLS if c in feature_cols]
            if encoder is not None and cat_present:
                df = df.copy()
                df[cat_present] = encoder.transform(df[cat_present].astype(str))

            X = df[feature_cols].astype(float)
            y_true      = df[target].to_numpy()
            y_true_log  = np.log10(np.clip(y_true, _EPS, None))
            y_pred_log  = model.predict(X)
            target_metrics[split] = compute_metrics(y_true_log, y_pred_log, y_true)

        all_metrics[target] = target_metrics

    metrics_df = build_metrics_df(all_metrics)
    print_metrics_table(metrics_df)

    out_dir = Path(args.out_dir) if args.out_dir else model_dir
    save_metrics(metrics_df, out_dir, prefix="eval")
    return 0


def _cmd_predict(args: argparse.Namespace) -> int:
    import json as _json

    import joblib
    import pandas as pd

    from src.config import load_config
    from src.processing.build_features import engineer_features
    from src.registry import ModelRegistry
    from src.training.train_advanced import CAT_COLS

    cfg = load_config(args.config)
    TARGET_COLS = ["max_displacement_m", "max_von_mises_pa"]
    _EPS = 1e-12

    # Resolve the model directory
    if args.model_dir:
        model_dir = Path(args.model_dir)
    else:
        registry = ModelRegistry(cfg.artifacts.registry_dir, cfg.artifacts.model_name)
        model_dir = registry.latest_path()
        if model_dir is None:
            print(
                "ERROR: No registered model found. Pass --model-dir or run 'train'.",
                file=sys.stderr,
            )
            return 1

    # Load the input case
    if args.case_json:
        case = _json.loads(args.case_json)
    elif args.case_file:
        case = _json.loads(Path(args.case_file).read_text(encoding="utf-8-sig"))
    else:
        print("ERROR: Provide --case-json or --case-file.", file=sys.stderr)
        return 1

    predictions: dict = {}
    for target in TARGET_COLS:
        model_file = model_dir / f"lgbm_{target}.joblib"
        if not model_file.exists():
            continue

        artifact     = joblib.load(model_file)
        model        = artifact["model"]
        feature_cols = artifact["feature_cols"]
        encoder      = artifact.get("encoder")

        # Automatically compute features if raw parameters were provided
        case_df = pd.DataFrame([case])
        if any(c not in case_df.columns for c in feature_cols):
            case_df = engineer_features(case_df, require_targets=False)

        cat_present = [c for c in CAT_COLS if c in feature_cols]
        if encoder is not None and cat_present:
            case_df = case_df.copy()
            case_df[cat_present] = encoder.transform(case_df[cat_present].astype(str))

        X = case_df[feature_cols].astype(float)
        y_pred_log = model.predict(X)
        predictions[target] = float(10.0 ** y_pred_log[0])

    output = {
        "model_dir":   str(model_dir),
        "input":       case,
        "predictions": predictions,
    }
    print(_json.dumps(output, indent=2, ensure_ascii=True))
    return 0


def _cmd_verify_artifacts(args: argparse.Namespace) -> int:
    from src.config import load_config
    from src.registry import ModelRegistry
    from src.utils.integrity import verify_checksums

    cfg = load_config(args.config)

    if args.checksums_path:
        checksums_path = Path(args.checksums_path)
    else:
        registry = ModelRegistry(cfg.artifacts.registry_dir, cfg.artifacts.model_name)
        model_dir = registry.latest_path()
        if model_dir is None:
            print("ERROR: No registered model found.", file=sys.stderr)
            return 1
        checksums_path = model_dir / "checksums"

    print(f"Verifying: {checksums_path}")
    ok, errors = verify_checksums(checksums_path)

    if ok:
        print("All artifacts OK — checksums match.")
        return 0
    else:
        print(f"VERIFICATION FAILED ({len(errors)} error(s)):", file=sys.stderr)
        for e in errors:
            print(f"  {e}", file=sys.stderr)
        return 1


# ── Argument parser ───────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.cli",
        description="FEM Surrogate ML Pipeline — production CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  .venv\\Scripts\\python -m src.cli build-features --input data/raw\n"
            "  .venv\\Scripts\\python -m src.cli train --n-trials 60\n"
            "  .venv\\Scripts\\python -m src.cli evaluate\n"
            "  .venv\\Scripts\\python -m src.cli predict "
            "--case-json '{\"length_m\":1.2,\"height_m\":0.3,"
            "\"young_modulus_pa\":2.1e11,\"poisson_ratio\":0.3,"
            "\"traction_pa\":1.5e6,\"mesh_nx\":120,\"mesh_ny\":24,"
            "\"geometry_type\":\"without_hole\"}'\n"
            "  .venv\\Scripts\\python -m src.cli verify-artifacts\n"
        ),
    )
    parser.add_argument(
        "--config",
        metavar="PATH",
        default=None,
        help="YAML configuration file (default: configs/training.yaml or $CONFIG_PATH)",
    )
    sub = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    # ── build-features ────────────────────────────────────────────────────────
    p = sub.add_parser(
        "build-features",
        help="Compute physics features and create train/val/test splits.",
    )
    p.add_argument("--input",            metavar="DIR",  help="Raw parquet input directory (overrides config)")
    p.add_argument("--out-dir",          metavar="DIR",  help="Output directory for processed splits")
    p.add_argument("--features-out-dir", metavar="DIR",  help="Output directory for the feature store")
    p.add_argument("--train-ratio",      type=float, default=None, metavar="FLOAT")
    p.add_argument("--val-ratio",        type=float, default=None, metavar="FLOAT")
    p.add_argument("--seed",             type=int,   default=None, metavar="INT")
    p.add_argument("--split-strategy",   choices=["hash", "random"], default=None)
    p.add_argument("--keep-ambiguous",   action="store_true", default=False,
                   help="Treat ambiguous rows as without_hole instead of dropping them")

    # ── train ─────────────────────────────────────────────────────────────────
    p = sub.add_parser(
        "train",
        help="Train the advanced LightGBM surrogate with Optuna optimisation.",
    )
    p.add_argument("--data-dir",     metavar="DIR", help="Directory containing train/val/test .parquet files")
    p.add_argument("--out-dir",      metavar="DIR", help="Output directory for model artifacts")
    p.add_argument("--n-trials",     type=int, default=None, metavar="INT",
                   help="Optuna trials per target (0 = use default params, skip search)")
    p.add_argument("--cv-folds",     type=int, default=None, metavar="INT")
    p.add_argument("--random-state", type=int, default=None, metavar="INT")
    p.add_argument("--no-registry",  action="store_true",
                   help="Do not register artifacts in the model registry")
    p.add_argument("--mlflow",        action="store_true",
                   help="Log experiment to MLflow (enabled automatically if MLFLOW_TRACKING_URI is set)")
    p.add_argument("--mlflow-tracking-uri", metavar="URI",
                   help="MLflow tracking URI (default: $MLFLOW_TRACKING_URI)")
    p.add_argument("--mlflow-experiment", metavar="NAME",
                   help="MLflow experiment name (default: $MLFLOW_EXPERIMENT_NAME or 'fem-surrogate')")
    p.add_argument("--mlflow-run-name", metavar="NAME",
                   help="MLflow run name (default: auto)")

    # ── evaluate ──────────────────────────────────────────────────────────────
    p = sub.add_parser(
        "evaluate",
        help="Evaluate the trained model on val/test splits (uses latest registry version by default).",
    )
    p.add_argument("--model-dir", metavar="DIR", help="Model version directory (overrides registry lookup)")
    p.add_argument("--data-dir",  metavar="DIR", help="Directory containing split parquet files")
    p.add_argument("--out-dir",   metavar="DIR", help="Directory for evaluation output files")

    # ── predict ───────────────────────────────────────────────────────────────
    p = sub.add_parser(
        "predict",
        help="Run inference on a single case (features are computed automatically from raw parameters).",
    )
    p.add_argument("--model-dir",  metavar="DIR",  help="Model directory (default: latest registry version)")
    p.add_argument("--case-json",  metavar="JSON", help="Inline JSON dict for the input case")
    p.add_argument("--case-file",  metavar="FILE", help="Path to a JSON file containing the case parameters")

    # ── verify-artifacts ──────────────────────────────────────────────────────
    p = sub.add_parser(
        "verify-artifacts",
        help="Verify SHA-256 checksums of model artifacts.",
    )
    p.add_argument("--checksums-path", metavar="PATH",
                   help="Checksums file path without extension (default: latest registry version)")

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    handlers = {
        "build-features":   _cmd_build_features,
        "train":            _cmd_train,
        "evaluate":         _cmd_evaluate,
        "predict":          _cmd_predict,
        "verify-artifacts": _cmd_verify_artifacts,
    }

    handler = handlers.get(args.command)
    if handler is None:
        parser.print_help()
        sys.exit(1)

    sys.exit(handler(args))


if __name__ == "__main__":
    main()
