"""Point d'entrée CLI pour le pipeline ML surrogate FEM.

Commandes
---------
  build-features    Calcule les features et crée les splits train/val/test.
  train             Entraîne le modèle surrogate LightGBM avancé.
  evaluate          Évalue un modèle entraîné sur les splits val/test.
  predict           Lance l'inférence sur un cas unique.
  verify-artifacts  Vérifie l'intégrité SHA-256 des artefacts enregistrés.

Utilisation (Windows PowerShell)
---------------------------------
  .venv\\Scripts\\python -m src.cli build-features --input data/raw
  .venv\\Scripts\\python -m src.cli train --n-trials 60
  .venv\\Scripts\\python -m src.cli evaluate
  .venv\\Scripts\\python -m src.cli predict --case-json '{"length_m": 1.2, ...}'
  .venv\\Scripts\\python -m src.cli verify-artifacts

Configuration
-------------
  Toutes les commandes lisent configs/training.yaml par défaut.
  Remplacer avec --config <chemin> ou la variable d'environnement CONFIG_PATH.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


# ── Gestionnaires de sous-commandes ──────────────────────────────────────────────

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

    # MLflow : activé via --mlflow ou la variable MLFLOW_TRACKING_URI
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
        mlflow_run_name=args.mlflow_run_name if hasattr(args, "mlflow_run_name") else None,
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

    # ── Registre ──────────────────────────────────────────────────────────────
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

    # Résoudre le répertoire du modèle
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

    # Résoudre le répertoire du modèle
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

    # Charger le cas d'entrée
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

        # Calculer automatiquement les features si des paramètres bruts ont été fournis
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


# ── Parseur d'arguments ──────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.cli",
        description="Pipeline ML Surrogate FEM — CLI de production",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Exemples:\n"
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
        help="Fichier de configuration YAML (défaut : configs/training.yaml ou $CONFIG_PATH)",
    )
    sub = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    # ── build-features ────────────────────────────────────────────────────────
    p = sub.add_parser(
        "build-features",
        help="Calcule les features physiques et crée les splits train/val/test.",
    )
    p.add_argument("--input",            metavar="DIR",  help="Répertoire d'entrée parquet brut (remplace la config)")
    p.add_argument("--out-dir",          metavar="DIR",  help="Répertoire de sortie des splits traités")
    p.add_argument("--features-out-dir", metavar="DIR",  help="Répertoire de sortie du feature store")
    p.add_argument("--train-ratio",      type=float, default=None, metavar="FLOAT")
    p.add_argument("--val-ratio",        type=float, default=None, metavar="FLOAT")
    p.add_argument("--seed",             type=int,   default=None, metavar="INT")
    p.add_argument("--split-strategy",   choices=["hash", "random"], default=None)
    p.add_argument("--keep-ambiguous",   action="store_true", default=False,
                   help="Traiter les lignes ambiguës comme without_hole au lieu de les supprimer")

    # ── train ─────────────────────────────────────────────────────────────────
    p = sub.add_parser(
        "train",
        help="Entraîne le surrogate LightGBM avancé avec l'optimisation Optuna.",
    )
    p.add_argument("--data-dir",     metavar="DIR", help="Répertoire contenant les fichiers .parquet train/val/test")
    p.add_argument("--out-dir",      metavar="DIR", help="Répertoire de sortie pour les artefacts du modèle")
    p.add_argument("--n-trials",     type=int, default=None, metavar="INT",
                   help="Essais Optuna par cible (0 = utiliser les paramètres par défaut, ignorer la recherche)")
    p.add_argument("--cv-folds",     type=int, default=None, metavar="INT")
    p.add_argument("--random-state", type=int, default=None, metavar="INT")
    p.add_argument("--no-registry",  action="store_true",
                   help="Ne pas enregistrer les artefacts dans le registre de modèles")
    p.add_argument("--mlflow",        action="store_true",
                   help="Enregistrer l'expérience dans MLflow (activé automatiquement si MLFLOW_TRACKING_URI est défini)")
    p.add_argument("--mlflow-tracking-uri", metavar="URI",
                   help="URI de suivi MLflow (défaut : $MLFLOW_TRACKING_URI)")
    p.add_argument("--mlflow-experiment", metavar="NAME",
                   help="Nom de l'expérience MLflow (défaut : $MLFLOW_EXPERIMENT_NAME ou 'fem-surrogate')")
    p.add_argument("--mlflow-run-name", metavar="NAME",
                   help="Nom du run MLflow (défaut : auto)")

    # ── evaluate ──────────────────────────────────────────────────────────────
    p = sub.add_parser(
        "evaluate",
        help="Évalue le modèle entraîné sur les splits val/test (utilise la dernière version du registre par défaut).",
    )
    p.add_argument("--model-dir", metavar="DIR", help="Répertoire de version du modèle (remplace la recherche dans le registre)")
    p.add_argument("--data-dir",  metavar="DIR", help="Répertoire contenant les fichiers parquet des splits")
    p.add_argument("--out-dir",   metavar="DIR", help="Répertoire pour les fichiers de sortie de l'évaluation")

    # ── predict ───────────────────────────────────────────────────────────────
    p = sub.add_parser(
        "predict",
        help="Lance l'inférence sur un cas unique (calcule automatiquement les features depuis les paramètres bruts).",
    )
    p.add_argument("--model-dir",  metavar="DIR",  help="Répertoire du modèle (défaut : dernière version du registre)")
    p.add_argument("--case-json",  metavar="JSON", help="Dict JSON en ligne pour le cas d'entrée")
    p.add_argument("--case-file",  metavar="FILE", help="Chemin vers un fichier JSON contenant les paramètres du cas")

    # ── verify-artifacts ──────────────────────────────────────────────────────
    p = sub.add_parser(
        "verify-artifacts",
        help="Vérifie les checksums SHA-256 des artefacts du modèle.",
    )
    p.add_argument("--checksums-path", metavar="PATH",
                   help="Chemin du fichier de checksums sans extension (défaut : dernière version du registre)")

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
