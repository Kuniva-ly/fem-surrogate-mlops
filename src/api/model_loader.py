"""Singleton model loader for the FEM Surrogate API.

Loads both target models once at startup and caches them in memory.
Safe for concurrent requests (joblib artifacts are read-only after loading).
"""
from __future__ import annotations

import logging
import os
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import joblib

logger = logging.getLogger(__name__)

TARGET_COLS = ["max_displacement_m", "max_von_mises_pa"]
_EPS = 1e-12


@dataclass
class LoadedModel:
    target: str
    model: object
    feature_cols: list[str]
    encoder: object        # OrdinalEncoder or None
    normalize_by: str | None = None  # feature column used to normalise the log target at training


@dataclass
class ModelBundle:
    """Container for all loaded models and shared metadata."""
    version: str
    model_dir: Path
    models: dict[str, LoadedModel] = field(default_factory=dict)

    @property
    def feature_count(self) -> int:
        if not self.models:
            return 0
        return len(next(iter(self.models.values())).feature_cols)


_bundle: Optional[ModelBundle] = None


def load_models(model_dir: Path | str | None = None) -> ModelBundle:
    """Load both surrogate models from *model_dir*.

    Uses the latest registered version if model_dir is not provided.
    Raises RuntimeError if no model can be located.
    """
    global _bundle

    if model_dir is None:
        model_dir = _resolve_model_dir()

    model_dir = Path(model_dir)
    if not model_dir.exists():
        raise RuntimeError(f"Model directory not found: {model_dir}")

    version = model_dir.name
    bundle = ModelBundle(version=version, model_dir=model_dir)

    for target in TARGET_COLS:
        artifact_path = model_dir / f"lgbm_{target}.joblib"
        if not artifact_path.exists():
            logger.warning("Model artifact not found for target '%s' at %s", target, artifact_path)
            continue
        try:
            artifact = joblib.load(artifact_path)
            bundle.models[target] = LoadedModel(
                target=target,
                model=artifact["model"],
                feature_cols=artifact["feature_cols"],
                encoder=artifact.get("encoder"),
                normalize_by=artifact.get("normalize_by"),
            )
            logger.info("Loaded model for '%s' from %s", target, artifact_path)
        except Exception as exc:
            logger.error("Failed to load model for '%s': %s", target, exc)
            raise RuntimeError(f"Model load failed for {target}: {exc}") from exc

    if not bundle.models:
        raise RuntimeError(f"No model artifacts found in {model_dir}")

    _bundle = bundle
    logger.info("Model bundle ready — version=%s, targets=%s", version, list(bundle.models))
    return bundle


def get_bundle() -> Optional[ModelBundle]:
    """Return the currently loaded bundle (None if not yet loaded)."""
    return _bundle


def _resolve_model_dir() -> Path:
    """Resolve the model directory.

    Priority:
    1. MODEL_DIR env var (explicit override)
    2. MLflow tracking server — downloads latest run artifacts from MinIO
    3. Local registry latest.txt (local dev fallback)
    """
    env_path = os.environ.get("MODEL_DIR")
    if env_path:
        return Path(env_path)

    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if tracking_uri:
        try:
            return _download_from_mlflow(tracking_uri)
        except Exception as exc:
            logger.warning("MLflow download failed, falling back to local registry: %s", exc)

    registry_dir = Path(os.environ.get("REGISTRY_DIR", "artifacts/models"))
    model_name   = os.environ.get("MODEL_NAME", "lgbm_surrogate")
    latest_file  = registry_dir / model_name / "latest.txt"
    if latest_file.exists():
        version = latest_file.read_text(encoding="utf-8").strip()
        return registry_dir / model_name / version

    raise RuntimeError(
        "Cannot locate model directory. "
        "Set MODEL_DIR, point MLFLOW_TRACKING_URI to a server with runs, "
        "or ensure artifacts/models/lgbm_surrogate/latest.txt exists."
    )


def _download_from_mlflow(tracking_uri: str) -> Path:
    """Download the latest run's .joblib artifacts from MLflow/MinIO to a temp dir.

    Retries up to MLFLOW_DOWNLOAD_RETRIES times (default 3) with exponential
    backoff. Timeout is set via MLFLOW_HTTP_REQUEST_TIMEOUT (default 30 s).
    """
    import mlflow
    from mlflow.tracking import MlflowClient

    max_retries = int(os.environ.get("MLFLOW_DOWNLOAD_RETRIES", "3"))
    timeout_s   = int(os.environ.get("MLFLOW_HTTP_REQUEST_TIMEOUT", "30"))
    os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", str(timeout_s))

    experiment_name = os.environ.get("MLFLOW_EXPERIMENT_NAME", "fem-surrogate")
    run_id = os.environ.get("MLFLOW_RUN_ID")

    client = MlflowClient(tracking_uri=tracking_uri)

    if not run_id:
        experiment = client.get_experiment_by_name(experiment_name)
        if experiment is None:
            raise RuntimeError(f"MLflow experiment '{experiment_name}' not found")
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["start_time DESC"],
            max_results=1,
        )
        if not runs:
            raise RuntimeError(f"No runs found in experiment '{experiment_name}'")
        run_id = runs[0].info.run_id

    cache_root = Path(os.environ.get("MODEL_CACHE_DIR", tempfile.gettempdir())) / "fem_models"
    dst = cache_root / run_id
    if dst.exists():
        logger.info("Using cached MLflow artifacts — run=%s @ %s", run_id, dst)
        return dst

    dst.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, max_retries + 1):
        try:
            logger.info(
                "Downloading MLflow artifacts (attempt %d/%d) — run=%s timeout=%ds",
                attempt, max_retries, run_id, timeout_s,
            )
            client.download_artifacts(run_id, ".", str(dst))
            logger.info("Downloaded MLflow artifacts — run=%s → %s", run_id, dst)
            return dst
        except Exception as exc:
            if attempt == max_retries:
                logger.error(
                    "MLflow download failed after %d attempts: %s", max_retries, exc
                )
                raise
            wait = 2 ** attempt
            logger.warning(
                "MLflow download attempt %d/%d failed: %s — retrying in %ds",
                attempt, max_retries, exc, wait,
            )
            time.sleep(wait)

    raise RuntimeError("Unreachable")
