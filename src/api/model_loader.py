"""Chargeur de modèle singleton pour l'API Surrogate FEM.

Charge les deux modèles cibles une seule fois au démarrage et les met en cache en mémoire.
Sûr pour les requêtes concurrentes (les artefacts joblib sont en lecture seule après chargement).
"""
from __future__ import annotations

import logging
import os
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
    encoder: object  # OrdinalEncoder or None


@dataclass
class ModelBundle:
    """Conteneur pour tous les modèles chargés et les métadonnées partagées."""
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
    """Charge les deux modèles surrogate depuis *model_dir*.

    Utilise la dernière version enregistrée si model_dir n'est pas fourni.
    Lève RuntimeError si aucun modèle ne peut être localisé.
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
    """Retourne le bundle actuellement chargé (None si pas encore chargé)."""
    return _bundle


def _resolve_model_dir() -> Path:
    """Résout le répertoire du modèle depuis la variable d'env ou le pointeur latest du registre."""
    env_path = os.environ.get("MODEL_DIR")
    if env_path:
        return Path(env_path)

    # Essayer le registre
    registry_dir = Path(os.environ.get("REGISTRY_DIR", "artifacts/models"))
    model_name   = os.environ.get("MODEL_NAME", "lgbm_surrogate")
    latest_file  = registry_dir / model_name / "latest.txt"
    if latest_file.exists():
        version = latest_file.read_text(encoding="utf-8").strip()
        return registry_dir / model_name / version

    raise RuntimeError(
        "Cannot locate model directory. "
        "Set MODEL_DIR env var or ensure artifacts/models/lgbm_surrogate/latest.txt exists."
    )
