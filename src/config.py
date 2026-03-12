"""Chargeur de configuration YAML avec validation.

Utilisation
-----------
    from src.config import load_config

    cfg = load_config()            # lit configs/training.yaml
    cfg = load_config("my.yaml")   # chemin explicite
    # ou définir la variable d'env :  CONFIG_PATH=my.yaml

Tous les paramètres du pipeline ont des valeurs par défaut raisonnables,
le fichier de configuration est donc optionnel pour les scripts rapides,
mais requis pour les runs de production reproductibles.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

try:
    import yaml
except ImportError as exc:
    raise ImportError("PyYAML is required: pip install pyyaml") from exc

_DEFAULT_CONFIG = Path("configs/training.yaml")


# ── Dataclasses de section ────────────────────────────────────────────────────

@dataclass
class DataConfig:
    raw_dir: Path = Path("data/raw")
    processed_dir: Path = Path("data/processed")
    features_dir: Path = Path("data/features/advanced")

    def __post_init__(self) -> None:
        self.raw_dir = Path(self.raw_dir)
        self.processed_dir = Path(self.processed_dir)
        self.features_dir = Path(self.features_dir)


@dataclass
class FeaturesConfig:
    seed: int = 42
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    split_strategy: str = "hash"
    keep_ambiguous: bool = False

    def __post_init__(self) -> None:
        if not (0.0 < self.train_ratio < 1.0):
            raise ValueError(f"train_ratio must be in (0, 1), got {self.train_ratio}")
        if not (0.0 <= self.val_ratio < 1.0):
            raise ValueError(f"val_ratio must be in [0, 1), got {self.val_ratio}")
        if self.train_ratio + self.val_ratio >= 1.0:
            raise ValueError(
                f"train_ratio + val_ratio must be < 1.0, "
                f"got {self.train_ratio} + {self.val_ratio} = {self.train_ratio + self.val_ratio}"
            )
        if self.split_strategy not in ("hash", "random"):
            raise ValueError(
                f"split_strategy must be 'hash' or 'random', got {self.split_strategy!r}"
            )


@dataclass
class TrainingConfig:
    random_state: int = 42
    cv_folds: int = 5
    n_trials: int = 60

    def __post_init__(self) -> None:
        if self.cv_folds < 2:
            raise ValueError(f"cv_folds must be >= 2, got {self.cv_folds}")
        if self.n_trials < 0:
            raise ValueError(f"n_trials must be >= 0, got {self.n_trials}")


@dataclass
class ArtifactsConfig:
    registry_dir: Path = Path("artifacts/models")
    model_name: str = "lgbm_surrogate"

    def __post_init__(self) -> None:
        self.registry_dir = Path(self.registry_dir)
        if not self.model_name:
            raise ValueError("artifacts.model_name cannot be empty")


@dataclass
class EvaluationConfig:
    metrics: List[str] = field(default_factory=lambda: [
        "r2_log", "rmse_log", "mae_log", "r2_orig", "rmse_orig", "mape"
    ])
    splits: List[str] = field(default_factory=lambda: ["val", "test"])

    def __post_init__(self) -> None:
        _valid = {"r2_log", "rmse_log", "mae_log", "r2_orig", "rmse_orig", "mape"}
        bad = set(self.metrics) - _valid
        if bad:
            raise ValueError(f"Unknown evaluation metrics: {sorted(bad)}. Valid: {sorted(_valid)}")
        if not self.splits:
            raise ValueError("evaluation.splits cannot be empty")


@dataclass
class PipelineConfig:
    data: DataConfig = field(default_factory=DataConfig)
    features: FeaturesConfig = field(default_factory=FeaturesConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    artifacts: ArtifactsConfig = field(default_factory=ArtifactsConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)


# ── Chargeur ─────────────────────────────────────────────────────────────────

def load_config(path: str | Path | None = None) -> PipelineConfig:
    """Charge et valide la configuration YAML du pipeline.

    Args:
        path: Chemin vers le fichier de configuration YAML.
              Par défaut : configs/training.yaml ou la variable CONFIG_PATH.

    Returns:
        PipelineConfig validé.

    Raises:
        FileNotFoundError: Le fichier de configuration n'existe pas.
        ValueError: La configuration contient des valeurs invalides.
        yaml.YAMLError: Le fichier de configuration n'est pas un YAML valide.
    """
    if path is None:
        path = Path(os.environ.get("CONFIG_PATH", str(_DEFAULT_CONFIG)))
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Fichier de configuration introuvable : {path}\n"
            f"Conseil : copiez configs/training.yaml pour démarrer, "
            f"ou définissez la variable CONFIG_PATH."
        )

    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}

    if not isinstance(raw, dict):
        raise ValueError(f"Le fichier de configuration doit être un mapping YAML, obtenu {type(raw).__name__} : {path}")

    try:
        return PipelineConfig(
            data=DataConfig(**raw.get("data", {})),
            features=FeaturesConfig(**raw.get("features", {})),
            training=TrainingConfig(**raw.get("training", {})),
            artifacts=ArtifactsConfig(**raw.get("artifacts", {})),
            evaluation=EvaluationConfig(**raw.get("evaluation", {})),
        )
    except TypeError as exc:
        raise ValueError(f"La configuration contient une clé inconnue : {exc}") from exc
