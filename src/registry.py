"""Registre local simple de modèles.

Structure des répertoires
-------------------------
    artifacts/models/
        <model_name>/
            <version>/          ex. v20260310_143022/
                *.joblib
                feature_columns.txt
                training_config.yaml
                metrics.json
                metrics.csv
                manifest.json
                checksums.sha256
                checksums.json
            latest.txt          contient le nom du dernier répertoire de version

Format de version : ``v{AAAAMMJJ}_{HHMMSS}`` (UTC).

Utilisation
-----------
    from src.registry import ModelRegistry

    reg = ModelRegistry(Path("artifacts/models"), "lgbm_surrogate")
    version, path = reg.register(source_dir=Path("data/models/advanced"))
    print(reg.latest_version())   # "v20260310_143022"
    print(reg.list_versions())    # ["v20260310_143022"]
"""
from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

_VERSION_FMT = "%Y%m%d_%H%M%S"


def make_version(ts: datetime | None = None) -> str:
    """Génère une chaîne de version basée sur un horodatage (UTC)."""
    ts = ts or datetime.now(timezone.utc)
    return f"v{ts.strftime(_VERSION_FMT)}"


class ModelRegistry:
    """Registre de modèles sur le système de fichiers avec stockage d'artefacts versionné."""

    def __init__(self, registry_dir: Path, model_name: str) -> None:
        self.root = Path(registry_dir) / model_name
        self.model_name = model_name

    # ── Enregistrement ────────────────────────────────────────────────────────

    def register(
        self,
        source_dir: Path,
        version: str | None = None,
        metrics: dict | None = None,
    ) -> tuple[str, Path]:
        """Copie tous les fichiers de *source_dir* dans un nouvel emplacement versionné.

        Args:
            source_dir: Répertoire contenant les artefacts entraînés.
            version:    Chaîne de version à utiliser (défaut : horodatage UTC).
            metrics:    Dict de métriques optionnel sauvegardé en ``metrics.json``.

        Returns:
            ``(version_string, registry_path)``
        """
        if version is None:
            version = make_version()
        dest = self.root / version
        dest.mkdir(parents=True, exist_ok=True)

        for src in Path(source_dir).iterdir():
            if src.is_file():
                shutil.copy2(src, dest / src.name)

        if metrics is not None:
            (dest / "metrics.json").write_text(
                json.dumps(metrics, indent=2, default=str),
                encoding="utf-8",
            )

        # Mettre à jour le pointeur "latest"
        (self.root / "latest.txt").write_text(version, encoding="utf-8")

        print(f"[registry] Registered {self.model_name}  version={version}  -> {dest}")
        return version, dest

    # ── Requêtes ──────────────────────────────────────────────────────────────

    def latest_version(self) -> str | None:
        """Retourne la dernière chaîne de version enregistrée, ou None."""
        p = self.root / "latest.txt"
        return p.read_text(encoding="utf-8").strip() if p.exists() else None

    def latest_path(self) -> Path | None:
        """Retourne le répertoire de la dernière version, ou None."""
        v = self.latest_version()
        if v is None:
            return None
        p = self.root / v
        return p if p.exists() else None

    def list_versions(self) -> list[str]:
        """Retourne tous les répertoires de version enregistrés, triés chronologiquement."""
        if not self.root.exists():
            return []
        return sorted(
            d.name for d in self.root.iterdir()
            if d.is_dir() and d.name.startswith("v")
        )

    def get_version_path(self, version: str) -> Path:
        """Retourne le chemin pour *version*, lève FileNotFoundError si absent."""
        p = self.root / version
        if not p.exists():
            raise FileNotFoundError(
                f"Version '{version}' not found in registry at {self.root}"
            )
        return p

    def load_metrics(self, version: str | None = None) -> dict:
        """Charge le fichier metrics.json pour *version* (dernière version par défaut)."""
        v = version or self.latest_version()
        if v is None:
            raise RuntimeError("No registered version found.")
        p = self.root / v / "metrics.json"
        if not p.exists():
            raise FileNotFoundError(f"metrics.json not found in version {v}")
        return json.loads(p.read_text(encoding="utf-8"))
