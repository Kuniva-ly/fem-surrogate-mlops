"""Manifeste de run pour la reproductibilité.

Capture et persiste :
  - horodatage du run (UTC ISO-8601)
  - hash du commit git
  - versions des paquets installés
  - snapshot de la configuration
  - empreinte du jeu de données (SHA-256 sur tous les fichiers d'entraînement)
  - liste des fichiers du jeu de données

Utilisation
-----------
    from src.utils.manifest import build_manifest, save_manifest

    manifest = build_manifest(
        config_snapshot={"n_trials": 60, "random_state": 42},
        dataset_paths=[Path("data/processed/train.parquet"), ...],
    )
    save_manifest(manifest, Path("artifacts/models/.../manifest.json"))
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Paquets dont on suit toujours les versions
_TRACKED_PACKAGES = [
    "lightgbm",
    "scikit-learn",
    "optuna",
    "pandas",
    "numpy",
    "pyarrow",
    "joblib",
    "pyyaml",
]


def _git_commit() -> str:
    """Retourne le hash du commit HEAD actuel, ou 'unavailable'."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.stdout.strip() if result.returncode == 0 else "unavailable"
    except Exception:
        return "unavailable"


def _package_versions(packages: list[str]) -> dict[str, str]:
    versions: dict[str, str] = {}
    for pkg in packages:
        try:
            versions[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            versions[pkg] = "not_installed"
    return versions


def dataset_fingerprint(paths: list[Path]) -> str:
    """Calcule un condensé SHA-256 unique sur tous les fichiers fournis (triés)."""
    h = hashlib.sha256()
    for p in sorted(str(x) for x in paths):
        fp = Path(p)
        if fp.exists():
            h.update(fp.read_bytes())
    return h.hexdigest()


def build_manifest(
    config_snapshot: dict[str, Any],
    dataset_paths: list[Path],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Construit un dictionnaire manifeste de reproductibilité.

    Args:
        config_snapshot: Dict sérialisable des paramètres du run.
        dataset_paths:   Liste des fichiers de jeu de données consommés par ce run.
        extra:           Paires clé-valeur supplémentaires à inclure.

    Returns:
        Dict manifeste prêt pour la sérialisation JSON.
    """
    manifest: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "package_versions": _package_versions(_TRACKED_PACKAGES),
        "config": config_snapshot,
        "dataset_fingerprint": dataset_fingerprint(dataset_paths),
        "dataset_files": sorted(str(p) for p in dataset_paths if Path(p).exists()),
    }
    if extra:
        manifest.update(extra)
    return manifest


def save_manifest(manifest: dict[str, Any], out_path: Path) -> None:
    """Écrit le dict manifeste dans un fichier JSON."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(manifest, indent=2, default=str),
        encoding="utf-8",
    )
    print(f"Manifest saved : {out_path}")
