"""Simple local model registry.

Directory structure
-------------------
    artifacts/models/
        <model_name>/
            <version>/          e.g. v20260310_143022/
                *.joblib
                feature_columns.txt
                training_config.yaml
                metrics.json
                metrics.csv
                manifest.json
                checksums.sha256
                checksums.json
            latest.txt          contains the name of the latest version directory

Version format: ``v{YYYYMMDD}_{HHMMSS}`` (UTC).

Usage
-----
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
    """Generate a timestamp-based version string (UTC)."""
    ts = ts or datetime.now(timezone.utc)
    return f"v{ts.strftime(_VERSION_FMT)}"


class ModelRegistry:
    """Filesystem model registry with versioned artifact storage."""

    def __init__(self, registry_dir: Path, model_name: str) -> None:
        self.root = Path(registry_dir) / model_name
        self.model_name = model_name

    # Registration

    def register(
        self,
        source_dir: Path,
        version: str | None = None,
        metrics: dict | None = None,
    ) -> tuple[str, Path]:
        """Copy all files from *source_dir* into a new versioned location.

        Args:
            source_dir: Directory containing trained artifacts.
            version:    Version string to use (default: UTC timestamp).
            metrics:    Optional metrics dict saved as ``metrics.json``.

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

        # Update the "latest" pointer
        (self.root / "latest.txt").write_text(version, encoding="utf-8")

        print(f"[registry] Registered {self.model_name}  version={version}  -> {dest}")
        return version, dest

    # Queries

    def latest_version(self) -> str | None:
        """Return the latest registered version string, or None."""
        p = self.root / "latest.txt"
        return p.read_text(encoding="utf-8").strip() if p.exists() else None

    def latest_path(self) -> Path | None:
        """Return the latest version directory, or None."""
        v = self.latest_version()
        if v is None:
            return None
        p = self.root / v
        return p if p.exists() else None

    def list_versions(self) -> list[str]:
        """Return all registered version directories, sorted chronologically."""
        if not self.root.exists():
            return []
        return sorted(
            d.name for d in self.root.iterdir()
            if d.is_dir() and d.name.startswith("v")
        )

    def get_version_path(self, version: str) -> Path:
        """Return the path for *version*, raise FileNotFoundError if absent."""
        p = self.root / version
        if not p.exists():
            raise FileNotFoundError(
                f"Version '{version}' not found in registry at {self.root}"
            )
        return p

    def load_metrics(self, version: str | None = None) -> dict:
        """Load the metrics.json file for *version* (latest version by default)."""
        v = version or self.latest_version()
        if v is None:
            raise RuntimeError("No registered version found.")
        p = self.root / v / "metrics.json"
        if not p.exists():
            raise FileNotFoundError(f"metrics.json not found in version {v}")
        return json.loads(p.read_text(encoding="utf-8"))
