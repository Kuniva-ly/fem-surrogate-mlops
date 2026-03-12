"""Intégrité des artefacts : génération et vérification des checksums SHA-256.

Utilisation
-----------
    from src.utils.integrity import generate_checksums, save_checksums, verify_checksums

    # Générer
    paths = list(Path("artifacts/models/lgbm_surrogate/v20260310_120000").glob("*"))
    checksums = generate_checksums(paths)
    save_checksums(checksums, Path("artifacts/.../checksums"))

    # Vérifier
    ok, errors = verify_checksums(Path("artifacts/.../checksums"))
    if not ok:
        for e in errors:
            print(e)
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def sha256_file(path: Path) -> str:
    """Calcule le condensé hexadécimal SHA-256 d'un fichier (en flux, fonctionne pour les grands fichiers)."""
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(65_536), b""):
            h.update(chunk)
    return h.hexdigest()


def generate_checksums(paths: list[Path]) -> dict[str, str]:
    """Retourne {str(path): sha256_hex} pour chaque chemin. Fichiers manquants → 'MISSING'."""
    return {
        str(p): sha256_file(p) if Path(p).exists() else "MISSING"
        for p in paths
    }


def save_checksums(checksums: dict[str, str], out_path: Path) -> None:
    """Sauvegarde les checksums dans deux formats :

    - ``<out_path>.sha256``  — format texte compatible sha256sum
    - ``<out_path>.json``    — dict JSON pour accès programmatique
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Format texte (compatible sha256sum / certutil)
    lines = [f"{digest}  {file_path}" for file_path, digest in sorted(checksums.items())]
    out_path.with_suffix(".sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Format JSON
    out_path.with_suffix(".json").write_text(
        json.dumps(checksums, indent=2), encoding="utf-8"
    )
    print(f"Checksums saved: {out_path.with_suffix('.sha256')}")


def verify_checksums(checksums_path: Path) -> tuple[bool, list[str]]:
    """Vérifie les fichiers par rapport aux checksums SHA-256 précédemment sauvegardés.

    Essaie ``<checksums_path>.json`` en premier, puis ``.sha256``.

    Returns:
        (ok, errors): ok=True si tous les fichiers sont valides ; errors liste les échecs.
    """
    checksums_path = Path(checksums_path)
    json_path = checksums_path.with_suffix(".json")
    sha_path = checksums_path.with_suffix(".sha256")

    if json_path.exists():
        checksums: dict[str, str] = json.loads(json_path.read_text(encoding="utf-8"))
    elif sha_path.exists():
        checksums = {}
        for line in sha_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                parts = line.split("  ", 1)
                if len(parts) == 2:
                    checksums[parts[1]] = parts[0]
    else:
        raise FileNotFoundError(
            f"No checksums file found at {checksums_path} (.json or .sha256)"
        )

    errors: list[str] = []
    for file_path, expected in checksums.items():
        p = Path(file_path)
        if expected == "MISSING":
            # Était déjà absent lors de la génération des checksums — ignorer
            errors.append(f"SKIP (was missing at generation): {file_path}")
            continue
        if not p.exists():
            errors.append(f"MISSING: {file_path}")
            continue
        actual = sha256_file(p)
        if actual != expected:
            errors.append(
                f"CHECKSUM_MISMATCH: {file_path}\n"
                f"  expected = {expected}\n"
                f"  actual   = {actual}"
            )

    return (len(errors) == 0), errors
