"""Artifact integrity: SHA-256 checksum generation and verification.

Usage
-----
    from src.utils.integrity import generate_checksums, save_checksums, verify_checksums

    # Generate
    paths = list(Path("artifacts/models/lgbm_surrogate/v20260310_120000").glob("*"))
    checksums = generate_checksums(paths)
    save_checksums(checksums, Path("artifacts/.../checksums"))

    # Verify
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
    """Compute SHA-256 hex digest of a file (streaming, works for large files)."""
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(65_536), b""):
            h.update(chunk)
    return h.hexdigest()


def generate_checksums(paths: list[Path]) -> dict[str, str]:
    """Return {str(path): sha256_hex} for each path. Missing files → 'MISSING'."""
    return {
        str(p): sha256_file(p) if Path(p).exists() else "MISSING"
        for p in paths
    }


def save_checksums(checksums: dict[str, str], out_path: Path) -> None:
    """Save checksums in two formats:

    - ``<out_path>.sha256``  — sha256sum-compatible text format
    - ``<out_path>.json``    — JSON dict for programmatic access
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Text format (compatible with sha256sum / certutil)
    lines = [f"{digest}  {file_path}" for file_path, digest in sorted(checksums.items())]
    out_path.with_suffix(".sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # JSON format
    out_path.with_suffix(".json").write_text(
        json.dumps(checksums, indent=2), encoding="utf-8"
    )
    print(f"Checksums saved: {out_path.with_suffix('.sha256')}")


def verify_checksums(checksums_path: Path) -> tuple[bool, list[str]]:
    """Verify files against previously saved SHA-256 checksums.

    Tries ``<checksums_path>.json`` first, then ``.sha256``.

    Returns:
        (ok, errors): ok=True if all files pass; errors lists failures.
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
            # Was already missing when checksums were generated — skip
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
