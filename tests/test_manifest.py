"""Tests for src/utils/manifest.py."""
import json

import pytest

from src.utils.manifest import (
    _TRACKED_PACKAGES,
    _git_commit,
    _package_versions,
    build_manifest,
    dataset_fingerprint,
    save_manifest,
)


# ── _git_commit ───────────────────────────────────────────────────────────────

def test_git_commit_returns_string():
    result = _git_commit()
    assert isinstance(result, str)
    assert len(result) > 0


def test_git_commit_is_hex_or_unavailable():
    result = _git_commit()
    assert result == "unavailable" or all(c in "0123456789abcdef" for c in result)


def test_git_commit_returns_unavailable_on_error(monkeypatch):
    import subprocess
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: (_ for _ in ()).throw(OSError("no git")))
    assert _git_commit() == "unavailable"


# ── _package_versions ─────────────────────────────────────────────────────────

def test_package_versions_known_package():
    versions = _package_versions(["numpy"])
    assert "numpy" in versions
    assert versions["numpy"] != "not_installed"


def test_package_versions_unknown_package():
    versions = _package_versions(["this-package-xyz-does-not-exist"])
    assert versions["this-package-xyz-does-not-exist"] == "not_installed"


def test_package_versions_multiple():
    versions = _package_versions(["numpy", "pandas"])
    assert "numpy" in versions
    assert "pandas" in versions


# ── dataset_fingerprint ───────────────────────────────────────────────────────

def test_dataset_fingerprint_empty_list():
    fp = dataset_fingerprint([])
    assert isinstance(fp, str)
    assert len(fp) == 64  # SHA-256 hex digest length


def test_dataset_fingerprint_with_file(tmp_path):
    f = tmp_path / "data.bin"
    f.write_bytes(b"hello world")
    fp = dataset_fingerprint([f])
    assert len(fp) == 64


def test_dataset_fingerprint_nonexistent_file_ignored(tmp_path):
    fp = dataset_fingerprint([tmp_path / "missing.parquet"])
    assert isinstance(fp, str)


def test_dataset_fingerprint_deterministic(tmp_path):
    f = tmp_path / "file.txt"
    f.write_bytes(b"reproducibility")
    assert dataset_fingerprint([f]) == dataset_fingerprint([f])


def test_dataset_fingerprint_changes_with_content(tmp_path):
    f1 = tmp_path / "a.txt"
    f2 = tmp_path / "b.txt"
    f1.write_bytes(b"aaa")
    f2.write_bytes(b"bbb")
    assert dataset_fingerprint([f1]) != dataset_fingerprint([f2])


# ── build_manifest ────────────────────────────────────────────────────────────

def test_build_manifest_has_required_keys():
    m = build_manifest(config_snapshot={}, dataset_paths=[])
    for key in ("timestamp_utc", "git_commit", "package_versions",
                "config", "dataset_fingerprint", "dataset_files"):
        assert key in m


def test_build_manifest_config_stored():
    m = build_manifest(config_snapshot={"n_trials": 60, "seed": 42}, dataset_paths=[])
    assert m["config"]["n_trials"] == 60
    assert m["config"]["seed"] == 42


def test_build_manifest_with_extra():
    m = build_manifest({}, [], extra={"custom_key": "custom_value"})
    assert m["custom_key"] == "custom_value"


def test_build_manifest_tracked_packages_present():
    m = build_manifest({}, [])
    versions = m["package_versions"]
    for pkg in _TRACKED_PACKAGES:
        assert pkg in versions


def test_build_manifest_dataset_files_only_existing(tmp_path):
    real = tmp_path / "real.parquet"
    real.write_bytes(b"fake parquet data")
    missing = tmp_path / "missing.parquet"
    m = build_manifest({}, [real, missing])
    assert str(real) in m["dataset_files"]
    assert str(missing) not in m["dataset_files"]


# ── save_manifest ─────────────────────────────────────────────────────────────

def test_save_manifest_creates_file(tmp_path):
    m = build_manifest({"n_trials": 5}, [])
    out = tmp_path / "manifest.json"
    save_manifest(m, out)
    assert out.exists()


def test_save_manifest_valid_json(tmp_path):
    m = build_manifest({}, [])
    out = tmp_path / "manifest.json"
    save_manifest(m, out)
    data = json.loads(out.read_text())
    assert "git_commit" in data


def test_save_manifest_creates_parent_dirs(tmp_path):
    m = build_manifest({}, [])
    out = tmp_path / "a" / "b" / "manifest.json"
    save_manifest(m, out)
    assert out.exists()
