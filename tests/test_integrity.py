"""Tests for artifact integrity (src/utils/integrity.py)."""
import hashlib
from pathlib import Path

import pytest

from src.utils.integrity import (
    generate_checksums,
    save_checksums,
    sha256_file,
    verify_checksums,
)


# ── sha256_file ───────────────────────────────────────────────────────────────

def test_known_digest(tmp_path):
    p = tmp_path / "hello.txt"
    p.write_bytes(b"hello world")
    expected = hashlib.sha256(b"hello world").hexdigest()
    assert sha256_file(p) == expected


def test_different_content_different_digest(tmp_path):
    p1 = tmp_path / "a.txt"
    p2 = tmp_path / "b.txt"
    p1.write_bytes(b"aaa")
    p2.write_bytes(b"bbb")
    assert sha256_file(p1) != sha256_file(p2)


# ── generate_checksums / save_checksums / verify_checksums ───────────────────

def test_generate_returns_correct_keys(tmp_path):
    f1 = tmp_path / "a.txt"
    f2 = tmp_path / "b.txt"
    f1.write_bytes(b"aaa")
    f2.write_bytes(b"bbb")
    cs = generate_checksums([f1, f2])
    assert str(f1) in cs
    assert str(f2) in cs


def test_missing_file_marked_as_MISSING(tmp_path):
    cs = generate_checksums([tmp_path / "ghost.txt"])
    assert list(cs.values())[0] == "MISSING"


def test_save_creates_both_formats(tmp_path):
    f = tmp_path / "model.joblib"
    f.write_bytes(b"model_data")
    cs = generate_checksums([f])
    save_checksums(cs, tmp_path / "checksums")
    assert (tmp_path / "checksums.sha256").exists()
    assert (tmp_path / "checksums.json").exists()


def test_verify_passes_for_unchanged_files(tmp_path):
    f = tmp_path / "model.joblib"
    f.write_bytes(b"model_data")
    cs = generate_checksums([f])
    save_checksums(cs, tmp_path / "checksums")
    ok, errors = verify_checksums(tmp_path / "checksums")
    assert ok
    assert errors == []


def test_verify_fails_for_modified_file(tmp_path):
    f = tmp_path / "model.joblib"
    f.write_bytes(b"original")
    cs = generate_checksums([f])
    save_checksums(cs, tmp_path / "checksums")
    f.write_bytes(b"tampered")
    ok, errors = verify_checksums(tmp_path / "checksums")
    assert not ok
    assert len(errors) == 1
    assert "CHECKSUM_MISMATCH" in errors[0]


def test_verify_fails_for_deleted_file(tmp_path):
    f = tmp_path / "model.joblib"
    f.write_bytes(b"data")
    cs = generate_checksums([f])
    save_checksums(cs, tmp_path / "checksums")
    f.unlink()
    ok, errors = verify_checksums(tmp_path / "checksums")
    assert not ok
    assert "MISSING" in errors[0]


def test_verify_missing_checksums_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        verify_checksums(tmp_path / "nonexistent_checksums")
