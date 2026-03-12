"""Tests pour l'intégrité des artefacts (src/utils/integrity.py)."""
import tempfile
import unittest
from pathlib import Path

from src.utils.integrity import (
    generate_checksums,
    save_checksums,
    sha256_file,
    verify_checksums,
)


class TestSha256File(unittest.TestCase):
    def test_known_digest(self) -> None:
        import hashlib
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"hello world")
            path = Path(f.name)
        expected = hashlib.sha256(b"hello world").hexdigest()
        self.assertEqual(sha256_file(path), expected)

    def test_different_content_different_digest(self) -> None:
        with tempfile.NamedTemporaryFile(delete=False) as f1:
            f1.write(b"aaa")
            p1 = Path(f1.name)
        with tempfile.NamedTemporaryFile(delete=False) as f2:
            f2.write(b"bbb")
            p2 = Path(f2.name)
        self.assertNotEqual(sha256_file(p1), sha256_file(p2))


class TestGenerateAndVerify(unittest.TestCase):

    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.base = Path(self._tmpdir.name)

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def _make_file(self, name: str, content: bytes) -> Path:
        p = self.base / name
        p.write_bytes(content)
        return p

    def test_generate_returns_correct_keys(self) -> None:
        f1 = self._make_file("a.txt", b"aaa")
        f2 = self._make_file("b.txt", b"bbb")
        cs = generate_checksums([f1, f2])
        self.assertIn(str(f1), cs)
        self.assertIn(str(f2), cs)

    def test_missing_file_marked_as_MISSING(self) -> None:
        cs = generate_checksums([self.base / "ghost.txt"])
        self.assertEqual(list(cs.values())[0], "MISSING")

    def test_save_creates_both_formats(self) -> None:
        f = self._make_file("model.joblib", b"model_data")
        cs = generate_checksums([f])
        save_checksums(cs, self.base / "checksums")
        self.assertTrue((self.base / "checksums.sha256").exists())
        self.assertTrue((self.base / "checksums.json").exists())

    def test_verify_passes_for_unchanged_files(self) -> None:
        f = self._make_file("model.joblib", b"model_data")
        cs = generate_checksums([f])
        save_checksums(cs, self.base / "checksums")
        ok, errors = verify_checksums(self.base / "checksums")
        self.assertTrue(ok)
        self.assertEqual(errors, [])

    def test_verify_fails_for_modified_file(self) -> None:
        f = self._make_file("model.joblib", b"original")
        cs = generate_checksums([f])
        save_checksums(cs, self.base / "checksums")
        # Altérer le fichier
        f.write_bytes(b"tampered")
        ok, errors = verify_checksums(self.base / "checksums")
        self.assertFalse(ok)
        self.assertEqual(len(errors), 1)
        self.assertIn("CHECKSUM_MISMATCH", errors[0])

    def test_verify_fails_for_deleted_file(self) -> None:
        f = self._make_file("model.joblib", b"data")
        cs = generate_checksums([f])
        save_checksums(cs, self.base / "checksums")
        f.unlink()
        ok, errors = verify_checksums(self.base / "checksums")
        self.assertFalse(ok)
        self.assertIn("MISSING", errors[0])

    def test_verify_missing_checksums_file_raises(self) -> None:
        with self.assertRaises(FileNotFoundError):
            verify_checksums(self.base / "nonexistent_checksums")


if __name__ == "__main__":
    unittest.main()
