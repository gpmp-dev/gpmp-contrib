"""Tests for the public gpmpcontrib package version."""

import unittest
from pathlib import Path

import gpmpcontrib as gpc


class TestPackageVersion(unittest.TestCase):
    def test_source_version_is_exposed(self):
        version_file = Path(__file__).resolve().parents[1] / "VERSION"
        expected = version_file.read_text(encoding="utf-8").strip()
        self.assertEqual(gpc.__version__, expected)


if __name__ == "__main__":
    unittest.main()
