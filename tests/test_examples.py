"""Smoke tests for lightweight gpmpcontrib examples.

Author: Emmanuel Vazquez <emmanuel.vazquez@centralesupelec.fr>
Copyright (c) 2022-2026, CentraleSupelec
License: GPLv3 (see LICENSE)
"""

import importlib.util
import os
import sys
import unittest
from pathlib import Path


def _find_repo_root(start_path: Path) -> Path:
    """Find repository root containing both pyproject.toml and examples/."""
    for parent in [start_path, *start_path.parents]:
        if (parent / "pyproject.toml").exists() and (parent / "examples").is_dir():
            return parent
    cwd = Path.cwd().resolve()
    for parent in [cwd, *cwd.parents]:
        if (parent / "pyproject.toml").exists() and (parent / "examples").is_dir():
            return parent
    raise RuntimeError("Could not locate gpmp-contrib repository root.")


REPO_ROOT = _find_repo_root(Path(__file__).resolve())
EXAMPLES_DIR = REPO_ROOT / "examples"
LIGHT_EXAMPLES = [
    "example01_computer_experiment",
    "example02_models",
    "example03_sequential_prediction",
    "example04_sequential_prediction_with_noise",
    "example10_optim_EI_gridsearch",
]


def run_example(example_name):
    """Dynamically import and execute one example script."""
    example_path = EXAMPLES_DIR / f"{example_name}.py"
    spec = importlib.util.spec_from_file_location(example_name, example_path)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise ImportError(f"Unable to load example: {example_name}")
    spec.loader.exec_module(module)
    return module


class TestExamples(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._old_mplbackend = os.environ.get("MPLBACKEND")
        os.environ["MPLBACKEND"] = "Agg"
        cls._sys_path_added = []
        for path in (str(REPO_ROOT), str(EXAMPLES_DIR)):
            if path not in sys.path:
                sys.path.insert(0, path)
                cls._sys_path_added.append(path)
        examples_str = str(EXAMPLES_DIR)

    @classmethod
    def tearDownClass(cls):
        for path in cls._sys_path_added:
            if path in sys.path:
                sys.path.remove(path)
        if cls._old_mplbackend is None:
            os.environ.pop("MPLBACKEND", None)
        else:
            os.environ["MPLBACKEND"] = cls._old_mplbackend

    def test_light_examples(self):
        for example_name in LIGHT_EXAMPLES:
            with self.subTest(example=example_name):
                run_example(example_name)


if __name__ == "__main__":
    unittest.main()
