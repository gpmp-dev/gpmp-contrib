"""Test examples 01-04 from gpmpcontrib.

These tests verify that the basic examples can be imported and run without errors.

Author: Emmanuel Vazquez <emmanuel.vazquez@centralesupelec.fr>
Copyright (c) 2022-2026, CentraleSupelec
License: GPLv3 (see LICENSE)
"""

import unittest
import sys
import importlib.util
from pathlib import Path


def import_example(example_name):
    """Dynamically import an example script as a module."""
    example_path = Path(__file__).parent.parent / "examples" / f"{example_name}.py"
    spec = importlib.util.spec_from_file_location(example_name, example_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestExamples(unittest.TestCase):
    def test_example01_computer_experiment(self):
        """Test example 01: ComputerExperiment class."""
        import_example("example01_computer_experiment")

    def test_example02_models(self):
        """Test example 02: Models."""
        import_example("example02_models")

    def test_example03_sequential_prediction(self):
        """Test example 03: Sequential prediction."""
        import_example("example03_sequential_prediction")

    def test_example04_sequential_prediction_with_noise(self):
        """Test example 04: Sequential prediction with noise."""
        import_example("example04_sequential_prediction_with_noise")


if __name__ == "__main__":
    unittest.main()
