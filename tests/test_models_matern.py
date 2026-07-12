"""Tests for the free-nu Matérn REML model."""

import contextlib
import io
import unittest

import gpmp.num as gnp
from gpmp.parameter import Normalization

import gpmpcontrib as gpc


def _observations():
    xi = gnp.linspace(0.0, 1.0, 16).reshape(-1, 1)
    zi = gnp.sin(2.0 * gnp.pi * xi[:, 0]) + 0.3 * gnp.abs(xi[:, 0] - 0.45)
    return xi, zi


class TestMaternREMLModel(unittest.TestCase):
    def test_initial_guess_and_param_object(self):
        xi, zi = _observations()
        model = gpc.Model_ConstantMean_Matern_REML(
            "matern", 1, mean_specification={"type": "constant"}
        )

        initial_guess = model[0]["parameters_initial_guess_procedure"]
        covparam0 = initial_guess(model[0].model, xi, zi)
        self.assertEqual(tuple(covparam0.shape), (3,))

        param_from_vectors, _ = model.build_param_procedures(
            0,
            name_prefix="z0_",
            logsigma2_bounds=(-10.0, 10.0),
            lognu_bounds=(-2.0, 4.0),
            loginvrho_bounds=(-8.0, 8.0),
        )
        param = param_from_vectors(gnp.asarray([]), covparam0)
        self.assertEqual(param.names, ["z0_sigma2", "z0_nu", "z0_rho_0"])
        self.assertEqual(
            param.paths,
            [
                ["covparam", "variance"],
                ["covparam", "regularity"],
                ["covparam", "lengthscale"],
            ],
        )
        self.assertEqual(
            param.normalizations,
            [Normalization.LOG, Normalization.LOG, Normalization.LOG_INV],
        )
        self.assertEqual(
            param.bounds,
            [(-10.0, 10.0), (-2.0, 4.0), (-8.0, 8.0)],
        )

    def test_selection_prediction_and_diagnosis(self):
        xi, zi = _observations()
        model = gpc.Model_ConstantMean_Matern_REML(
            "matern", 1, mean_specification={"type": "constant"}
        )
        model.select_params(xi, zi, method_options={"maxiter": 80})

        covparam = model[0].model.covparam
        self.assertTrue(model[0].info.success)
        self.assertEqual(tuple(covparam.shape), (3,))
        self.assertTrue(gnp.to_scalar(gnp.isfinite(covparam[1])))
        self.assertGreater(gnp.to_scalar(gnp.exp(covparam[1])), 0.0)

        zpm, zpv = model.predict(xi, zi, xi)
        self.assertEqual(tuple(zpm.shape), (16, 1))
        self.assertEqual(tuple(zpv.shape), (16, 1))

        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            model.run_diagnosis(xi, zi)
        self.assertIn("z0_nu", output.getvalue())

    def test_multi_output_construction(self):
        xi, zi0 = _observations()
        zi = gnp.stack((zi0, gnp.cos(2.0 * gnp.pi * xi[:, 0])), axis=1)
        model = gpc.Model_ConstantMean_Matern_REML(
            "matern_multi", 2, mean_specification={"type": "constant"}
        )

        self.assertEqual(len(model.models), 2)
        for output_idx in range(2):
            initial_guess = model[output_idx]["parameters_initial_guess_procedure"]
            covparam0 = initial_guess(
                model[output_idx].model, xi, zi[:, output_idx]
            )
            self.assertEqual(tuple(covparam0.shape), (3,))


if __name__ == "__main__":
    unittest.main()
