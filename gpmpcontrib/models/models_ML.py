# gpmpcontrib/models/models_ML.py
# --------------------------------------------------------------
# Author: Emmanuel Vazquez <emmanuel.vazquez@centralesupelec.fr>
# Copyright (c) 2022-2026, CentraleSupelec
# License: GPLv3 (see LICENSE)
# --------------------------------------------------------------
"""ML Matern model classes."""

import gpmp as gp
import gpmpcontrib.modelcontainer


class Model_ConstantMean_Maternp_ML(gpmpcontrib.modelcontainer.ModelContainer):
    """GP model with constant mean and Matern covariance, parameters by ML."""

    def __init__(self, name, output_dim, covariance_specification=None):
        super().__init__(
            name,
            output_dim,
            parameterized_mean=True,
            mean_specification={"type": "constant"},
            covariance_specification=covariance_specification,
        )

    def build_mean_function(self, output_idx: int, param: dict):
        if "type" not in param:
            raise ValueError("Mean 'type' should be specified in 'param'")
        if param["type"] == "constant":
            return (gpmpcontrib.modelcontainer.mean_parameterized_constant, 1)
        raise NotImplementedError(f"Mean type {param['type']} not implemented")

    def build_covariance(self, output_idx: int, param: dict):
        if ("p" not in param) or (not isinstance(param["p"], int)):
            raise ValueError("Regularity 'p' should be integer and specified in 'param'")

        p = param["p"]

        def maternp_covariance(x, y, covparam, pairwise=False):
            return gp.kernel.maternp_covariance(x, y, p, covparam, pairwise)

        return maternp_covariance

    def build_parameters_initial_guess_procedure(self, output_idx: int, **build_param):
        return gp.kernel.anisotropic_parameters_initial_guess_constant_mean

    def build_selection_criterion(self, output_idx: int, **build_params):
        def ml_criterion(model, meanparam, covparam, xi, zi):
            return model.negative_log_likelihood(meanparam, covparam, xi, zi)

        return ml_criterion

