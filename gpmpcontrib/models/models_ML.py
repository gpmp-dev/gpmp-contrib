# gpmpcontrib/models/models_ML.py
# --------------------------------------------------------------
# Author: Emmanuel Vazquez <emmanuel.vazquez@centralesupelec.fr>
# Copyright (c) 2022-2026, CentraleSupelec
# License: GPLv3 (see LICENSE)
# --------------------------------------------------------------
"""ML Matern model classes."""

import gpmp as gp
import gpmp.num as gnp
import gpmpcontrib.modelcontainer
from gpmp.misc.param import Normalization, Param


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

    def build_param_procedures(self, output_idx: int, **kwargs):
        mpl = int(self.models[output_idx]["mean_paramlength"] or 0)
        logsigma2_bounds = kwargs.get("logsigma2_bounds", None)
        loginvrho_bounds = kwargs.get("loginvrho_bounds", None)
        name_prefix = kwargs.get("name_prefix", "")
        mean_names = kwargs.get("mean_names", None)

        def _mean_param_from_vector(meanparam):
            mp = gnp.asarray(meanparam).reshape(-1)
            m = int(mp.shape[0])
            if m == 0:
                return Param(values=gnp.zeros(0), names=[], paths=[], normalizations=[], bounds=[])
            if mean_names is not None:
                if len(mean_names) != m:
                    raise ValueError("mean_names length must match number of mean parameters")
                names = [f"{name_prefix}{nm}" for nm in mean_names]
            else:
                names = [f"{name_prefix}beta_{j}" for j in range(m)]
            paths = [["meanparam"]] * m
            norms = [Normalization.NONE] * m
            return Param(values=mp, names=names, paths=paths, normalizations=norms)

        def _cov_param_from_vector(covparam):
            cp = gnp.asarray(covparam).reshape(-1)
            d = int(cp.shape[0]) - 1
            names = [f"{name_prefix}sigma2"] + [f"{name_prefix}rho_{j}" for j in range(d)]
            paths = [["covparam", "variance"]] + [["covparam", "lengthscale"]] * d
            norms = [Normalization.LOG] + [Normalization.LOG_INV] * d
            bounds = [logsigma2_bounds] + [loginvrho_bounds] * d
            return Param(
                values=cp,
                names=names,
                paths=paths,
                normalizations=norms,
                bounds=bounds,
            )

        def param_from_vectors(meanparam, covparam):
            meanp = _mean_param_from_vector(meanparam if mpl > 0 else gnp.zeros(0))
            covp = _cov_param_from_vector(covparam)
            return Param.concat(meanp, covp)

        def vectors_from_param(param):
            mp = (
                gnp.asarray(param.get_by_path(["meanparam"], prefix_match=True)).reshape(-1)
                if mpl > 0
                else gnp.asarray([])
            )
            cp = gnp.asarray(param.get_by_path(["covparam"], prefix_match=True)).reshape(-1)
            return mp, cp

        return param_from_vectors, vectors_from_param

    def build_parameters_initial_guess_procedure(self, output_idx: int, **build_param):
        return gp.kernel.anisotropic_parameters_initial_guess_constant_mean

    def build_selection_criterion(self, output_idx: int, **build_params):
        def ml_criterion(model, meanparam, covparam, xi, zi):
            return model.negative_log_likelihood(meanparam, covparam, xi, zi)

        return ml_criterion
