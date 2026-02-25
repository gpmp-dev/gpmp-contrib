# gpmpcontrib/models/models_noisy_REML.py
# --------------------------------------------------------------
# Author: Emmanuel Vazquez <emmanuel.vazquez@centralesupelec.fr>
# Copyright (c) 2022-2026, CentraleSupelec
# License: GPLv3 (see LICENSE)
# --------------------------------------------------------------
"""Noisy-output REML Matern model classes."""

from math import log

import gpmp as gp
import gpmp.num as gnp
import gpmpcontrib.modelcontainer
from gpmp.misc.param import Normalization, Param


def build_mown_kernel(output_idx: int, **params):
    """Build Matern kernel handling per-output noise dimensions."""
    p = params.get("p", 2)

    def kernel_ii_or_tt(x, param, output_idx, pairwise=False):
        sigma2 = gnp.exp(param[0])
        loginvrho = param[1:]
        d = loginvrho.shape[0]
        noise_idx = d + output_idx
        nugget = 10 * gnp.finfo(gnp.float64).eps

        if pairwise:
            k = sigma2 * gnp.ones((x.shape[0],)) + x[:, noise_idx] + nugget
        else:
            k = gnp.scaled_distance(loginvrho, x[:, :d], x[:, :d])
            k = sigma2 * gp.kernel.maternp_kernel(p, k) + gnp.diag(
                x[:, noise_idx] + nugget
            )
        return k

    def kernel_it(x, y, param, pairwise=False):
        sigma2 = gnp.exp(param[0])
        loginvrho = param[1:]
        d = loginvrho.shape[0]

        if pairwise:
            k = gnp.scaled_distance_elementwise(loginvrho, x[:, :d], y[:, :d])
        else:
            k = gnp.scaled_distance(loginvrho, x[:, :d], y[:, :d])
        return sigma2 * gp.kernel.maternp_kernel(p, k)

    def kernel(x, y, param, pairwise=False):
        if y is x or y is None:
            return kernel_ii_or_tt(x, param, output_idx, pairwise)
        return kernel_it(x, y, param, pairwise)

    return kernel


def noisy_outputs_parameters_initial_guess(model, xi, zi, output_dim):
    """Initial guess for noisy multi-output models, using spatial dimensions only."""
    xi_ = gnp.asarray(xi)
    zi_ = gnp.asarray(zi).reshape(-1, 1)
    n = xi_.shape[0]
    d = xi_.shape[1] - output_dim

    spatial_xi = xi[:, :-output_dim]
    delta = gnp.max(spatial_xi, axis=0) - gnp.min(spatial_xi, axis=0)
    rho = gnp.exp(gnp.gammaln(d / 2 + 1) / d) / (gnp.pi**0.5) * delta

    covparam = gnp.concatenate((gnp.array([log(1.0)]), -gnp.log(rho)))
    sigma2_gls = 1.0 / n * model.norm_k_sqrd(xi_, zi_, covparam)
    return gnp.concatenate((gnp.log(sigma2_gls), -gnp.log(rho)))


class Model_Noisy_ConstantMean_Maternp_REML(gpmpcontrib.modelcontainer.ModelContainer):
    def __init__(self, name, output_dim, mean_specification, covariance_specification):
        super().__init__(
            name,
            output_dim,
            parameterized_mean=False,
            mean_specification=mean_specification,
            covariance_specification=covariance_specification,
        )

    def build_mean_function(self, output_idx: int, mean_build_param: dict):
        if "type" not in mean_build_param:
            raise ValueError("Mean 'type' should be specified in 'mean_build_param'")
        if mean_build_param["type"] == "constant":
            return (gpmpcontrib.modelcontainer.mean_linpred_constant, 0)
        raise NotImplementedError(f"Mean type {mean_build_param['type']} not implemented")

    def build_covariance(self, output_idx: int, covariance_build_param: dict):
        if ("p" not in covariance_build_param) or (
            not isinstance(covariance_build_param["p"], int)
        ):
            raise ValueError(
                "Regularity 'p' should be integer and specified in 'covariance_build_param'"
            )
        return build_mown_kernel(output_idx, **covariance_build_param)

    def build_param_procedures(self, output_idx: int, **kwargs):
        mpl = int(self.models[output_idx]["mean_paramlength"] or 0)
        logsigma2_bounds = kwargs.get("logsigma2_bounds", None)
        loginvrho_bounds = kwargs.get("loginvrho_bounds", None)
        name_prefix = kwargs.get("name_prefix", "")
        mean_names = kwargs.get("mean_names", None)

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

        def param_from_vectors(meanparam, covparam):
            covp = _cov_param_from_vector(covparam)
            meanp = _mean_param_from_vector(meanparam if mpl > 0 else gnp.zeros(0))
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
        return lambda model, xi, zi: noisy_outputs_parameters_initial_guess(
            model, xi, zi, output_dim=self.output_dim
        )

    def build_selection_criterion(self, output_idx: int, **build_params):
        def reml_criterion_noisy(model, covparam, xi, zi):
            return model.negative_log_restricted_likelihood(covparam, xi, zi)

        return reml_criterion_noisy

