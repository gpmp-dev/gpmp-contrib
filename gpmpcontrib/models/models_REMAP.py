# gpmpcontrib/models/models_REMAP.py
# --------------------------------------------------------------
# Author: Emmanuel Vazquez <emmanuel.vazquez@centralesupelec.fr>
# Copyright (c) 2022-2026, CentraleSupelec
# License: GPLv3 (see LICENSE)
# --------------------------------------------------------------
"""REMAP Matern model classes."""

import gpmp as gp

from .models_REML import Model_ConstantMean_Maternp_REML


class Model_ConstantMean_Maternp_REMAP_power_laws(Model_ConstantMean_Maternp_REML):
    def __init__(self, name, output_dim, mean_specification, covariance_specification):
        super().__init__(
            name,
            output_dim,
            mean_specification=mean_specification,
            covariance_specification=covariance_specification,
        )

    def build_selection_criterion(self, output_idx: int, **build_params):
        def remap_criterion(model, covparam, xi, zi):
            return gp.kernel.neg_log_restricted_posterior_power_laws_prior(
                model, covparam, xi, zi
            )

        return remap_criterion


class Model_ConstantMean_Maternp_REMAP_gaussian_logsigma2(
    Model_ConstantMean_Maternp_REML
):
    def __init__(
        self, name, output_dim, mean_specification, covariance_specification, gamma=1.1
    ):
        self.gamma = float(gamma)
        super().__init__(
            name,
            output_dim,
            mean_specification=mean_specification,
            covariance_specification=covariance_specification,
        )

    def build_selection_criterion(self, output_idx: int, **build_params):
        gamma = float(build_params.get("gamma", self.gamma))
        cache = {"log_sigma2_0": None, "xi_id": None, "zi_id": None}

        def remap_criterion(model, covparam, xi, zi):
            if (
                cache["log_sigma2_0"] is None
                or cache["xi_id"] != id(xi)
                or cache["zi_id"] != id(zi)
            ):
                covparam0 = gp.kernel.anisotropic_parameters_initial_guess(
                    model, xi, zi
                )
                cache["log_sigma2_0"] = covparam0[0]
                cache["xi_id"] = id(xi)
                cache["zi_id"] = id(zi)
            return gp.kernel.neg_log_restricted_posterior_gaussian_logsigma2_prior(
                model,
                covparam,
                xi,
                zi,
                log_sigma2_0=cache["log_sigma2_0"],
                gamma=gamma,
            )

        return remap_criterion


class Model_ConstantMean_Maternp_REMAP_gaussian_logsigma2_and_logrho_prior(
    Model_ConstantMean_Maternp_REML
):
    def __init__(
        self,
        name,
        output_dim,
        mean_specification,
        covariance_specification,
        gamma=1.5,
        alpha=10.0,
        rho_min_range_factor=20.0,
        logrho_min=None,
        logrho_0=None,
    ):
        self.gamma = float(gamma)
        self.alpha = float(alpha)
        self.rho_min_range_factor = float(rho_min_range_factor)
        self.logrho_min = logrho_min
        self.logrho_0 = logrho_0
        super().__init__(
            name,
            output_dim,
            mean_specification=mean_specification,
            covariance_specification=covariance_specification,
        )

    def build_selection_criterion(self, output_idx: int, **build_params):
        gamma = float(build_params.get("gamma", self.gamma))
        alpha = float(build_params.get("alpha", self.alpha))
        rho_min_range_factor = float(
            build_params.get("rho_min_range_factor", self.rho_min_range_factor)
        )
        logrho_min_user = build_params.get("logrho_min", self.logrho_min)
        logrho_0_user = build_params.get("logrho_0", self.logrho_0)
        cache = {"covparam0": None, "xi_id": None, "zi_id": None}

        def remap_criterion(model, covparam, xi, zi):
            if (
                cache["covparam0"] is None
                or cache["xi_id"] != id(xi)
                or cache["zi_id"] != id(zi)
            ):
                cache["covparam0"] = gp.kernel.anisotropic_parameters_initial_guess(
                    model, xi, zi
                )
                cache["xi_id"] = id(xi)
                cache["zi_id"] = id(zi)

            covparam0 = cache["covparam0"]
            log_sigma2_0 = covparam0[0]
            logrho_0 = -covparam0[1:] if logrho_0_user is None else logrho_0_user
            if logrho_min_user is None:
                if xi is None:
                    raise ValueError("xi must be provided when logrho_min is None.")
                logrho_min = gp.kernel.compute_logrho_min_from_xi(
                    xi, rho_min_range_factor=rho_min_range_factor
                )
            else:
                logrho_min = logrho_min_user

            return gp.kernel.neg_log_restricted_posterior_gaussian_logsigma2_and_logrho_prior(
                model,
                covparam,
                xi,
                zi,
                log_sigma2_0=log_sigma2_0,
                gamma=gamma,
                logrho_min=logrho_min,
                logrho_0=logrho_0,
                alpha=alpha,
            )

        return remap_criterion


# Alias: default REMAP model uses gaussian logsigma2 + logrho prior.
Model_ConstantMean_Maternp_REMAP = (
    Model_ConstantMean_Maternp_REMAP_gaussian_logsigma2_and_logrho_prior
)
