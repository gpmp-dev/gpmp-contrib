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
        self,
        name,
        output_dim,
        mean_specification,
        covariance_specification,
        gamma=None,
        sigma2_coverage=None,
    ):
        self.gamma = gamma
        self.sigma2_coverage = sigma2_coverage
        super().__init__(
            name,
            output_dim,
            mean_specification=mean_specification,
            covariance_specification=covariance_specification,
        )

    def build_selection_criterion(self, output_idx: int, **build_params):
        gamma_user = build_params.get("gamma", self.gamma)
        sigma2_coverage_user = build_params.get(
            "sigma2_coverage", self.sigma2_coverage
        )
        cache = {"log_sigma2_0": None, "xi_id": None, "zi_id": None}

        def remap_criterion(model, covparam, xi, zi):
            defaults = gp.config.get_prior_defaults_from_dataset(xi)
            gamma = defaults["gamma"] if gamma_user is None else float(gamma_user)
            sigma2_coverage = (
                defaults["sigma2_coverage"]
                if sigma2_coverage_user is None
                else float(sigma2_coverage_user)
            )
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
                sigma2_coverage=sigma2_coverage,
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
        gamma=None,
        sigma2_coverage=None,
        alpha=None,
        rho_min_range_factor=None,
        logrho_min=None,
        logrho_0=None,
    ):
        self.gamma = gamma
        self.sigma2_coverage = sigma2_coverage
        self.alpha = alpha
        self.rho_min_range_factor = (
            None if rho_min_range_factor is None else float(rho_min_range_factor)
        )
        self.logrho_min = logrho_min
        self.logrho_0 = logrho_0
        super().__init__(
            name,
            output_dim,
            mean_specification=mean_specification,
            covariance_specification=covariance_specification,
        )

    def build_selection_criterion(self, output_idx: int, **build_params):
        gamma_user = build_params.get("gamma", self.gamma)
        sigma2_coverage_user = build_params.get(
            "sigma2_coverage", self.sigma2_coverage
        )
        alpha_user = build_params.get("alpha", self.alpha)
        rho_min_range_factor_user = build_params.get(
            "rho_min_range_factor", self.rho_min_range_factor
        )
        logrho_min_user = build_params.get("logrho_min", self.logrho_min)
        logrho_0_user = build_params.get("logrho_0", self.logrho_0)
        cache = {"covparam0": None, "xi_id": None, "zi_id": None}

        def remap_criterion(model, covparam, xi, zi):
            defaults = gp.config.get_default_prior_hyperparameters(xi)
            gamma = defaults["gamma"] if gamma_user is None else float(gamma_user)
            sigma2_coverage = (
                defaults["sigma2_coverage"]
                if sigma2_coverage_user is None
                else float(sigma2_coverage_user)
            )
            alpha = defaults["alpha"] if alpha_user is None else float(alpha_user)
            rho_min_range_factor = (
                defaults["rho_min_range_factor"]
                if rho_min_range_factor_user is None
                else float(rho_min_range_factor_user)
            )
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
                sigma2_coverage=sigma2_coverage,
                logrho_min=logrho_min,
                logrho_0=logrho_0,
                alpha=alpha,
            )

        return remap_criterion


# Alias: default REMAP model uses gaussian logsigma2 + logrho prior.
Model_ConstantMean_Maternp_REMAP = (
    Model_ConstantMean_Maternp_REMAP_gaussian_logsigma2_and_logrho_prior
)
