# gpmpcontrib/models/models_REMAP.py
# --------------------------------------------------------------
# Author: Emmanuel Vazquez <emmanuel.vazquez@centralesupelec.fr>
# Copyright (c) 2022-2026, CentraleSupelec
# License: GPLv3 (see LICENSE)
# --------------------------------------------------------------
"""REMAP Matern model classes.

This module provides REMAP-oriented model subclasses built on top of
``Model_ConstantMean_Maternp_REML``.

Public model classes
--------------------
- ``Model_ConstantMean_Maternp_REMAP_power_laws``:
  REMAP criterion with the power-law prior from ``gpmp.kernel``.
- ``Model_ConstantMean_Maternp_REMAP_logsigma2``:
  REMAP criterion with a Gaussian prior on ``log(sigma^2)``.
- ``Model_ConstantMean_Maternp_REMAP_logsigma2_and_logrho_prior``:
  REMAP criterion with a Gaussian prior on ``log(sigma^2)`` and a
  barrier-linear prior on ``logrho``.
- ``Model_ConstantMean_Maternp_REMAP``:
  Alias of ``Model_ConstantMean_Maternp_REMAP_logsigma2_and_logrho_prior``.

Prior containers
----------------
- ``LogSigma2Prior``
- ``LogSigma2AndLogRhoPrior``

These classes hold resolved prior values (per output) and expose
dictionary-like field access through ``__getitem__``.

Prior resolution policy
-----------------------
For REMAP models with configurable priors, values are stored per output and
resolved when selection criteria are (re)built from the current
``(model, xi, zi)`` context. Explicit user anchors (for example
``logsigma2_0_prior`` and ``logrho_0_prior``) take precedence; otherwise
anchors are inferred from ``covparam0_prior`` or from anisotropic initial
guesses.

"""

from typing import Any

import gpmp as gp
import gpmp.num as gnp

from .models_REML import Model_ConstantMean_Maternp_REML

_UNSET = object()


def _as_scalar(value, field_name):
    if value is None:
        return None
    dtype = gnp.get_dtype()
    arr = gnp.asarray(value, dtype=dtype)
    if arr.ndim != 0:
        raise ValueError(f"{field_name} must be scalar (0D), got shape {arr.shape}.")
    return arr.reshape(())


def _as_vector(value, field_name):
    if value is None:
        return None
    dtype = gnp.get_dtype()
    arr = gnp.asarray(value, dtype=dtype)
    if arr.ndim == 0:
        raise ValueError(f"{field_name} must be vector-like (1D), got scalar.")
    return arr.reshape(-1)


class _PriorAccess:
    def __getitem__(self, key):
        return getattr(self, key)

    def as_dict(self):
        return dict(self.__dict__)

    def __str__(self):
        items = ",\n".join(f"{k}={v!r}" for k, v in self.__dict__.items())
        return f"{type(self).__name__}\n----\n{items}"

    __repr__ = __str__


class LogSigma2Prior(_PriorAccess):
    def __init__(
        self,
        gamma=None,
        sigma2_coverage=None,
        covparam0=None,
        logsigma2_0_prior=None,
        log_sigma2_0=None,
        covparam0_param_object: Any | None = None,
    ):
        self.gamma = _as_scalar(gamma, "gamma")
        self.sigma2_coverage = _as_scalar(sigma2_coverage, "sigma2_coverage")
        self.covparam0 = _as_vector(covparam0, "covparam0")
        self.logsigma2_0_prior = _as_scalar(logsigma2_0_prior, "logsigma2_0_prior")
        self.log_sigma2_0 = _as_scalar(log_sigma2_0, "log_sigma2_0")
        self.covparam0_param_object = covparam0_param_object


class LogSigma2AndLogRhoPrior(_PriorAccess):
    def __init__(
        self,
        gamma=None,
        sigma2_coverage=None,
        alpha=None,
        rho_min_range_factor=None,
        logrho_min=None,
        covparam0=None,
        logsigma2_0_prior=None,
        logrho_0_prior=None,
        log_sigma2_0=None,
        logrho_0=None,
        covparam0_param_object: Any | None = None,
    ):
        self.gamma = _as_scalar(gamma, "gamma")
        self.sigma2_coverage = _as_scalar(sigma2_coverage, "sigma2_coverage")
        self.alpha = _as_scalar(alpha, "alpha")
        self.rho_min_range_factor = _as_scalar(
            rho_min_range_factor, "rho_min_range_factor"
        )
        self.logrho_min = _as_vector(logrho_min, "logrho_min")
        self.covparam0 = _as_vector(covparam0, "covparam0")
        self.logsigma2_0_prior = _as_scalar(logsigma2_0_prior, "logsigma2_0_prior")
        self.logrho_0_prior = _as_vector(logrho_0_prior, "logrho_0_prior")
        self.log_sigma2_0 = _as_scalar(log_sigma2_0, "log_sigma2_0")
        self.logrho_0 = _as_vector(logrho_0, "logrho_0")
        self.covparam0_param_object = covparam0_param_object


def _expand_prior_by_output(
    value,
    output_dim,
    *,
    allow_1d_per_output=False,
    sequence_is_value=False,
):
    """Normalize a prior input into a per-output list."""
    if value is None:
        return [None] * output_dim
    if isinstance(value, (list, tuple)):
        if sequence_is_value:
            # For vector-like prior fields, avoid ambiguity when len(value) == output_dim:
            # treat as per-output only for nested sequences / array-likes.
            if len(value) == int(output_dim):
                is_nested = all(
                    isinstance(v, (list, tuple))
                    or (hasattr(v, "ndim") and int(getattr(v, "ndim")) > 0)
                    for v in value
                )
                if is_nested:
                    return list(value)
            return [value] * output_dim
        if len(value) == int(output_dim):
            return list(value)
        return [value] * output_dim
    arr = gnp.asarray(value)
    if allow_1d_per_output and arr.ndim == 1 and int(arr.shape[0]) == int(output_dim):
        return [arr[i] for i in range(output_dim)]
    if arr.ndim >= 2 and int(arr.shape[-1]) == int(output_dim):
        return [arr[..., i] for i in range(output_dim)]
    return [value] * output_dim


def _check_output_idx(output_idx, output_dim):
    if output_idx is None:
        return None
    k = int(output_idx)
    if k < 0 or k >= int(output_dim):
        raise ValueError(f"output_idx must be in [0, {int(output_dim) - 1}].")
    return k


class Model_ConstantMean_Maternp_REMAP_power_laws(Model_ConstantMean_Maternp_REML):
    def __init__(self, name, output_dim, mean_specification, covariance_specification):
        super().__init__(
            name,
            output_dim,
            mean_specification=mean_specification,
            covariance_specification=covariance_specification,
        )
        self.selection_criterion_build_policy = "needs_observations"
        self.rebuild_selection_criterion_on_select_params = True

    def build_selection_criterion(self, output_idx: int, context=None, **build_params):
        def remap_criterion(model, covparam, xi, zi):
            return gp.kernel.neg_log_restricted_posterior_power_laws_prior(
                model, covparam, xi, zi
            )

        return remap_criterion


class Model_ConstantMean_Maternp_REMAP_logsigma2(Model_ConstantMean_Maternp_REML):
    def __init__(
        self,
        name,
        output_dim,
        mean_specification,
        covariance_specification,
        gamma=None,
        sigma2_coverage=None,
    ):
        super().__init__(
            name,
            output_dim,
            mean_specification=mean_specification,
            covariance_specification=covariance_specification,
        )
        self.selection_criterion_build_policy = "needs_observations"
        self.rebuild_selection_criterion_on_select_params = True
        gamma_by_output = _expand_prior_by_output(
            gamma, output_dim, allow_1d_per_output=True
        )
        sigma2_by_output = _expand_prior_by_output(
            sigma2_coverage, output_dim, allow_1d_per_output=True
        )
        for i in range(int(output_dim)):
            self.models[i]["prior"] = LogSigma2Prior(
                gamma=gamma_by_output[i],
                sigma2_coverage=sigma2_by_output[i],
            )
            self._sync_prior_derived_fields(self.models[i]["prior"])

    @staticmethod
    def _sync_prior_derived_fields(prior: LogSigma2Prior):
        if prior.logsigma2_0_prior is not None:
            prior.log_sigma2_0 = _as_scalar(prior.logsigma2_0_prior, "log_sigma2_0")
        elif prior.covparam0 is not None:
            prior.log_sigma2_0 = _as_scalar(prior.covparam0[0], "log_sigma2_0")
        else:
            prior.log_sigma2_0 = None

        if prior.covparam0 is not None and prior.log_sigma2_0 is not None:
            cov = gnp.asarray(prior.covparam0).reshape(-1)
            cov[0] = prior.log_sigma2_0
            prior.covparam0 = _as_vector(cov, "covparam0")

        # Param object is derived from covparam0 and becomes stale on edits.
        prior.covparam0_param_object = None

    @staticmethod
    def _is_resolved_prior(prior):
        return (
            isinstance(prior, LogSigma2Prior)
            and prior.gamma is not None
            and prior.sigma2_coverage is not None
            and prior.covparam0 is not None
            and prior.log_sigma2_0 is not None
        )

    def set_prior(
        self,
        *,
        gamma=_UNSET,
        sigma2_coverage=_UNSET,
        covparam0_prior=_UNSET,
        logsigma2_0_prior=_UNSET,
        output_idx=None,
    ):
        k = _check_output_idx(output_idx, self.output_dim)
        idxs = range(self.output_dim) if k is None else [k]
        gamma_values = (
            _expand_prior_by_output(gamma, self.output_dim, allow_1d_per_output=True)
            if gamma is not _UNSET
            else None
        )
        sigma2_values = (
            _expand_prior_by_output(
                sigma2_coverage, self.output_dim, allow_1d_per_output=True
            )
            if sigma2_coverage is not _UNSET
            else None
        )
        covparam0_values = (
            _expand_prior_by_output(
                covparam0_prior,
                self.output_dim,
                allow_1d_per_output=False,
                sequence_is_value=True,
            )
            if covparam0_prior is not _UNSET
            else None
        )
        logsigma2_values = (
            _expand_prior_by_output(
                logsigma2_0_prior, self.output_dim, allow_1d_per_output=True
            )
            if logsigma2_0_prior is not _UNSET
            else None
        )
        for i in idxs:
            prior = self.models[i].get("prior", None)
            if not isinstance(prior, LogSigma2Prior):
                prior = LogSigma2Prior()
            if gamma_values is not None:
                prior.gamma = _as_scalar(gamma_values[i], "gamma")
            if sigma2_values is not None:
                prior.sigma2_coverage = _as_scalar(sigma2_values[i], "sigma2_coverage")
            if covparam0_values is not None:
                prior.covparam0 = _as_vector(covparam0_values[i], "covparam0")
            if logsigma2_values is not None:
                prior.logsigma2_0_prior = _as_scalar(
                    logsigma2_values[i], "logsigma2_0_prior"
                )
            self._sync_prior_derived_fields(prior)
            self.models[i]["prior"] = prior
        return self

    def get_prior(
        self, output_idx=None, resolved=True
    ) -> LogSigma2Prior | list[LogSigma2Prior]:
        if not resolved:
            raise ValueError("Only resolved prior values are available.")
        k = _check_output_idx(output_idx, self.output_dim)
        if k is None:
            priors = [self.models[i].get("prior", None) for i in range(self.output_dim)]
            if any(not self._is_resolved_prior(p) for p in priors):
                raise ValueError(
                    "Resolved priors are not available yet. Run select_params first."
                )
            return priors
        prior_k = self.models[k].get("prior", None)
        if not self._is_resolved_prior(prior_k):
            raise ValueError(
                f"Resolved prior for output {k} is not available yet. Run select_params first."
            )
        return prior_k

    def _resolve_prior_for_output(self, output_idx, context, build_params):
        prior_current = self.models[output_idx].get("prior", None)
        if not isinstance(prior_current, LogSigma2Prior):
            prior_current = LogSigma2Prior()
        gamma_user = build_params.get("gamma", prior_current.gamma)
        sigma2_coverage_user = build_params.get(
            "sigma2_coverage", prior_current.sigma2_coverage
        )
        covparam0_prior_user = build_params.get(
            "covparam0_prior", prior_current.covparam0
        )
        logsigma2_0_prior_user = build_params.get(
            "logsigma2_0_prior", prior_current.logsigma2_0_prior
        )

        model_build = None if context is None else context.model
        xi_build = None if context is None else context.xi
        zi_build = None if context is None else context.zi
        if model_build is None or xi_build is None or zi_build is None:
            raise ValueError(
                "context with model, xi and zi must be provided to build REMAP criterion."
            )

        defaults = gp.kernel.prior_defaults.get_default_prior_hyperparameters(xi_build)
        gamma = (
            defaults["gamma"]
            if gamma_user is None
            else float(gnp.to_scalar(_as_scalar(gamma_user, "gamma")))
        )
        sigma2_coverage = (
            defaults["sigma2_coverage"]
            if sigma2_coverage_user is None
            else float(
                gnp.to_scalar(_as_scalar(sigma2_coverage_user, "sigma2_coverage"))
            )
        )

        if covparam0_prior_user is None:
            covparam0_prior = gp.kernel.anisotropic_parameters_initial_guess(
                model_build, xi_build, zi_build
            )
        else:
            covparam0_prior = _as_vector(covparam0_prior_user, "covparam0")

        log_sigma2_0 = (
            covparam0_prior[0]
            if logsigma2_0_prior_user is None
            else _as_scalar(logsigma2_0_prior_user, "logsigma2_0_prior")
        )
        covparam0_prior_resolved = gnp.asarray(covparam0_prior).reshape(-1)
        covparam0_prior_resolved[0] = log_sigma2_0

        prior = LogSigma2Prior(
            gamma=float(gamma),
            sigma2_coverage=float(sigma2_coverage),
            covparam0=covparam0_prior_resolved,
            logsigma2_0_prior=logsigma2_0_prior_user,
            log_sigma2_0=gnp.asarray(log_sigma2_0).reshape(()),
        )
        self._sync_prior_derived_fields(prior)
        pf = self.models[output_idx].get("param_from_vectors", None)
        if callable(pf):
            mpl = int(self.models[output_idx]["mean_paramlength"])
            mean0 = gnp.asarray([]) if mpl == 0 else gnp.zeros(mpl)
            prior.covparam0_param_object = pf(mean0, prior.covparam0)
        return prior

    def build_selection_criterion(self, output_idx: int, context=None, **build_params):
        prior = self._resolve_prior_for_output(output_idx, context, build_params)
        self.models[output_idx]["prior"] = prior

        log_sigma2_0 = prior.log_sigma2_0
        gamma = prior.gamma
        sigma2_coverage = prior.sigma2_coverage

        def remap_criterion(model, covparam, xi, zi):
            return gp.kernel.neg_log_restricted_posterior_gaussian_logsigma2_prior(
                model,
                covparam,
                xi,
                zi,
                log_sigma2_0=log_sigma2_0,
                gamma=gamma,
                sigma2_coverage=sigma2_coverage,
            )

        return remap_criterion


class Model_ConstantMean_Maternp_REMAP_logsigma2_and_logrho_prior(
    Model_ConstantMean_Maternp_REML
):
    """
    REMAP Matern model with Gaussian prior on ``log(sigma^2)`` and barrier prior on ``logrho``.

    Parameters
    ----------
    name : str
        Model name.
    output_dim : int
        Number of outputs.
    mean_specification : dict | list[dict]
        Mean-function specification forwarded to ``ModelContainer``.
    covariance_specification : dict | list[dict]
        Covariance-function specification forwarded to ``ModelContainer``.
    gamma : float, optional
        Multiplicative calibration factor for the Gaussian prior on ``log(sigma^2)``.
    sigma2_coverage : float, optional
        Coverage probability used to calibrate the Gaussian prior on ``log(sigma^2)``.
    alpha : float, optional
        Right-tail slope parameter for the ``logrho`` barrier-linear prior.
    rho_min_range_factor : float, optional
        Safeguard factor used when inferring ``logrho_min`` from observation points.
    logrho_min : array_like, optional
        Optional fixed lower bound for ``logrho``. If None, it is inferred from data.
    covparam0_prior : array_like, optional
        Optional prior anchor in covariance-parameter space
        ``[log(sigma^2), loginvrho_1, ...]`` used to derive missing prior centers.
    logsigma2_0_prior : float, optional
        Optional direct prior center for ``log(sigma^2)``.
    logrho_0_prior : array_like, optional
        Optional direct prior center for ``logrho``.

    Notes
    -----
    Precedence for prior centers:
    1. ``logsigma2_0_prior`` / ``logrho_0_prior`` when provided.
    2. Otherwise derived from ``covparam0_prior`` when provided.
    3. Otherwise derived from anisotropic initial guess on current data.
    """

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
        covparam0_prior=None,
        logsigma2_0_prior=None,
        logrho_0_prior=None,
    ):
        super().__init__(
            name,
            output_dim,
            mean_specification=mean_specification,
            covariance_specification=covariance_specification,
        )
        self.selection_criterion_build_policy = "needs_observations"
        self.rebuild_selection_criterion_on_select_params = True
        rho_min_range_factor = (
            None if rho_min_range_factor is None else float(rho_min_range_factor)
        )
        gamma_by_output = _expand_prior_by_output(
            gamma, output_dim, allow_1d_per_output=True
        )
        sigma2_by_output = _expand_prior_by_output(
            sigma2_coverage, output_dim, allow_1d_per_output=True
        )
        alpha_by_output = _expand_prior_by_output(
            alpha, output_dim, allow_1d_per_output=True
        )
        rho_min_factor_by_output = _expand_prior_by_output(
            rho_min_range_factor, output_dim, allow_1d_per_output=True
        )
        logrho_min_by_output = _expand_prior_by_output(
            logrho_min, output_dim, allow_1d_per_output=False, sequence_is_value=True
        )
        covparam0_by_output = _expand_prior_by_output(
            covparam0_prior,
            output_dim,
            allow_1d_per_output=False,
            sequence_is_value=True,
        )
        logsigma2_by_output = _expand_prior_by_output(
            logsigma2_0_prior, output_dim, allow_1d_per_output=True
        )
        logrho0_by_output = _expand_prior_by_output(
            logrho_0_prior,
            output_dim,
            allow_1d_per_output=False,
            sequence_is_value=True,
        )
        for i in range(int(output_dim)):
            self.models[i]["prior"] = LogSigma2AndLogRhoPrior(
                gamma=gamma_by_output[i],
                sigma2_coverage=sigma2_by_output[i],
                alpha=alpha_by_output[i],
                rho_min_range_factor=rho_min_factor_by_output[i],
                logrho_min=logrho_min_by_output[i],
                covparam0=covparam0_by_output[i],
                logsigma2_0_prior=logsigma2_by_output[i],
                logrho_0_prior=logrho0_by_output[i],
            )
            self._sync_prior_derived_fields(self.models[i]["prior"])

    @staticmethod
    def _sync_prior_derived_fields(prior: LogSigma2AndLogRhoPrior):
        log_sigma2 = (
            None
            if prior.logsigma2_0_prior is None
            else _as_scalar(prior.logsigma2_0_prior, "logsigma2_0_prior")
        )
        logrho_0 = (
            None
            if prior.logrho_0_prior is None
            else _as_vector(prior.logrho_0_prior, "logrho_0_prior")
        )
        if prior.covparam0 is not None:
            cov = _as_vector(prior.covparam0, "covparam0")
            if log_sigma2 is None:
                log_sigma2 = _as_scalar(cov[0], "log_sigma2_0")
            if logrho_0 is None:
                logrho_0 = _as_vector(-cov[1:], "logrho_0")

        prior.log_sigma2_0 = (
            None if log_sigma2 is None else _as_scalar(log_sigma2, "log_sigma2_0")
        )
        prior.logrho_0 = None if logrho_0 is None else _as_vector(logrho_0, "logrho_0")

        if prior.log_sigma2_0 is not None and prior.logrho_0 is not None:
            prior.covparam0 = _as_vector(
                gnp.concatenate(
                    (
                        gnp.asarray([prior.log_sigma2_0]).reshape(-1),
                        -gnp.asarray(prior.logrho_0).reshape(-1),
                    )
                ),
                "covparam0",
            )

        # Param object is derived from covparam0 and becomes stale on edits.
        prior.covparam0_param_object = None

    @staticmethod
    def _is_resolved_prior(prior):
        return (
            isinstance(prior, LogSigma2AndLogRhoPrior)
            and prior.gamma is not None
            and prior.sigma2_coverage is not None
            and prior.alpha is not None
            and prior.rho_min_range_factor is not None
            and prior.logrho_min is not None
            and prior.covparam0 is not None
            and prior.log_sigma2_0 is not None
            and prior.logrho_0 is not None
        )

    def set_prior(
        self,
        *,
        gamma=_UNSET,
        sigma2_coverage=_UNSET,
        alpha=_UNSET,
        rho_min_range_factor=_UNSET,
        logrho_min=_UNSET,
        covparam0_prior=_UNSET,
        logsigma2_0_prior=_UNSET,
        logrho_0_prior=_UNSET,
        output_idx=None,
    ):
        k = _check_output_idx(output_idx, self.output_dim)
        idxs = range(self.output_dim) if k is None else [k]
        gamma_values = (
            _expand_prior_by_output(gamma, self.output_dim, allow_1d_per_output=True)
            if gamma is not _UNSET
            else None
        )
        sigma2_values = (
            _expand_prior_by_output(
                sigma2_coverage, self.output_dim, allow_1d_per_output=True
            )
            if sigma2_coverage is not _UNSET
            else None
        )
        alpha_values = (
            _expand_prior_by_output(alpha, self.output_dim, allow_1d_per_output=True)
            if alpha is not _UNSET
            else None
        )
        rho_min_factor_values = (
            _expand_prior_by_output(
                rho_min_range_factor, self.output_dim, allow_1d_per_output=True
            )
            if rho_min_range_factor is not _UNSET
            else None
        )
        logrho_min_values = (
            _expand_prior_by_output(
                logrho_min,
                self.output_dim,
                allow_1d_per_output=False,
                sequence_is_value=True,
            )
            if logrho_min is not _UNSET
            else None
        )
        covparam0_values = (
            _expand_prior_by_output(
                covparam0_prior,
                self.output_dim,
                allow_1d_per_output=False,
                sequence_is_value=True,
            )
            if covparam0_prior is not _UNSET
            else None
        )
        logsigma2_values = (
            _expand_prior_by_output(
                logsigma2_0_prior, self.output_dim, allow_1d_per_output=True
            )
            if logsigma2_0_prior is not _UNSET
            else None
        )
        logrho0_values = (
            _expand_prior_by_output(
                logrho_0_prior,
                self.output_dim,
                allow_1d_per_output=False,
                sequence_is_value=True,
            )
            if logrho_0_prior is not _UNSET
            else None
        )

        for i in idxs:
            prior = self.models[i].get("prior", None)
            if not isinstance(prior, LogSigma2AndLogRhoPrior):
                prior = LogSigma2AndLogRhoPrior()
            if gamma_values is not None:
                prior.gamma = _as_scalar(gamma_values[i], "gamma")
            if sigma2_values is not None:
                prior.sigma2_coverage = _as_scalar(sigma2_values[i], "sigma2_coverage")
            if alpha_values is not None:
                prior.alpha = _as_scalar(alpha_values[i], "alpha")
            if rho_min_factor_values is not None:
                prior.rho_min_range_factor = _as_scalar(
                    rho_min_factor_values[i], "rho_min_range_factor"
                )
            if logrho_min_values is not None:
                prior.logrho_min = _as_vector(logrho_min_values[i], "logrho_min")
            if covparam0_values is not None:
                prior.covparam0 = _as_vector(covparam0_values[i], "covparam0")
            if logsigma2_values is not None:
                prior.logsigma2_0_prior = _as_scalar(
                    logsigma2_values[i], "logsigma2_0_prior"
                )
            if logrho0_values is not None:
                prior.logrho_0_prior = _as_vector(logrho0_values[i], "logrho_0_prior")
            self._sync_prior_derived_fields(prior)
            self.models[i]["prior"] = prior
        return self

    def get_prior(
        self, output_idx=None, resolved=True
    ) -> LogSigma2AndLogRhoPrior | list[LogSigma2AndLogRhoPrior]:
        if not resolved:
            raise ValueError("Only resolved prior values are available.")
        k = _check_output_idx(output_idx, self.output_dim)
        if k is None:
            priors = [self.models[i].get("prior", None) for i in range(self.output_dim)]
            if any(not self._is_resolved_prior(p) for p in priors):
                raise ValueError(
                    "Resolved priors are not available yet. Run select_params first."
                )
            return priors
        prior_k = self.models[k].get("prior", None)
        if not self._is_resolved_prior(prior_k):
            raise ValueError(
                f"Resolved prior for output {k} is not available yet. Run select_params first."
            )
        return prior_k

    def _resolve_prior_for_output(self, output_idx, context, build_params):
        prior_current = self.models[output_idx].get("prior", None)
        if not isinstance(prior_current, LogSigma2AndLogRhoPrior):
            prior_current = LogSigma2AndLogRhoPrior()
        gamma_candidate = build_params.get("gamma", prior_current.gamma)
        sigma2_coverage_candidate = build_params.get(
            "sigma2_coverage", prior_current.sigma2_coverage
        )
        alpha_candidate = build_params.get("alpha", prior_current.alpha)
        rho_min_range_factor_candidate = build_params.get(
            "rho_min_range_factor", prior_current.rho_min_range_factor
        )
        logrho_min_candidate = build_params.get("logrho_min", prior_current.logrho_min)
        covparam0_prior_candidate = build_params.get(
            "covparam0_prior", prior_current.covparam0
        )
        logsigma2_0_prior_candidate = build_params.get(
            "logsigma2_0_prior", prior_current.logsigma2_0_prior
        )
        logrho_0_prior_candidate = build_params.get(
            "logrho_0_prior",
            build_params.get("logrho_0", prior_current.logrho_0_prior),
        )

        model_build = None if context is None else context.model
        xi_build = None if context is None else context.xi
        zi_build = None if context is None else context.zi
        if model_build is None or xi_build is None or zi_build is None:
            raise ValueError(
                "context with model, xi and zi must be provided to build REMAP criterion."
            )

        defaults = gp.kernel.prior_defaults.get_default_prior_hyperparameters(xi_build)
        gamma = (
            defaults["gamma"]
            if gamma_candidate is None
            else float(gnp.to_scalar(_as_scalar(gamma_candidate, "gamma")))
        )
        sigma2_coverage = (
            defaults["sigma2_coverage"]
            if sigma2_coverage_candidate is None
            else float(
                gnp.to_scalar(_as_scalar(sigma2_coverage_candidate, "sigma2_coverage"))
            )
        )
        alpha = (
            defaults["alpha"]
            if alpha_candidate is None
            else float(gnp.to_scalar(_as_scalar(alpha_candidate, "alpha")))
        )
        rho_min_range_factor = (
            defaults["rho_min_range_factor"]
            if rho_min_range_factor_candidate is None
            else float(
                gnp.to_scalar(
                    _as_scalar(rho_min_range_factor_candidate, "rho_min_range_factor")
                )
            )
        )

        log_sigma2_0 = (
            None
            if logsigma2_0_prior_candidate is None
            else _as_scalar(logsigma2_0_prior_candidate, "logsigma2_0_prior")
        )
        logrho_0 = (
            None
            if logrho_0_prior_candidate is None
            else _as_vector(logrho_0_prior_candidate, "logrho_0_prior")
        )
        if log_sigma2_0 is None or logrho_0 is None:
            if covparam0_prior_candidate is None:
                covparam0_prior = gp.kernel.anisotropic_parameters_initial_guess(
                    model_build, xi_build, zi_build
                )
            else:
                covparam0_prior = _as_vector(covparam0_prior_candidate, "covparam0")
            if log_sigma2_0 is None:
                log_sigma2_0 = covparam0_prior[0]
            if logrho_0 is None:
                logrho_0 = -covparam0_prior[1:]

        covparam0_prior_resolved = gnp.concatenate(
            (
                gnp.asarray([log_sigma2_0]).reshape(-1),
                -gnp.asarray(logrho_0).reshape(-1),
            )
        )
        if logrho_min_candidate is None:
            logrho_min = gp.kernel.compute_logrho_min_from_xi(
                xi_build,
                prior_rho_min_range_factor=rho_min_range_factor,
            )
        else:
            logrho_min = _as_vector(logrho_min_candidate, "logrho_min")

        prior = LogSigma2AndLogRhoPrior(
            gamma=float(gamma),
            sigma2_coverage=float(sigma2_coverage),
            alpha=float(alpha),
            rho_min_range_factor=float(rho_min_range_factor),
            logrho_min=gnp.asarray(logrho_min).reshape(-1),
            covparam0=covparam0_prior_resolved,
            logsigma2_0_prior=logsigma2_0_prior_candidate,
            logrho_0_prior=logrho_0_prior_candidate,
            log_sigma2_0=gnp.asarray(log_sigma2_0).reshape(()),
            logrho_0=gnp.asarray(logrho_0).reshape(-1),
        )
        self._sync_prior_derived_fields(prior)
        pf = self.models[output_idx].get("param_from_vectors", None)
        if callable(pf):
            mpl = int(self.models[output_idx]["mean_paramlength"])
            mean0 = gnp.asarray([]) if mpl == 0 else gnp.zeros(mpl)
            prior.covparam0_param_object = pf(mean0, prior.covparam0)
        return prior

    def build_selection_criterion(self, output_idx: int, context=None, **build_params):
        prior = self._resolve_prior_for_output(output_idx, context, build_params)
        self.models[output_idx]["prior"] = prior

        log_sigma2_0 = prior.log_sigma2_0
        gamma = prior.gamma
        sigma2_coverage = prior.sigma2_coverage
        logrho_min = prior.logrho_min
        logrho_0 = prior.logrho_0
        alpha = prior.alpha

        def remap_criterion(model, covparam, xi, zi):
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
    Model_ConstantMean_Maternp_REMAP_logsigma2_and_logrho_prior
)
