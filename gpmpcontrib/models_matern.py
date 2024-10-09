"""Gaussian Process Models with Matérn Covariance Function

This script implements Gaussian process  models using the Matérn
covariance function. It makes it easy to configure single- or
multi-output models. The models enable parameter selection through
Maximum Likelihood (ML) and Restricted Maximum Likelihood (REML).

Author: Emmanuel Vazquez <emmanuel.vazquez@centralesupelec.fr>
Copyright: 2022-2024, CentraleSupelec
License: GPLv3 (refer to LICENSE file for usage terms)

"""

import gpmp.num as gnp
import gpmp as gp
import gpmpcontrib.modelcontainer
from math import log


# ==============================================================================
# Model_ConstantMean_Maternp_REML Class
# ==============================================================================


class Model_ConstantMean_Maternp_REML(gpmpcontrib.modelcontainer.ModelContainer):
    def __init__(self, name, output_dim, mean_params, covariance_params):
        """
        Initialize a Model.

        Parameters
        ----------
        name : str
            The name of the model.
        output_dim : int
            The number of outputs for the model.
        mean_params : dict or list of dicts
            Type of mean function to use.
        covariance_params : dict or list of dicts
            Parameters for each covariance function, including 'p'
        """
        super().__init__(
            name,
            output_dim,
            parameterized_mean=False,
            mean_params=mean_params,
            covariance_params=covariance_params,
        )

    def build_mean_function(self, output_idx: int, param: dict):
        """Build the mean function based on the mean type.

        Parameters
        ----------
        output_idx : int
            The index of the output for which the mean function
            is being created.
        param : dict
            Must contain a "type" key with value "constant" or "linear".

        Returns
        -------
        (callable, int)
            The corresponding mean function and number of parameters

        Raises
        ------
        NotImplementedError
            If the mean type is not implemented.

        """
        if "type" not in param:
            raise ValueError(f"Mean 'type' should be specified in 'param'")

        if param["type"] == "constant":
            return (gpmpcontrib.modelcontainer.mean_linpred_constant, 0)
        elif param["type"] == "linear":
            return (gpmpcontrib.modelcontainer.mean_linpred_linear, 0)
        else:
            raise NotImplementedError(f"Mean type {param['type']} not implemented")

    def build_covariance(self, output_idx: int, param: dict):
        """Create a Matérn covariance function for a specific output
        index with given parameters.

        Parameters
        ----------
        output_idx : int
            The index of the output for which the covariance function
            is being created.
        param : dict
            Additional parameters for the Matérn covariance function,
            including regularity 'p'.

        Returns
        -------
        function
            A Matern covariance function.

        """
        if ("p" not in param) or (not isinstance(param["p"], int)):
            raise ValueError(
                f"Regularity 'p' should be integer and specified in 'param'"
            )

        p = param["p"]
        # FIXME: p = params.get("p", 2)  # Default value of p if not provided

        def maternp_covariance(x, y, covparam, pairwise=False):
            # Implementation of the Matérn covariance function using p and other parameters
            return gp.kernel.maternp_covariance(x, y, p, covparam, pairwise)

        return maternp_covariance

    def build_parameters_initial_guess_procedure(self, output_idx: int, **build_param):
        """Build an initial guess procedure for anisotropic parameters.

        Parameters
        ----------
        output_dim : int
            Number of output dimensions for the model.

        Returns
        -------
        function
            A function to compute initial guesses for anisotropic parameters.
        """
        return gp.kernel.anisotropic_parameters_initial_guess

    def build_selection_criterion(self, output_idx: int, **build_params):
        def reml_criterion(model, covparam, xi, zi):
            nlrel = model.negative_log_restricted_likelihood(covparam, xi, zi)
            return nlrel

        return reml_criterion


# ==============================================================================
# Model_ConstantMean_Maternp_ML
# ==============================================================================


class Model_ConstantMean_Maternp_ML(gpmpcontrib.modelcontainer.ModelContainer):
    """GP model with a constant mean and a Matern covariance function. Parameters are estimated by ML"""

    def __init__(self, name, output_dim, covariance_params=None):
        """
        Initialize a Model.

        Parameters
        ----------
        name : str
            The name of the model.
        output_dim : int
            The number of outputs for the model.
        covariance_params : dict or list of dicts, optional
            Parameters for each covariance function, including 'p'
        """
        super().__init__(
            name,
            output_dim,
            parameterized_mean=True,
            mean_params={"type": "constant"},
            covariance_params=covariance_params,
        )

    def build_mean_function(self, output_idx: int, param: dict):
        """Build the mean function based on the mean type.

        Parameters
        ----------
        output_idx : int
            The index of the output for which the covariance function
            is being created.
        param : dict
            Must contain a "type" key with value "constant".

        Returns
        -------
        (callable, int)
            The corresponding mean function and number of parameters

        Raises
        ------
        NotImplementedError
            If the mean type is not implemented.

        """
        if "type" not in param:
            raise ValueError(f"Mean 'type' should be specified in 'param'")

        if param["type"] == "constant":
            return (gpmpcontrib.modelcontainer.mean_parameterized_constant, 1)
        else:
            raise NotImplementedError(f"Mean type {param['type']} not implemented")

    def build_covariance(self, output_idx: int, param: dict):
        """Create a Matérn covariance function for a specific output
        index with given parameters.

        Parameters
        ----------
        output_idx : int
            The index of the output for which the covariance function
            is being created.
        params : dict
            Additional parameters for the Matérn covariance function,
            including regularity 'p'.

        Returns
        -------
        function
            A Matern covariance function.

        """
        if ("p" not in param) or (not isinstance(param["p"], int)):
            raise ValueError(
                f"Regularity 'p' should be integer and specified in 'param'"
            )

        p = param["p"]

        def maternp_covariance(x, y, covparam, pairwise=False):
            # Implementation of the Matérn covariance function using p and other parameters
            return gp.kernel.maternp_covariance(x, y, p, covparam, pairwise)

        return maternp_covariance

    def build_parameters_initial_guess_procedure(self, output_idx: int, **build_param):
        return gp.kernel.anisotropic_parameters_initial_guess_constant_mean

    def build_selection_criterion(self, output_idx: int, **build_params):
        def ml_criterion(model, meanparam, covparam, xi, zi):
            nll = model.negative_log_likelihood(meanparam, covparam, xi, zi)
            return nll

        return ml_criterion


# ==============================================================================
# Model_ConstantMean_Maternp_REML_Noisy Class
# ==============================================================================

# ==============================================================================
# This section contains the kernel function for the multi-output noisy model


def build_mown_kernel(output_idx: int, **params):
    """Covariance of the observations at points given by x.
    This kernel function is specific to multi-output models with noise.

    Parameters
    ----------
    output_idx : int
        Index indicating where to fetch noise_variance.
    **params : dict
        Additional parameters for the Matérn covariance function, including regularity 'p'.

    Returns
    -------
    kernel : callable
        Covariance function
    """
    p = params.get("p", 2)  # Default value of p if not provided

    def kernel_ii_or_tt(x, param, output_idx, pairwise=False):
        # parameters
        param_dim = param.shape[0]
        sigma2 = gnp.exp(param[0])
        loginvrho = param[1:]
        d = loginvrho.shape[0]
        noise_idx = d + output_idx
        nugget = 10 * gnp.finfo(gnp.float64).eps

        if pairwise:
            # return a vector of covariances between predictands
            K = sigma2 * gnp.ones((x.shape[0],)) + x[:, noise_idx] + nugget  # nx x 0
        else:
            # return a covariance matrix between observations
            K = gnp.scaled_distance(loginvrho, x[:, :d], x[:, :d])  # nx x nx
            K = sigma2 * gp.kernel.maternp_kernel(p, K) + gnp.diag(
                x[:, noise_idx] + nugget
            )

        return K

    def kernel_it(x, y, param, pairwise=False):
        # parameters
        param_dim = param.shape[0]
        sigma2 = gnp.exp(param[0])
        loginvrho = param[1:]
        d = loginvrho.shape[0]

        if pairwise:
            # return a vector of covariances
            K = gnp.scaled_distance_elementwise(loginvrho, x[:, :d], y[:, :d])  # nx x 0
        else:
            # return a covariance matrix
            K = gnp.scaled_distance(loginvrho, x[:, :d], y[:, :d])  # nx x ny

        K = sigma2 * gp.kernel.maternp_kernel(p, K)
        return K

    def kernel(x, y, param, pairwise=False):
        if y is x or y is None:
            return kernel_ii_or_tt(x, param, output_idx, pairwise)
        else:
            return kernel_it(x, y, param, pairwise)

    return kernel


# ==============================================================================
# This section contains functions dedicated to building and configuring Gaussian
# Process (GP) models in GPmp. These functions include routines for setting up mean
# functions, covariance functions, parameter initialization procedures, and
# selection criteria essential for the construction and optimization of GP models.


def noisy_outputs_parameters_initial_guess(model, xi, zi, output_dim):
    """
    Initial guess procedure for models with noisy outputs.

    This function refines the anisotropic parameters initial guess by
    considering only the spatial dimensions of the data, excluding the
    dimensions that pertain to outputs, which may include noise or other
    modalities.

    Parameters
    ----------
    model : object
        An instance of a Gaussian process model designed to handle noisy data.
    xi : ndarray, shape (n, d+output_dim)
        Locations of the observed data points including output dimensions.
    zi : ndarray, shape (n,)
        Observed values at the data points.
    output_dim : int
        Number of output dimensions to exclude in the anisotropic parameter calculation.

    Returns
    -------
    initial_params : ndarray
        Initial guess for the anisotropic parameters, considering noise adjustments.

    """
    xi_ = gnp.asarray(xi)
    zi_ = gnp.asarray(zi).reshape(-1, 1)
    n = xi_.shape[0]
    d = xi_.shape[1] - output_dim

    # Extract the spatial dimensions by excluding the last 'output_dim' columns
    spatial_xi = xi[:, :-output_dim]

    delta = gnp.max(spatial_xi, axis=0) - gnp.min(spatial_xi, axis=0)
    rho = gnp.exp(gnp.gammaln(d / 2 + 1) / d) / (gnp.pi**0.5) * delta

    covparam = gnp.concatenate((gnp.array([log(1.0)]), -gnp.log(rho)))
    sigma2_GLS = 1.0 / n * model.norm_k_sqrd(xi_, zi_, covparam)

    return gnp.concatenate((gnp.log(sigma2_GLS), -gnp.log(rho)))


class Model_Noisy_ConstantMean_Maternp_REML(gpmpcontrib.modelcontainer.ModelContainer):
    def __init__(self, name, output_dim, mean_params, covariance_params):
        """
        Initialize a noisy Model designed to handle output-specific noise.

        Parameters
        ----------
        name : str
            The name of the model.
        output_dim : int
            The number of outputs for the model.
        mean_params : dict or list of dicts
            Type of mean function to use, expected to handle noise.
        covariance_params : dict or list of dicts
            Parameters for each covariance function, including 'p', designed for noisy data.

        """
        super().__init__(
            name,
            output_dim,
            parameterized_mean=False,
            mean_params=mean_params,
            covariance_params=covariance_params,
        )

    def build_mean_function(self, output_idx: int, param: dict):
        """Build the mean function tailored for noisy data."""
        if "type" not in param:
            raise ValueError(f"Mean 'type' should be specified in 'param'")

        if param["type"] == "constant":
            return (gpmpcontrib.modelcontainer.mean_linpred_constant, 0)
        else:
            raise NotImplementedError(f"Mean type {param['type']} not implemented")

    def build_covariance(self, output_idx: int, param: dict):
        """
        Create a Matérn covariance function for a specific output index with given parameters
        that includes handling for output-specific noise.
        """
        if ("p" not in param) or (not isinstance(param["p"], int)):
            raise ValueError(
                f"Regularity 'p' should be integer and specified in 'param'"
            )

        # Use the specialized kernel function that handles noisy outputs
        return build_mown_kernel(output_idx, **param)

    def build_parameters_initial_guess_procedure(self, output_idx: int, **build_param):
        """
        Custom initial guess procedure for models with noisy outputs using spatial data dimensions.
        """
        return lambda model, xi, zi: noisy_outputs_parameters_initial_guess(
            model, xi, zi, output_dim=self.output_dim
        )

    def build_selection_criterion(self, output_idx: int, **build_params):
        """
        Define a REML criterion that may also account for the noise parameters.
        """

        def reml_criterion_noisy(model, covparam, xi, zi):
            nlrel = model.negative_log_restricted_likelihood(covparam, xi, zi)
            return nlrel

        return reml_criterion_noisy

    # def x_append_noise_variance(self, x, noise_variance):
    #     '''Append noise variance at each position'''

    #     assert noise_variance.shape[0] == x.shape[0], 'x_append_noise_variance: size mismatch'

    #     # ndarray n x (input_dim + output_dim)
    #     x = gnp.hstack((x, noise_variance))

    #     return x


# def negative_log_restricted_penalized_likelihood(
#     model, covparam, prior_mean, prior_invcov, xi, zi
# ):
#     delta = covparam - prior_mean
#     penalization = 0.5 * gnp.einsum("i,ij,j", delta, prior_invcov, delta)
#     nlrel = model.negative_log_restricted_likelihood(covparam, xi, zi)
#     # print(f'{delta}, p = {penalization}, nlrel={nlrel}')
#     return nlrel + penalization


# def build_remap_criterion(model, prior_mean, prior_invcov):
#     prior_mean = gnp.array([0, -log(1 / 3), -log(1 / 3)])
#     prior_invcov = gnp.diag(
#         gnp.array([0, 1 / log(5 / 3) ** 2, 1 / log(5 / 3) ** 2]))

#     def remap_criterion(covparam, xi, zi):
#         nlrepl = negative_log_restricted_penalized_likelihood(
#             model, covparam, prior_mean, prior_invcov, xi, zi
#         )
#         return nlrepl

#     return remap_criterion
