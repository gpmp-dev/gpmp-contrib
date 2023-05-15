"""
Model specification for multi-objective stochastic optimization

Author: Emmanuel Vazquez <emmanuel.vazquez@centralesupelec.fr>
Copyright (c) 2022-2023, CentraleSupelec
License: GPLv3 (see LICENSE)

"""
import gpmp.num as gnp
import gpmp as gp
from math import log


def kernel_ii_or_tt(x, param, output_idx, pairwise=False):
    """Covariance of the observations at points given by x
    Parameters
    ----------
    x : ndarray(n, d + output_dim)
        Data points in dimension d. The last columns contain the noise
        variance at the locations for each outputs.
    param : ndarray(1 + d)
        sigma2 and range parameters
    output_idx : where to fetch noise_variance

    """
    # parameters
    p = 2
    param_dim = param.shape[0]
    sigma2 = gnp.exp(param[0])
    loginvrho = param[1:]
    d = loginvrho.shape[0]
    noise_idx = d + output_idx
    nugget = 10 * gnp.finfo(gnp.float64).eps

    if pairwise:
        # return a vector of covariances between predictands
        K = sigma2 * gnp.ones((x.shape[0],)) + \
            x[:, noise_idx] + nugget  # nx x 0
    else:
        # return a covariance matrix between observations
        K = gnp.scaled_distance(loginvrho, x[:, :d], x[:, :d])  # nx x nx
        K = sigma2 * \
            gp.kernel.maternp_kernel(p, K) + gnp.diag(x[:, noise_idx] + nugget)

    return K


def kernel_it(x, y, param, pairwise=False):
    """Covariance between observations and prediction points"""
    # parameters
    p = 2
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


def build_kernel(output_idx):

    def kernel(x, y, param, pairwise=False):

        if y is x or y is None:
            return kernel_ii_or_tt(x, param, output_idx, pairwise)
        else:
            return kernel_it(x, y, param, pairwise)

    return kernel


def constant_mean(x, param):
    return gnp.ones((x.shape[0], 1))


def build_anisotropic_parameters_initial_guess(output_dim):
    # since xi carries noise variance parameters
    
    def anisotropic_parameters_initial_guess(model, xi, zi):

        xi_ = gnp.asarray(xi)
        zi_ = gnp.asarray(zi)

        rho = gnp.std(xi_[:, :-output_dim], axis=0)
        loginvrho = -gnp.log(rho)
        covparam = gnp.concatenate((gnp.array([log(1.0)]), loginvrho))

        # n = xi.shape[0]
        # sigma2_GLS = 1 / n * model.norm_k_sqrd(
        #     xi,
        #     zi.reshape((-1, )),
        #     covparam
        # )
        # FIXME
        sigma2 = gnp.std(zi_)

        return gnp.concatenate(
            (gnp.array([gnp.log(sigma2)]), -gnp.log(rho)))

    return anisotropic_parameters_initial_guess


def negative_log_restricted_penalized_likelihood(model,
                                                 covparam,
                                                 mean_prior,
                                                 invcov_prior,
                                                 xi,
                                                 zi):

    delta = covparam - mean_prior
    penalization = 0.5 * gnp.einsum("i,ij,j", delta, invcov_prior, delta)
    nlrel = model.negative_log_restricted_likelihood(covparam, xi, zi)
    # print(f'{delta}, p = {penalization}, nlrel={nlrel}')
    return nlrel + penalization


def build_remap_criterion(model, mean_prior, invcov_prior):

    def remap_criterion(covparam, xi, zi):
        nlrepl = negative_log_restricted_penalized_likelihood(
            model,
            covparam,
            mean_prior,
            invcov_prior,
            xi,
            zi
        )     
        return nlrepl

    return remap_criterion


def build_models(output_dim):

    models = [{
        "name": "",
        "model": None,
        "parameters_initial_guess_procedure": None,
        "selection_criterion": None,
    } for i in range(output_dim)]

    # same hyper prior for all outputs

    # TODO : set mean prior for sigma^2 adaptively
    mean_prior = gnp.array([0, -log(1 / 3), -log(1 / 3)])
    invcov_prior = gnp.diag(gnp.array([0, 1 / log(5/3)**2, 1 / log(5/3)**2]))

    for i in range(output_dim):
        covariance = build_kernel(i)

        models[i]['model'] = gp.core.Model(constant_mean,
                                           covariance,
                                           None,
                                           None)
        models[i][
            "parameters_initial_guess_procedure"
        ] = build_anisotropic_parameters_initial_guess(output_dim)
        
        models[i]["selection_criterion"] = build_remap_criterion(
            models[i]['model'],
            mean_prior,
            invcov_prior
        )

    return models
