"""Sequential prediction object

Author: Emmanuel Vazquez <emmanuel.vazquez@centralesupelec.fr>
Copyright (c) 2022-2023, CentraleSupelec
License: GPLv3 (see LICENSE)

"""
import time
import numpy as np
import gpmp.num as gnp
import gpmp as gp


class SequentialPrediction:
    """Sequential predictions made easier:
    - builds and stores GP models
    - stores the dataset (xi, zi) of size ni x d, ni x p
    - append new evaluations
    - make predicions
    - simulate conditional sample paths
    """

    def __init__(self, output_dim=1, models=None):
        """properties initialization"""

        # dataset
        # xi : ndarray(ni, d)
        # zi : ndarray(ni, output_dim)
        self.xi = None
        self.zi = None

        # initialize models
        self.output_dim = output_dim
        self.build_models(models)

        # parameter selection
        self.force_param_initial_guess = False

        # unconditional & conditional simulations
        self.n_samplepaths = None
        self.xtsim = None
        self.xtsim_xi_ind = None
        self.xtsim_xt_ind = None
        self.zsim = None
        self.zpsim = None

    def __repr__(self):
        info = {
            "input_dim": self.xi.shape[1],
            "output_dim": self.zi.shape[1],
            "data_size": self.xi.shape[0],
        }
        return str(info)

    def build_models(self, models):
        if models is None:
            self.models = [
                {
                    "name": "",
                    "model": None,
                    "parameters_initial_guess_procedure": None,
                    "selection_criterion": None,
                    "info": None,
                }
                for i in range(self.output_dim)
            ]

            for i in range(self.output_dim):
                self.models[i]['model'] = gp.core.Model(
                    self.constant_mean, self.default_covariance, None, None
                )
                self.models[i][
                    "parameters_initial_guess_procedure"
                ] = gp.kernel.anisotropic_parameters_initial_guess
                self.models[i][
                    "selection_criterion"
                ] = self.models[i]['model'].negative_log_restricted_likelihood
        else:
            self.models = models

    def constant_mean(self, x, param):
        return gnp.ones((x.shape[0], 1))

    def default_covariance(self, x, y, covparam, pairwise=False):
        p = 2
        return gp.kernel.maternp_covariance(x, y, p, covparam, pairwise)

    def set_data(self, xi, zi):
        self.xi = xi
        self.zi = zi.reshape(zi.shape[0], -1)  # self.zi is a matrix
        # even if zi is a vector

    def set_data_with_model_selection(self, xi, zi):
        self.set_data(xi, zi)
        self.update_params()

    def set_new_eval(self, xnew, znew):
        self.xi = np.vstack((self.xi, xnew))
        self.zi = np.vstack((self.zi.reshape(self.zi.shape[0], -1), znew))

    def set_new_eval_with_model_selection(self, xnew, znew):
        self.set_new_eval(xnew, znew)
        self.update_params()

    def update_params(self):
        """Parameter selection"""

        for i in range(self.output_dim):

            tic = time.time()

            if self.models[i]['model'].covparam is None or self.force_param_initial_guess:
                covparam0 = self.models[i]['parameters_initial_guess_procedure'](
                    self.models[i]['model'], self.xi, self.zi[:, i]
                )

            else:
                covparam0 = self.models[i]['model'].covparam

            covparam0 = gnp.asarray(covparam0)
            
            crit, dcrit = gp.kernel.make_selection_criterion_with_gradient(
                self.models[i]['selection_criterion'],
                self.xi,
                self.zi[:, i]
            )
            
            covparam, info = gp.kernel.autoselect_parameters(
                covparam0, crit, dcrit, silent=True, info=True
            )

            self.models[i]['model'].covparam = gnp.asarray(covparam)

            self.models[i]['info'] = info

            self.models[i]['info']['covparam0'] = covparam0
            self.models[i]['info']['covparam'] = covparam
            self.models[i]['info']['selection_criterion'] = crit
            self.models[i]['info']['time'] = time.time() - tic

    def predict(self, xt):
        """Prediction"""
        zpm = np.empty((xt.shape[0], self.output_dim))
        zpv = np.empty((xt.shape[0], self.output_dim))
        for i in range(self.output_dim):              
            zpm[:, i], zpv[:, i] = self.models[i]['model'].predict(
                self.xi, self.zi[:, i], xt
            )
        return zpm, zpv

    def compute_conditional_simulations(
            self,
            compute_zsim=True,
            n_samplepaths=0,
            xt='None',
            type='intersection',
            method='chol'
    ):
        """Generate conditional sample paths

        Parameters
        ----------
        xt : ndarray(nt, d)
            Simulation points
        n_samplepaths : int
            Number of sample paths
        type : 'intersection', 'disjoint'
            If type is 'intersection', xi and xt may have a non-empty
            intersection (as when xi is a subset of xt).
            If type is 'disjoint', xi and xt must be disjoint
        compute_zsim : compute zsim if True or else use self.zsim
        method : method to draw unconditional sample paths

        """
        if compute_zsim:
            # initialize self.xtsim and unconditional sample paths on self.xtsim
            self.n_samplepaths = n_samplepaths
            ni = self.xi.shape[0]
            nt = xt.shape[0]

            self.xtsim = np.vstack((self.xi, xt))
            if type == "intersection":
                self.xtsim, indices = np.unique(self.xtsim, return_inverse=True, axis=0)
                self.xtsim_xi_ind = indices[0:ni]
                self.xtsim_xt_ind = indices[ni : (ni + nt)]
                n = self.xtsim.shape[0]
            elif type == "disjoint":
                self.xtsim_xi_ind = np.arange(ni)
                self.xtsim_xt_ind = np.arange(nt) + ni
                n = ni + nt

            # sample paths on xtsim
            self.zsim = np.empty((n, n_samplepaths, self.output_dim))

            for i in range(self.output_dim):
                self.zsim[:, :, i] = gnp.to_np(
                    self.models[i]['model'].sample_paths(
                        self.xtsim, self.n_samplepaths, method='svd'
                    )
                )

        # conditional sample paths
        self.zpsim = np.empty((nt, n_samplepaths, self.output_dim))

        for i in range(self.output_dim):
            zpm, zpv, lambda_t = self.models[i]['model'].predict(
                self.xi,
                self.zi[:, i],
                self.xtsim[self.xtsim_xt_ind],
                return_lambdas=True
            )
            self.zpsim[:, :, i] = gnp.to_np(
                self.models[i]['model'].conditional_sample_paths(
                    self.zsim[:, :, i],
                    self.xtsim_xi_ind,
                    self.zi[:, i],
                    self.xtsim_xt_ind,
                    lambda_t,
                )
            )
            

        if self.output_dim == 1:
            # drop last dimension
            self.zpsim = self.zpsim.reshape((self.zpsim.shape[0], self.zpsim.shape[1]))
