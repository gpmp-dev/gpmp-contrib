"""Sequential Prediction Module

This module contains the `SequentialPrediction` class, which
facilitates sequential predictions with Gaussian process (GP) models. It
allows building and storing GP models, managing datasets, appending
new evaluations, making predictions, and simulating conditional sample
paths.

Classes
-------
SequentialPrediction
    A class that encapsulates the functionality for sequential
    predictions using GP models.  It manages datasets, GP models, and
    performs various operations like updating models, making
    predictions, and generating conditional sample paths.


Author: Emmanuel Vazquez <emmanuel.vazquez@centralesupelec.fr>
Copyright (c) 2022-2023, CentraleSupelec
License: GPLv3 (see LICENSE)

"""
import gpmp.num as gnp
import gpmp as gp
from gpmpcontrib.models import Model

class SequentialPrediction:
    def __init__(self, model):
        if not isinstance(model, Model):
            raise TypeError("model must be an instance of Model or its subclasses")

        self.model = model
        self.xi = None
        self.zi = None
        self.force_param_initial_guess = True
        self.n_samplepaths = None
        self.xtsim = None
        self.xtsim_xi_ind = None
        self.xtsim_xt_ind = None
        self.zsim = None
        self.zpsim = None

    @property
    def models(self):
        return self.model.models

    @property
    def name(self):
        return self.model.name

    @property
    def output_dim(self):
        return self.model.output_dim
    
    def set_data(self, xi, zi):
        self.xi = gnp.asarray(xi)
        self.zi = gnp.asarray(zi).reshape(zi.shape[0], -1)  # Ensure zi is a matrix

    def set_data_with_model_selection(self, xi, zi):
        self.set_data(xi, zi)
        self.update_params()

    def set_new_eval(self, xnew, znew):
        xnew_ = gnp.asarray(xnew)
        znew_ = gnp.asarray(znew)
        self.xi = gnp.vstack((self.xi, xnew_))
        self.zi = gnp.vstack((self.zi, znew_.reshape(-1, self.output_dim)))

    def set_new_eval_with_model_selection(self, xnew, znew):
        self.set_new_eval(xnew, znew)
        self.update_params()

    def update_params(self):
        if self.xi is not None and self.zi is not None:
            self.model.select_params(self.xi, self.zi, force_param_initial_guess=self.force_param_initial_guess)

    def predict(self, xt, convert_out=True):
        if self.zi.ndim == 1:
            self.zi = self.zi.reshape(-1, 1)
        return self.model.predict(self.xi, self.zi, xt, convert_out=convert_out)

    def compute_conditional_simulations(self, xt):
        if self.zi.ndim == 1:
            self.zi = self.zi.reshape(-1, 1)
        return self.model.compute_conditional_simulations(self.xi, self.zi, xt)

# class SequentialPrediction:
#     """A class for managing and using Gaussian Process (GP) models for
#     sequential predictions.

#     This class is designed to simplify the process of working with GP
#     models for sequential predictions. It allows for the building and
#     storing of GP models, managing datasets, appending new
#     evaluations, making predictions, and simulating conditional sample
#     paths.

#     Attributes
#     ----------
#     xi : ndarray
#         The input data points.
#     zi : ndarray
#         The corresponding output values for the input data points.
#     output_dim : int
#         The dimension of the output space.
#     models : list of dicts
#         The list of GP models and their corresponding details.
#     force_param_initial_guess : bool
#         A flag to force using an initial guess for model parameters
#         at each new evaluation.
#     n_samplepaths : int
#         Number of conditional sample paths to simulate.
#     xtsim : ndarray
#         Simulation points for conditional sample paths.
#     xtsim_xi_ind : ndarray
#         Indices for xi in xtsim.
#     xtsim_xt_ind : ndarray
#         Indices for xt in xtsim.
#     zsim : ndarray
#         Unconditional sample paths.
#     zpsim : ndarray
#         Conditional sample paths.

#     Methods
#     -------
#     __init__(output_dim=1, models=None)
#         Initializes the SequentialPrediction instance.
#     build_models(models=None)
#         Builds GP models based on the provided models or default settings.
#     set_data(xi, zi)
#         Sets the dataset for the model.
#     set_data_with_model_selection(xi, zi)
#         Sets the dataset and updates model parameters.
#     set_new_eval(xnew, znew)
#         Adds a new evaluation to the dataset.
#     set_new_eval_with_model_selection(xnew, znew)
#         Adds a new evaluation and updates model parameters.
#     update_params()
#         Updates the parameters of the GP models.
#     predict(xt)
#         Makes predictions at the given points.
#     compute_conditional_simulations(...)
#         Generates conditional sample paths based on the current model and data.

#     """

#     def __init__(self, output_dim=1, models=None):
#         """Initializes the SequentialPrediction instance with specified output dimension and models."""

#         # dataset
#         # xi : ndarray(ni, d)
#         # zi : ndarray(ni, output_dim)
#         self.xi = None
#         self.zi = None

#         # initialize models
#         self.output_dim = output_dim
#         self.build_models(models)

#         # parameter selection
#         self.force_param_initial_guess = False

#         # unconditional & conditional simulations
#         self.n_samplepaths = None
#         self.xtsim = None
#         self.xtsim_xi_ind = None
#         self.xtsim_xt_ind = None
#         self.zsim = None
#         self.zpsim = None

#     def __repr__(self):
#         info = {
#             "input_dim": self.xi.shape[1],
#             "output_dim": self.zi.shape[1],
#             "data_size": self.xi.shape[0],
#         }
#         return str(info)

#     def build_models(self, models):
#         """Builds GP models based on the provided models or default settings."""
#         if models is None:
#             self.models = [
#                 {
#                     "name": "default",
#                     "model": None,
#                     "parameters_initial_guess_procedure": None,
#                     "parameters_initial_guess_procedure_builder": None,
#                     "parameters_initial_guess_procedure_builder_args": None,
#                     "selection_criterion": None,
#                     "selection_criterion_builder": None,
#                     "selection_criterion_builder_args": None,
#                     "info": None,
#                 }
#                 for i in range(self.output_dim)
#             ]

#             for i in range(self.output_dim):
#                 self.models[i]["model"] = gp.core.Model(
#                     self.constant_mean, self.default_covariance, None, None
#                 )
#                 self.models[i][
#                     "parameters_initial_guess_procedure"
#                 ] = gp.kernel.anisotropic_parameters_initial_guess
#                 self.models[i]["selection_criterion"] = self.models[i][
#                     "model"
#                 ].negative_log_restricted_likelihood
#         else:
#             self.models = models

#     def constant_mean(self, x, param):
#         return gnp.ones((x.shape[0], 1))

#     def default_covariance(self, x, y, covparam, pairwise=False):
#         p = 2
#         return gp.kernel.maternp_covariance(x, y, p, covparam, pairwise)

#     def set_data(self, xi, zi):
#         self.xi = xi
#         self.zi = zi.reshape(zi.shape[0], -1)  # self.zi is a matrix
#         # even if zi is a vector

#     def set_data_with_model_selection(self, xi, zi):
#         self.set_data(xi, zi)
#         self.update_params()

#     def set_new_eval(self, xnew, znew):
#         self.xi = np.vstack((self.xi, xnew))
#         self.zi = np.vstack((self.zi.reshape(self.zi.shape[0], -1), znew))

#     def set_new_eval_with_model_selection(self, xnew, znew):
#         self.set_new_eval(xnew, znew)
#         self.update_params()

#     def parameters_initial_guess(self, model_index):
#         model = self.models[model_index]

#         if callable(model["parameters_initial_guess_procedure"]):
#             initial_guess_procedure = model["parameters_initial_guess_procedure"]
#         elif callable(model["parameters_initial_guess_procedure_builder"]):
#             builder = model["parameters_initial_guess_procedure_builder"]
#             initial_guess_procedure = builder(
#                 model["parameters_initial_guess_builder_args"]
#             )
#         else:
#             raise ValueError("Invalid initial guess procedure setting")

#         covparam0 = initial_guess_procedure(
#             self.models[i]["model"], self.xi, self.zi[:, i]
#         )

#         return gnp.asarray(covparam0)

#     def make_selection_criterion_with_gradient(self):
#         crit, dcrit = gp.kernel.make_selection_criterion_with_gradient(
#             self.models[i]["selection_criterion"], self.xi, self.zi[:, i]
#         )
#         return crit, dcrit
        
#     def update_params(self):
#         """Parameter selection"""

#         for i in range(self.output_dim):
#             tic = time.time()

#             if (
#                 self.models[i]["model"].covparam is None
#                 or self.force_param_initial_guess
#             ):
#                 covparam0 = self.parameters_initial_guess(i)
#             else:
#                 covparam0 = self.models[i]["model"].covparam

#             covparam0 = gnp.asarray(covparam0)

#             crit, dcrit = self.make_selection_criterion_with_gradient(i)

#             covparam, info = gp.kernel.autoselect_parameters(
#                 covparam0, crit, dcrit, silent=True, info=True
#             )

#             self.models[i]["model"].covparam = gnp.asarray(covparam)

#             self.models[i]["info"] = info

#             self.models[i]["info"]["covparam0"] = covparam0
#             self.models[i]["info"]["covparam"] = covparam
#             self.models[i]["info"]["selection_criterion"] = crit
#             self.models[i]["info"]["time"] = time.time() - tic

#     def predict(self, xt):
#         """Prediction"""
#         zpm = np.empty((xt.shape[0], self.output_dim))
#         zpv = np.empty((xt.shape[0], self.output_dim))
#         for i in range(self.output_dim):
#             zpm[:, i], zpv[:, i] = self.models[i]["model"].predict(
#                 self.xi, self.zi[:, i], xt
#             )
#         return zpm, zpv

#     def compute_conditional_simulations(
#         self,
#         compute_zsim=True,
#         n_samplepaths=0,
#         xt="None",
#         type="intersection",
#         method="chol",
#     ):
#         """Generate conditional sample paths

#         Parameters
#         ----------
#         xt : ndarray(nt, d)
#             Simulation points
#         n_samplepaths : int
#             Number of sample paths
#         type : 'intersection', 'disjoint'
#             If type is 'intersection', xi and xt may have a non-empty
#             intersection (as when xi is a subset of xt).
#             If type is 'disjoint', xi and xt must be disjoint
#         compute_zsim : compute zsim if True or else use self.zsim
#         method : method to draw unconditional sample paths

#         """
#         if compute_zsim:
#             # initialize self.xtsim and unconditional sample paths on self.xtsim
#             self.n_samplepaths = n_samplepaths
#             ni = self.xi.shape[0]
#             nt = xt.shape[0]

#             self.xtsim = np.vstack((self.xi, xt))
#             if type == "intersection":
#                 self.xtsim, indices = np.unique(self.xtsim, return_inverse=True, axis=0)
#                 self.xtsim_xi_ind = indices[0:ni]
#                 self.xtsim_xt_ind = indices[ni : (ni + nt)]
#                 n = self.xtsim.shape[0]
#             elif type == "disjoint":
#                 self.xtsim_xi_ind = np.arange(ni)
#                 self.xtsim_xt_ind = np.arange(nt) + ni
#                 n = ni + nt

#             # sample paths on xtsim
#             self.zsim = np.empty((n, n_samplepaths, self.output_dim))

#             for i in range(self.output_dim):
#                 self.zsim[:, :, i] = gnp.to_np(
#                     self.models[i]["model"].sample_paths(
#                         self.xtsim, self.n_samplepaths, method="svd"
#                     )
#                 )

#         # conditional sample paths
#         self.zpsim = np.empty((nt, n_samplepaths, self.output_dim))

#         for i in range(self.output_dim):
#             zpm, zpv, lambda_t = self.models[i]["model"].predict(
#                 self.xi,
#                 self.zi[:, i],
#                 self.xtsim[self.xtsim_xt_ind],
#                 return_lambdas=True,
#             )
#             self.zpsim[:, :, i] = gnp.to_np(
#                 self.models[i]["model"].conditional_sample_paths(
#                     self.zsim[:, :, i],
#                     self.xtsim_xi_ind,
#                     self.zi[:, i],
#                     self.xtsim_xt_ind,
#                     lambda_t,
#                 )
#             )

#         if self.output_dim == 1:
#             # drop last dimension
#             self.zpsim = self.zpsim.reshape((self.zpsim.shape[0], self.zpsim.shape[1]))
