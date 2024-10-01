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
Copyright (c) 2022-2024, CentraleSupelec
License: GPLv3 (see LICENSE)

"""
import gpmp.num as gnp
import gpmp as gp
from gpmpcontrib.models import Model


class SequentialPrediction:
    def __init__(self, model):
        """Facilitates sequential predictions with Gaussian process
        (GP) models. This class supports the building, storing, and
        managing of GP models and datasets, appending new evaluations,
        making predictions, and simulating conditional sample paths.

        Attributes
        ----------
        model : Model
            An instance of the Model class or its subclasses, encapsulating the GP models.
        xi : ndarray, optional
            The input data points, updated dynamically as new evaluations are added.
        zi : ndarray, optional
            The corresponding output values for the input data points, reshaped as needed.
        force_param_initial_guess : bool
            Determines whether to force using an initial guess for model parameters during updates.
        n_samplepaths : int, optional
            Specifies the number of conditional sample paths to simulate.
        xtsim : ndarray, optional
            Points at which conditional sample paths are simulated.
        xtsim_xi_ind : ndarray, optional
            Indices of xi within xtsim, used for simulations.
        xtsim_xt_ind : ndarray, optional
            Indices of xt within xtsim, used for simulations.
        zsim : ndarray, optional
            Simulated unconditional sample paths.
        zpsim : ndarray, optional
            Simulated conditional sample paths.

        Methods
        -------
        __init__(model)
            Initializes the SequentialPrediction instance with a specified model.
        set_data(xi, zi)
            Sets the initial dataset for the model.
        set_data_with_model_selection(xi, zi)
            Sets the dataset and updates model parameters based on the new data.
        set_new_eval(xnew, znew)
            Adds a new evaluation to the dataset and adjusts the existing data arrays.
        set_new_eval_with_model_selection(xnew, znew)
            Adds a new evaluation to the dataset and updates model parameters accordingly.
        update_params()
            Updates the model parameters using the current dataset, optionally using an initial guess.
        predict(xt, convert_out=True)
            Makes predictions at the given points, optionally converting outputs.
        compute_conditional_simulations(xt)
            Generates conditional sample paths based on the current model state and data.

        """
        if not isinstance(model, Model):
            raise TypeError(
                "model must be an instance of Model or its subclasses")

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
        self.zi = gnp.asarray(zi).reshape(
            zi.shape[0], -1)  # Ensure zi is a matrix

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
            try:
                self.model.select_params(
                    self.xi,
                    self.zi,
                    force_param_initial_guess=self.force_param_initial_guess,
                )
            except:
                raise RuntimeError("Failed to update parameters")

    def predict(self, xt, convert_out=True):
        if self.zi.ndim == 1:
            self.zi = self.zi.reshape(-1, 1)
        return self.model.predict(self.xi, self.zi, xt, convert_out=convert_out)

    def compute_conditional_simulations(self, xt):
        if self.zi.ndim == 1:
            self.zi = self.zi.reshape(-1, 1)
        return self.model.compute_conditional_simulations(self.xi, self.zi, xt)
