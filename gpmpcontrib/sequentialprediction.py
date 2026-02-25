"""Sequential Prediction Module

This module implements the `SequentialPrediction` class, which
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
Copyright (c) 2022-2026, CentraleSupelec
License: GPLv3 (see LICENSE)

"""

import hashlib
import gpmp.num as gnp
import gpmp as gp
from gpmpcontrib.modelcontainer import ModelContainer


class SequentialPrediction:
    def __init__(self, model, force_param_initial_guess=True):
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
        __init__(model, force_param_initial_guess)
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
        if not isinstance(model, ModelContainer):
            raise TypeError("model must be an instance of Model or its subclasses")

        self.model = model
        self.xi = None
        self.zi = None
        self.force_param_initial_guess = force_param_initial_guess
        self.n_samplepaths = None
        self.xtsim = None
        self.xtsim_xi_ind = None
        self.xtsim_xt_ind = None
        self.zsim = None
        self.zpsim = None

        # Caching attributes
        self._cache_xt = None
        self._cache_zpm = None
        self._cache_zpv = None
        self._cache_key = None  # Hash for model parameters
        self._cache_xi = None  # Last used xi
        self._cache_zi = None  # Last used zi

    @property
    def models(self):
        return self.model.models

    @property
    def name(self):
        return self.model.name

    @property
    def input_dim(self):
        return self.xi.shape[1]

    @property
    def output_dim(self):
        return self.model.output_dim

    @property
    def ni(self):
        return self.xi.shape[0]

    def set_data(self, xi, zi):
        self.xi = gnp.asarray(xi)
        self.zi = gnp.asarray(zi).reshape(zi.shape[0], -1)  # Ensure zi is a matrix

    def set_data_with_model_selection(self, xi, zi):
        self.set_data(xi, zi)
        self.update_params()

    def set_new_eval(self, xnew, znew):
        xnew = gnp.asarray(xnew)
        if xnew.ndim == 1:
            xnew = xnew.reshape(1, -1)  # Reshape to (1, D)
        znew = gnp.asarray(znew).reshape(-1, self.output_dim)

        if self.xi is None or self.zi is None:
            self.set_data(xnew, znew)
        else:
            if xnew.shape[1] != self.xi.shape[1]:
                raise ValueError(
                    f"Input dimension mismatch: expected {self.xi.shape[1]}, got {xnew.shape[1]}"
                )
            if znew.shape[1] != self.output_dim:
                raise ValueError(
                    f"Output dimension mismatch: expected {self.output_dim}, got {znew.shape[1]}"
                )
            self.xi = gnp.vstack((self.xi, xnew))
            self.zi = gnp.vstack((self.zi, znew))

    def set_new_eval_with_model_selection(self, xnew, znew):
        self.set_new_eval(xnew, znew)
        self.update_params()

    def update_params(self):
        if self.xi is None or self.zi is None:
            raise ValueError("Data not set. Use `set_data` first.")

        try:
            self.model.select_params(
                self.xi,
                self.zi,
                force_param_initial_guess=self.force_param_initial_guess,
            )
        except Exception as e:
            raise RuntimeError(f"Failed to update parameters: {e}")

    def _compute_cache_key(self):
        """Compute a hash for model parameters."""
        state = self.model.get_state()
        return hashlib.sha256(str(state).encode()).hexdigest()

    def predict(self, xt, convert_out=True, use_cache=False):
        """Predict with caching to avoid redundant computations."""
        if self.xi is None or self.zi is None:
            raise ValueError("No data set. Use `set_data` first.")
        if self.zi.ndim == 1:
            self.zi = self.zi.reshape(-1, 1)
        xt = gnp.asarray(xt)

        if use_cache:
            # Compute new cache key for model parameters
            new_cache_key = self._compute_cache_key()

            # Check if xi, zi, xt, and model parameters remain unchanged
            if (
                self._cache_xt is not None
                and gnp.array_equal(self._cache_xt, xt)
                and self._cache_zi is not None
                and gnp.array_equal(self._cache_zi, self.zi)
                and self._cache_xi is not None
                and gnp.array_equal(self._cache_xi, self.xi)
                and self._cache_key == new_cache_key  # Model parameters hash
            ):
                return self._cache_zpm, self._cache_zpv

            # Otherwise, recompute predictions and update cache
            self._cache_xt = gnp.copy(xt)
            self._cache_zpm, self._cache_zpv = self.model.predict(
                self.xi, self.zi, self._cache_xt, convert_out=convert_out
            )
            self._cache_key = new_cache_key  # Update cache key
            self._cache_xi = gnp.copy(self.xi)  # Cache xi
            self._cache_zi = gnp.copy(self.zi)  # Cache zi

            return self._cache_zpm, self._cache_zpv
        else:
            return self.model.predict(self.xi, self.zi, xt, convert_out=convert_out)

    def compute_conditional_simulations(
        self,
        xt,
        n_samplepaths=1,
        type="intersection",
        method="svd",
        convert_in=True,
        convert_out=True,
    ):
        if self.zi.ndim == 1:
            self.zi = self.zi.reshape(-1, 1)
        return self.model.compute_conditional_simulations(
            self.xi,
            self.zi,
            xt,
            n_samplepaths=n_samplepaths,
            type=type,
            method=method,
            convert_in=convert_in,
            convert_out=convert_out,
        )
