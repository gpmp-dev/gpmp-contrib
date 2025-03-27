"""Gaussian Process (GP) Model Container

This module defines a `ModelContainer` object as a wrapper around the
`Model` object in `gpmp`, simplifying the creation and management of
Gaussian Process models. It provides tools for choosing the mean and
covariance functions, and accommodates multiple outputs. The module
supports parameter selection using user-provided methods (Maximum
Likelihood, Restricted Maximum Likelihood...).

It is used by `SequentialPrediction` and `SequentialStrategy`.

Author: Emmanuel Vazquez <emmanuel.vazquez@centralesupelec.fr>
Copyright: 2022-2025, CentraleSupelec
License: GPLv3 (refer to LICENSE file)

"""

import time
import gpmp.num as gnp
import gpmp as gp


# ==============================================================================
# ModelContainer Class
# ==============================================================================


class ModelContainer:
    def __init__(
        self,
        name,
        output_dim,
        parameterized_mean,
        mean_params,
        covariance_params,
        initial_guess_procedures=None,
        selection_criteria=None,
    ):
        """Base class for Gaussian Process (GP) models.

        This class defines GP models that can handle multiple outputs
        with distinct mean and covariance functions for each
        output. It supports parameter estimation through selection
        criteria such as Maximum Likelihood (ML) or Restricted Maximum
        Likelihood (REML), and allows for initial guess procedures for
        optimizing model parameters. It facilitates the use of the
        Model class in GPmp for making predictions or generating
        posterior sample paths.

        Parameters
        ----------
        name : str
            The name of the model.
        output_dim : int
            The number of outputs for the model.
        parameterized_mean : bool
            If True, the model uses a "parameterized" mean function. If False,
            it uses a "linear predictor" mean function.
        mean_params : list of dict or dict
            Parameters for defining mean functions. Can be a single dictionary applied to
            all outputs or a list of dictionaries, one for each output. Each dictionary includes:
            - 'function': The mean function to be used, either a callable or a string.
            - 'param_length': The length of the mean function's parameter vector, required if 'function' is a callable.
        covariance_params : dict or list of dicts
            Parameters for defining covariance functions. Each dictionary must include a key 'function'.
        initial_guess_procedures : list of callables, optional
            A list of procedures for initial guess of model parameters, one for each output.
        selection_criteria : list of callables, optional
            A list of selection criteria, one for each output.

        Attributes
        ----------
        name : str
            The name of the model.
        output_dim : int
            The number of outputs in the model.
        parameterized_mean : bool
            The type of mean function.
        mean_functions : list of callables
            A list of callable mean functions, one for each output.
        mean_functions_info : list of dict
            Descriptive information about each mean function,
            including its name and parameter length.
        covariance_functions : list of callables
            A list of callable covariance functions, one for each output.
        models : list of dict
            A list containing models for each output. Each dictionary includes:
            - output_name (str): Name of the output.
            - model (object): The Gaussian Process model instance for the output.
            - mean_fname (str): Name of the mean function used for the output.
            - mean_paramlength (int): Length of the mean function's parameter vector.
            - covariance_fname (str): Name of the covariance function
              used for the output.
            - parameters_initial_guess_procedure (callable): Initial
              guess procedure for model parameters.
            - selection_criterion (callable): Predefined selection
              criterion for the output.
            - info (dict): Additional information after model
              parameter selection.

        Methods
        -------
        __getitem__(index)
            Access individual models and their attributes by index.
        __repr__()
            Return a string representation of the model instance.
        __str__()
            Generate a formatted string representation of the model with details on mean and covariance functions.
        set_mean_functions(mean_params)
            Initialize and configure mean functions based on input parameters.
        set_covariance_functions(covariance_params)
            Initialize and configure covariance functions based on input parameters.
        set_parameters_initial_guess_procedures(initial_guess_procedures=None, build_params=None)
            Set procedures for initial guess of model parameters.
        set_selection_criteria(selection_criteria=None, build_params=None)
            Set selection criteria for parameter estimation.
        make_selection_criterion_with_gradient(model, xi_, zi_)
            Create a selection criterion with gradients for parameter optimization.
        select_params(xi, zi, force_param_initial_guess=True)
            Select parameters for the model based on input and output data.
        predict(xi, zi, xt, convert_in=True, convert_out=True)
            Predict outputs at test points using the trained model.
        compute_conditional_simulations(xi, zi, xt, n_samplepaths=1, type="intersection", method="svd", convert_in=True, convert_out=True)
            Generate conditional sample paths based on training and test data.
        build_mean_function(output_idx, param)
            Placeholder for building mean functions; should be implemented by subclasses.
        build_covariance(output_idx, param)
            Placeholder for building covariance functions; should be implemented by subclasses.
        build_parameters_initial_guess_procedure(output_idx, **build_param)
            Placeholder for building initial guess procedures for model parameters.
        build_selection_criterion(output_idx, **build_params)
            Placeholder for building selection criteria for parameter estimation.

        """
        self.name = name
        self.output_dim = output_dim

        self.parameterized_mean = parameterized_mean
        if self.parameterized_mean:
            mean_type = "parameterized"
        else:
            mean_type = "linear_predictor"

        self.mean_functions, self.mean_functions_info = self.set_mean_functions(
            mean_params
        )
        self.covariance_functions = self.set_covariance_functions(covariance_params)

        # Initialize the models
        self.models = []
        for i in range(output_dim):
            model = gp.core.Model(
                self.mean_functions[i],
                self.covariance_functions[i],
                meanparam=None,
                covparam=None,
                meantype=mean_type,
            )
            self.models.append(
                {
                    "output_name": f"output{i}",
                    "model": model,
                    "mean_fname": self.mean_functions_info[i]["description"],
                    "mean_paramlength": self.mean_functions_info[i]["param_length"],
                    "covariance_fname": model.covariance.__name__,
                    "parameters_initial_guess_procedure": None,
                    "selection_criterion": None,
                    "info": None,
                }
            )

        # Set initial guess procedures and selection criteria after model initialization
        parameters_initial_guess_procedures = (
            self.set_parameters_initial_guess_procedures(initial_guess_procedures)
        )
        selection_criteria = self.set_selection_criteria(selection_criteria)

        # Assign initial guess procedures and selection criteria to models
        for i in range(self.output_dim):
            self.models[i]["parameters_initial_guess_procedure"] = (
                parameters_initial_guess_procedures[i]
            )
            self.models[i]["selection_criterion"] = selection_criteria[i]

    def __getitem__(self, index):
        """
        Allows accessing the individual models and their attributes using the index.

        Parameters
        ----------
        index : int
            The index of the model to access.

        Returns
        -------
        dict
            The dictionary containing the model and its associated attributes.
        """
        return self.models[index]

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        """
        Return a string representation of the Model instance.
        """
        model_info = f"Model Name: {self.name}, Output Dimension: {self.output_dim}\n"
        for i, model in enumerate(self.models):
            mean_descr = self.mean_functions_info[i]["description"]
            mean_type = model["model"].meantype
            mean_params = model["model"].meanparam
            covariance = model["model"].covariance.__name__
            cov_params = model["model"].covparam
            initial_guess = model["parameters_initial_guess_procedure"]
            selection_criterion = model["selection_criterion"]

            model_info += f"\nGaussian process {i}:\n"
            model_info += f"  Output Name: {model['output_name']}\n"
            model_info += f"  Mean: {mean_descr}\n"
            model_info += f"  Mean Type: {mean_type}\n"
            model_info += f"  Mean Parameters: {mean_params}\n"
            model_info += f"  Covariance: {covariance}\n"
            model_info += f"  Covariance Parameters: {cov_params}\n"
            model_info += f"  Initial Guess Procedure: {initial_guess.__name__ if initial_guess else 'None'}\n"
            model_info += f"  Selection Criterion: {selection_criterion.__name__ if selection_criterion else 'None'}\n"

        return model_info

    def set_mean_functions(self, mean_params):
        """Set mean functions.

        This function sets mean functions based on the parameters
        provided in mean_params.  Each entry in mean_params should
        specify a mean function and its associated parameters.

        Parameters
        ----------
        mean_params : list of dict or dict
            Parameters for defining mean functions. Can be a single
            dictionary applied to all outputs or a list of
            dictionaries, one for each output. Dictionary may include:
            - 'function': A callable mean function to be used
            - 'param_length': The length of the mean function's
              parameter vector, required if 'function' is given and is
              a callable.

        Returns
        -------
        tuple of list, list
            A tuple containing a list of mean function callables and a
            list of their descriptions.

        Raises
        ------
        ValueError
            If mean_params are not correctly specified or do not match
            output_dim.

        """
        if isinstance(mean_params, dict):
            mean_params = [mean_params] * self.output_dim
        elif not isinstance(mean_params, list) or len(mean_params) != self.output_dim:
            raise ValueError(
                "mean_params must be a dict or a list of dicts of length output_dim"
            )

        mean_functions = []
        mean_functions_info = []

        for i, param in enumerate(mean_params):
            if "function" in param and callable(param["function"]):
                mean_function = param["function"]
                if "param_length" not in param:
                    raise ValueError(
                        "'param_length' key is required for callable mean functions in mean_params"
                    )
                param_length = param["param_length"]
            else:
                mean_function, param_length = self.build_mean_function(i, param)

            mean_functions.append(mean_function)
            mean_functions_info.append(
                {"description": mean_function.__name__, "param_length": param_length}
            )

        return mean_functions, mean_functions_info

    def set_covariance_functions(self, covariance_params):
        """Set covariance functions.

        This method sets covariance functions based on the parameters
        provided in params. Each entry in params should specify a
        covariance function and its associated parameters.

        Parameters
        ----------
        covarariance_params : list of dict or dict
            Parameters for defining covariance functions. Can be a
            single dictionary applied to all outputs or a list of
            dictionaries, one for each output. Each dictionary may
            have a 'function' key providing a callable covariance
            function.

        Returns
        -------
        list
            A list of covariance function callables.

        Raises
        ------
        ValueError
            If params are not correctly specified or do not
            match output_dim.
        """
        if isinstance(covariance_params, dict):
            covariance_params = [covariance_params] * self.output_dim
        elif (
            not isinstance(covariance_params, list)
            or len(covariance_params) != self.output_dim
        ):
            raise ValueError(
                "params must be a dict or a list of dicts of length output_dim"
            )

        covariance_functions = []

        for i, param in enumerate(covariance_params):
            if "function" in param and callable(param["function"]):
                covariance_function = param["function"]
            else:
                covariance_function = self.build_covariance(i, param)

            covariance_functions.append(covariance_function)

        return covariance_functions

    def set_parameters_initial_guess_procedures(
        self, initial_guess_procedures=None, build_params=None
    ):
        """
        Set initial guess procedures based on provided initial_guess_procedures and parameters.

        Parameters
        ----------
        initial_guess_procedures : list or callable
            The initial guess procedures to be used.
        build_params : dict or list of dicts, optional
            Parameters for each initial guess procedure. Can be None.

        Returns
        -------
        list
            A list of initial guess procedure callables.

        Raises
        ------
        ValueError
            If the length of initial_guess_procedures or params does not match output_dim.
        """
        if not isinstance(build_params, list):
            build_params = [build_params] * self.output_dim

        if len(build_params) != self.output_dim:
            raise ValueError("Length of params must match output_dim")

        if initial_guess_procedures is None:
            initial_guess_procedures = [
                self.build_parameters_initial_guess_procedure(i, **(param or {}))
                for i, param in enumerate(build_params)
            ]
        elif (
            isinstance(initial_guess_procedures, list)
            and len(initial_guess_procedures) != self.output_dim
        ):
            raise ValueError(
                "initial_guess_procedures must be a list of length output_dim"
            )

        return initial_guess_procedures

    def set_selection_criteria(self, selection_criteria=None, build_params=None):
        """
        Set selection criteria based on provided selection_criteria and parameters.

        Parameters
        ----------
        selection_criteria : list or callable
            The selection criteria procedures to be used.
        build_params : dict or list of dicts, optional
            Parameters for each selection criterion. Can be None.

        Returns
        -------
        list
            A list of selection criterion callables.

        Raises
        ------
        ValueError
            If the length of selection_criteria or params does not match output_dim.
        """
        if not isinstance(build_params, list):
            build_params = [build_params] * self.output_dim

        if len(build_params) != self.output_dim:
            raise ValueError("Length of params must match output_dim")

        if selection_criteria is None:
            selection_criteria = [
                self.build_selection_criterion(i, **(param or {}))
                for i, param in enumerate(build_params)
            ]
        elif (
            isinstance(selection_criteria, list)
            and len(selection_criteria) != self.output_dim
        ):
            raise ValueError("selection_criteria must be a list of length output_dim")

        return selection_criteria

    def make_selection_criterion_with_gradient(
        self,
        model,
        xi_,
        zi_,
    ):
        selection_criterion = model["selection_criterion"]
        mean_paramlength = model["mean_paramlength"]

        if mean_paramlength > 0:
            # make a selection criterion with mean parameter
            def crit_(param):
                meanparam = param[:mean_paramlength]
                covparam = param[mean_paramlength:]
                return selection_criterion(
                    model["model"], meanparam, covparam, xi_, zi_
                )

        else:
            # make a selection criterion without mean parameter
            def crit_(covparam):
                return selection_criterion(model["model"], covparam, xi_, zi_)

        crit = gnp.DifferentiableFunction(crit_)

        return crit.evaluate, crit.gradient

    def select_params(self, xi, zi, force_param_initial_guess=True):
        """Parameter selection"""

        xi_ = gnp.asarray(xi)
        zi_ = gnp.asarray(zi)
        if zi_.ndim == 1:
            zi_ = zi_.reshape(-1, 1)

        for i in range(self.output_dim):
            tic = time.time()

            model = self.models[i]
            mpl = model["mean_paramlength"]

            if model["model"].covparam is None or force_param_initial_guess:
                initial_guess_procedure = model["parameters_initial_guess_procedure"]
                if mpl == 0:
                    meanparam0 = gnp.array([])
                    covparam0 = initial_guess_procedure(model["model"], xi_, zi_[:, i])
                else:
                    (meanparam0, covparam0) = initial_guess_procedure(
                        model["model"], xi_, zi_[:, i]
                    )
            else:
                meanparam0 = model["model"].meanparam
                covparam0 = model["model"].covparam

            param0 = gnp.concatenate((meanparam0, covparam0))

            crit, dcrit = self.make_selection_criterion_with_gradient(
                model, xi_, zi_[:, i]
            )

            param, info = gp.kernel.autoselect_parameters(
                param0, crit, dcrit, silent=True, info=True
            )

            model["model"].meanparam = gnp.asarray(param[:mpl])
            model["model"].covparam = gnp.asarray(param[mpl:])
            model["info"] = info
            model["info"]["meanparam0"] = meanparam0
            model["info"]["covparam0"] = covparam0
            model["info"]["param0"] = param0
            model["info"]["meanparam"] = model["model"].meanparam
            model["info"]["covparam"] = model["model"].covparam
            model["info"]["param"] = param
            model["info"]["selection_criterion"] = crit
            model["info"]["time"] = time.time() - tic

    def diagnosis(self, xi, zi):
        for i in range(self.output_dim):
            gp.misc.modeldiagnosis.diag(
                self.models[i]["model"], self.models[i]["info"], xi, zi
            )
            print("\n")
            gp.misc.modeldiagnosis.perf(self.models[i]["model"], xi, zi, loo=True)

    def predict(self, xi, zi, xt, convert_in=True, convert_out=True):
        """Predict method"""
        xi, zi, xt = self._ensure_shapes_and_type(
            xi=xi, zi=zi, xt=xt, convert=convert_in
        )

        zpm_ = gnp.empty((xt.shape[0], self.output_dim))
        zpv_ = gnp.empty((xt.shape[0], self.output_dim))

        for i in range(self.output_dim):
            model_predict = self.models[i]["model"].predict
            zpm_i, zpv_i = model_predict(
                xi, zi[:, i], xt, convert_in=convert_in, convert_out=False
            )
            zpm_ = gnp.set_col2(zpm_, i, zpm_i)
            zpv_ = gnp.set_col2(zpv_, i, zpv_i)

        if convert_out:
            zpm = gnp.to_np(zpm_)
            zpv = gnp.to_np(zpv_)
        else:
            zpm = zpm_
            zpv = zpv_

        return zpm, zpv

    def loo(self, xi, zi, convert_in=True, convert_out=True):
        """
        Perform leave-one-out (LOO) cross-validation for each model within the container.

        Parameters:
        xi : ndarray
            Input data points, where each row represents a data point.
        zi : ndarray
            Output values corresponding to xi. Should have a shape of (n_samples, output_dim) where
            n_samples is the number of data points and output_dim is the number of outputs.
        convert_in : bool, optional
            If True, converts input data to the appropriate type for processing (default is True).
        convert_out : bool, optional
            If True, converts output data to numpy array type before returning (default is True).

        Returns:
        results : dict
            A dictionary containing LOO results for each output, including predictions, errors, and variances.
        """
        xi, zi, xt = self._ensure_shapes_and_type(xi=xi, zi=zi, convert=convert_in)

        zloo_ = gnp.empty((xi.shape[0], self.output_dim))
        sigma2loo_ = gnp.empty((xi.shape[0], self.output_dim))
        eloo_ = gnp.empty((xi.shape[0], self.output_dim))

        for i in range(self.output_dim):
            model_loo = self.models[i]["model"].loo
            zloo_i, sigma2loo_i, eloo_i = model_loo(
                xi, zi[:, i], convert_in=convert_in, convert_out=False
            )
            zloo_ = gnp.set_col2(zloo_, i, zloo_i)
            sigma2loo_ = gnp.set_col2(sigma2loo_, i, sigma2loo_i)
            eloo_ = gnp.set_col2(eloo_, i, eloo_i)

        if convert_out:
            zloo = gnp.to_np(zloo_)
            sigma2loo = gnp.to_np(sigma2loo_)
            eloo = gnp.to_np(eloo_)
        else:
            zloo = zloo_
            sigma2loo = sigma2loo_
            eloo = eloo_

        return zloo, sigma2loo, eloo

    def compute_conditional_simulations(
        self,
        xi,
        zi,
        xt,
        n_samplepaths=1,
        type="intersection",
        method="svd",
        convert_in=True,
        convert_out=True,
    ):
        """
        Generate conditional sample paths based on input data and simulation points.

        Parameters
        ----------
        xi : ndarray(ni, d)
            Input data points used in the GP model.
        zi : ndarray(ni, output_dim)
            Observations at the input data points xi.
        xt : ndarray(nt, d)
            Points at which to simulate.
        n_samplepaths : int, optional
            Number of sample paths to generate. Default is 1.
        type : str, optional
            Specifies the relationship between xi and xt. Can be 'intersection'
            (xi and xt may have a non-empty intersection) or 'disjoint'
            (xi and xt must be disjoint). Default is 'intersection'.
        method : str, optional
            Method to draw unconditional sample paths. Can be 'svd' or 'chol'. Default is 'svd'.

        Returns
        -------
        ndarray
            An array of conditional sample paths at simulation points xt.
            The shape of the array is (nt, n_samplepaths) for a single output model,
            and (nt, n_samplepaths, output_dim) for multi-output models.
        """

        xi_, zi_, xt_ = self._ensure_shapes_and_type(
            xi=xi, zi=zi, xt=xt, convert=convert_in
        )

        compute_zsim = True  # FIXME: allows for reusing past computations
        if compute_zsim:
            # initialize xtsim and unconditional sample paths on xtsim
            ni = xi_.shape[0]
            nt = xt_.shape[0]

            xtsim = gnp.vstack((xi_, xt_))
            if type == "intersection":
                xtsim, indices = gnp.unique(xtsim, return_inverse=True, axis=0)
                xtsim_xi_ind = indices[0:ni]
                xtsim_xt_ind = indices[ni : (ni + nt)]
                n = xtsim.shape[0]
            elif type == "disjoint":
                xtsim_xi_ind = gnp.arange(ni)
                xtsim_xt_ind = gnp.arange(nt) + ni
                n = ni + nt

            # sample paths on xtsim
            zsim = gnp.empty((n, n_samplepaths, self.output_dim))

            for i in range(self.output_dim):
                zsim_i = self.models[i]["model"].sample_paths(
                    xtsim, n_samplepaths, method=method
                )
                zsim = gnp.set_col3(zsim, i, zsim_i)

        # conditional sample paths
        zpsim = gnp.empty((nt, n_samplepaths, self.output_dim))

        for i in range(self.output_dim):
            zpm, zpv, lambda_t = self.models[i]["model"].predict(
                xi_,
                zi_[:, i],
                xtsim[xtsim_xt_ind],
                return_lambdas=True,
                convert_in=False,
                convert_out=False,
            )

            if self.models[i]["model"].meantype == "linear_predictor":
                zpsim_i = self.models[i]["model"].conditional_sample_paths(
                    zsim[:, :, i],
                    xtsim_xi_ind,
                    zi_[:, i],
                    xtsim_xt_ind,
                    lambda_t,
                    convert_out=False,
                )
            elif self.models[i]["model"].meantype == "parameterized":
                zpsim_i = self.models[i][
                    "model"
                ].conditional_sample_paths_parameterized_mean(
                    zsim[:, :, i],
                    xi_,
                    xtsim_xi_ind,
                    zi_[:, i],
                    xt_,
                    xtsim_xt_ind,
                    lambda_t,
                    convert_out=False,
                )
            else:
                raise ValueError(
                    f"gpmp.core.Model.meantype {self.models[i]['model'].meantype} not implemented"
                )

            zpsim = gnp.set_col3(zpsim, i, zpsim_i)

        if self.output_dim == 1:
            # drop last dimension
            zpsim = zpsim.reshape((zpsim.shape[0], zpsim.shape[1]))

        if convert_out:
            zpsim = gnp.to_np(zpsim)

        # r = {"xtsim": xtsim, "xtsim_xi_ind": xtsim_xi_ind, "xtsim_xt_ind": xtsim_xt_ind, "zsim": zsim}
        return zpsim

    def sample_parameters(self, model_indices=None, **kwargs):
        """Run MCMC sampling for GP model parameters from posterior distribution.

        If model_indices is not provided, all models are processed.

        Parameters
        ----------
        model_indices : list of int, optional
            Indices of models to sample. Defaults to all models.
        **kwargs
            Extra arguments passed to sample_from_selection_criterion
            (see
            `gpmp.misc.param_posterior.sample_from_selection_criterion`).

        Returns
        -------
        dict
            Dictionary mapping model index to:
              - 'samples': MCMC samples (np.ndarray).
              - 'mh': MetropolisHastings instance.

        """
        from gpmp.misc.param_posterior import sample_from_selection_criterion

        if model_indices is None:
            model_indices = list(range(self.output_dim))

        results = {}
        for idx in model_indices:
            model_info = self.models[idx].get("info")
            if model_info is None:
                raise ValueError(
                    f"Model {idx} missing 'info'. Run select_params() first."
                )
            samples, mh = sample_from_selection_criterion(model_info, **kwargs)
            results[idx] = {"samples": samples, "mh": mh}
        return results

    def build_mean_function(self, output_idx: int, param: dict):
        """Build a mean function

        Parameters
        ----------
        output_idx : int
            The index of the output for which the covariance function
            is being created.
        param : dict
            Additional parameters for the mean function

        Returns
        -------
        (callable, int)
            The corresponding mean function and the number of parameters.

        """
        raise NotImplementedError("This method should be implemented by subclasses")

    def build_covariance(self, output_idx: int, param: dict):
        """Create a covariance function

        Parameters
        ----------
        output_idx : int
            The index of the output for which the covariance function
            is being created.
        param : dict
            Additional parameters for the covariance function

        Returns
        -------
        callable
            A covariance function.

        """
        raise NotImplementedError("This method should be implemented by subclasses")

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
        raise NotImplementedError("This method should be implemented by subclasses")

    def build_selection_criterion(self, output_idx: int, **build_params):
        raise NotImplementedError("This method should be implemented by subclasses")

    # --- Serialization methods for model state ---

    def _filter_info(self, info):
        """Remove non-pickleable entries from the info dict."""
        if info is None:
            return None
        info_copy = info.copy()
        # Remove keys that are functions or other non-pickleable objects.
        for key in list(info_copy.keys()):
            if callable(info_copy[key]):
                del info_copy[key]
        return info_copy

    def get_state(self):
        """Return a serializable snapshot of model parameters and info."""
        state = {"models": []}
        for m in self.models:
            state["models"].append(
                {
                    "meanparam": (
                        gnp.copy(m["model"].meanparam)
                        if m["model"].meanparam is not None
                        else None
                    ),
                    "covparam": (
                        gnp.copy(m["model"].covparam)
                        if m["model"].covparam is not None
                        else None
                    ),
                    "info": self._filter_info(m["info"]),
                }
            )
        return state

    def set_state(self, state):
        """Load model parameters and info from a previously saved state."""
        for m, m_state in zip(self.models, state.get("models", [])):
            if m_state["meanparam"] is not None:
                m["model"].meanparam = m_state["meanparam"]
            if m_state["covparam"] is not None:
                m["model"].covparam = m_state["covparam"]
            m["info"] = m_state.get("info", None)

    def save_state(self, filename):
        """Save the model state to a file."""
        from pickle import dump

        state = self.get_state()
        with open(filename, "wb") as f:
            dump(state, f)

    def load_state(self, filename):
        """Load model state from a file."""
        from pickle import load

        with open(filename, "rb") as f:
            state = load(f)
        self.set_state(state)

    def _ensure_shapes_and_type(self, xi=None, zi=None, xt=None, convert=True):
        """Validate and adjust shapes/types of input arrays.

        Parameters
        ----------
        xi : array_like, optional
            Observation points (n, dim).
        zi : array_like, optional
            Observed values (n,) or (n, output_dim). Will be converted to a 2D array.
        xt : array_like, optional
            Prediction points (m, dim).
        convert : bool, optional
            If True, convert arrays to the backend type (default is True).

        Returns
        -------
        tuple
            (xi, zi, xt) with proper shapes and types.

        Raises
        ------
        AssertionError
            If the input arrays do not meet the required shape conditions.
        """
        if xi is not None:
            assert len(xi.shape) == 2, "xi must be a 2D array"
            if convert:
                xi = gnp.asarray(xi)

        if zi is not None:
            if convert:
                zi = gnp.asarray(zi)
            # Always ensure zi is a 2D array
            if zi.ndim == 1:
                assert (
                    self.output_dim == 1
                ), "zi provided as 1D array, but output_dim is not 1"
                zi = zi.reshape(-1, 1)
            else:
                assert zi.shape[1] == self.output_dim, "zi must have output_dim columns"

        if xt is not None:
            assert len(xt.shape) == 2, "xt must be a 2D array"
            if convert:
                xt = gnp.asarray(xt)

        if xi is not None and zi is not None:
            assert (
                xi.shape[0] == zi.shape[0]
            ), "Number of rows in xi must equal number of rows in zi"
        if xi is not None and xt is not None:
            assert (
                xi.shape[1] == xt.shape[1]
            ), "xi and xt must have the same number of columns"

        return xi, zi, xt


# ==============================================================================
# Mean Functions Section
# ==============================================================================
# This section includes implementation of common mean functions in GPmp


def mean_parameterized_constant(x, param):
    return param * gnp.ones((x.shape[0], 1))


def mean_linpred_constant(x, param):
    """Constant mean function for Gaussian Process models, linear predictor type.
    Parameters
    ----------
    x : ndarray(n, d)
        Input data points in dimension d.
    param : ndarray
        Parameters of the mean function (unused in constant mean).

    Returns
    -------
    ndarray
        Array of ones with shape (n, 1).
    """
    return gnp.ones((x.shape[0], 1))


def mean_linpred_linear(x, param):
    """Linear mean function for Gaussian Process models, linear predictor type.
    Parameters
    ----------
    x : ndarray(n, d)
        Input data points in dimension d.
    param : ndarray
        Parameters of the mean function (unused in linear mean).

    Returns
    -------
    ndarray
        Matrix [1, x_[1,1], ..., x_[1, d]
                1, x_[2,1], ..., x_[2, d]
                ...
                1, x_[n,1], ..., x_[n, d]]                   ]
    """
    return gnp.hstack((gnp.ones((x.shape[0], 1)), gnp.asarray(x)))
