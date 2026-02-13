"""
Gaussian Process Model Container

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


class AttrDict(dict):
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"'AttrDict' object has no attribute '{key}'")

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, key):
        try:
            del self[key]
        except KeyError:
            raise AttributeError(f"'AttrDict' object has no attribute '{key}'")

    def __repr__(self):
        return f"AttrDict({list(self.keys())})"


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
        """
        Base class for Gaussian process models.

        This class defines GP models that can handle multiple outputs with distinct
        mean and covariance functions for each output. It supports parameter estimation
        through criteria such as Maximum Likelihood (ML) or Restricted Maximum Likelihood
        (REML) and allows for initial guess procedures for optimizing model parameters.
        It is used by `SequentialPrediction` and `SequentialStrategy`.

        Parameters
        ----------
        name : str
            The name of the model.
        output_dim : int
            The number of outputs for the model.
        parameterized_mean : bool
            If True, the model uses a "parameterized" mean function; otherwise, it uses a
            "linear predictor" mean function.
        mean_params : dict or list of dict
            Parameters for defining mean functions. Each dict should include:
            - 'function': The mean function (callable or string identifier).
            - 'param_length': The length of the mean function's parameter vector (if callable).
        covariance_params : dict or list of dict
            Parameters for defining covariance functions. Each dict must include a key 'function'
            or provide enough info to build one in `build_covariance`.
        initial_guess_procedures : list of callables or callable, optional
            Procedures for initial guess of model parameters, one for each output (or a single callable).
        selection_criteria : list of callables or callable, optional
            Selection criteria for parameter estimation, one for each output (or a single callable).
        """
        self.name = name
        self.output_dim = output_dim

        self.parameterized_mean = parameterized_mean
        mean_type = "parameterized" if self.parameterized_mean else "linear_predictor"

        self.mean_functions, self.mean_functions_info = self.set_mean_functions(mean_params)
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
                AttrDict(
                    {
                        "output_name": f"output{i}",
                        "model": model,
                        "mean_fname": self.mean_functions_info[i]["description"],
                        "mean_paramlength": self.mean_functions_info[i]["param_length"],
                        "covariance_fname": getattr(model.covariance, "__name__", type(model.covariance).__name__),
                        "parameters_initial_guess_procedure": None,
                        "selection_criterion": None,
                        "info": None,
                    }
                )
            )

        # Set initial guess procedures and selection criteria (normalize to lists)
        parameters_initial_guess_procedures = self.set_parameters_initial_guess_procedures(initial_guess_procedures)
        selection_criteria = self.set_selection_criteria(selection_criteria)

        # Assign to models
        for i in range(self.output_dim):
            self.models[i]["parameters_initial_guess_procedure"] = parameters_initial_guess_procedures[i]
            self.models[i]["selection_criterion"] = selection_criteria[i]

    def __getitem__(self, index):
        """Access an individual model and its attributes by index."""
        return self.models[index]

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        """Return a string representation of the ModelContainer instance."""
        model_info = f"Model Name: {self.name}, Output Dimension: {self.output_dim}\n"
        for i, model in enumerate(self.models):
            mean_descr = self.mean_functions_info[i]["description"]
            mean_type = model["model"].meantype
            mean_params = model["model"].meanparam
            cov_fn = model["model"].covariance
            covariance = getattr(cov_fn, "__name__", type(cov_fn).__name__)
            cov_params = model["model"].covparam
            ig = model["parameters_initial_guess_procedure"]
            sc = model["selection_criterion"]
            ig_name = getattr(ig, "__name__", type(ig).__name__) if ig else "None"
            sc_name = getattr(sc, "__name__", type(sc).__name__) if sc else "None"

            model_info += (
                f"\nGaussian process {i}:\n"
                f"  Output Name: {model['output_name']}\n"
                f"  Mean: {mean_descr}\n"
                f"  Mean Type: {mean_type}\n"
                f"  Mean Parameters: {mean_params}\n"
                f"  Covariance: {covariance}\n"
                f"  Covariance Parameters: {cov_params}\n"
                f"  Initial Guess Procedure: {ig_name}\n"
                f"  Selection Criterion: {sc_name}\n"
            )
        return model_info

    # --------------------------------------------------------------------------
    # Mean / Covariance configuration
    # --------------------------------------------------------------------------

    def set_mean_functions(self, mean_params):
        """Set mean functions.

        Each entry in mean_params specifies a mean function and its parameters.

        Parameters
        ----------
        mean_params : list of dict or dict
            Single dict (applied to all outputs) or list of dicts (one per output).
            Dict may include:
              - 'function': callable mean function
              - 'param_length': int, required when 'function' is provided and callable

        Returns
        -------
        (mean_functions, mean_functions_info)
            mean_functions: list[callable]
            mean_functions_info: list[dict] with keys 'description', 'param_length'

        Raises
        ------
        ValueError
            If mean_params are not correctly specified or do not match output_dim.
        """
        if isinstance(mean_params, dict):
            mean_params = [mean_params] * self.output_dim
        elif not isinstance(mean_params, list) or len(mean_params) != self.output_dim:
            raise ValueError("mean_params must be a dict or a list of dicts of length output_dim")

        mean_functions = []
        mean_functions_info = []

        for i, param in enumerate(mean_params):
            if "function" in param and callable(param["function"]):
                mean_function = param["function"]
                if "param_length" not in param:
                    raise ValueError("'param_length' is required for callable mean functions in mean_params")
                param_length = param["param_length"]
            else:
                mean_function, param_length = self.build_mean_function(i, param)

            mean_functions.append(mean_function)
            desc = getattr(mean_function, "__name__", type(mean_function).__name__)
            mean_functions_info.append({"description": desc, "param_length": param_length})

        return mean_functions, mean_functions_info

    def set_covariance_functions(self, covariance_params):
        """Set covariance functions.

        Parameters
        ----------
        covariance_params : list of dict or dict
            Single dict (applied to all outputs) or list of dicts (one per output).
            Each dict may have a 'function' key providing a callable covariance function.

        Returns
        -------
        list
            A list of covariance function callables.

        Raises
        ------
        ValueError
            If covariance_params are not correctly specified or do not match output_dim.
        """
        if isinstance(covariance_params, dict):
            covariance_params = [covariance_params] * self.output_dim
        elif not isinstance(covariance_params, list) or len(covariance_params) != self.output_dim:
            raise ValueError("covariance_params must be a dict or a list of dicts of length output_dim")

        covariance_functions = []
        for i, param in enumerate(covariance_params):
            if "function" in param and callable(param["function"]):
                covariance_function = param["function"]
            else:
                covariance_function = self.build_covariance(i, param)
            covariance_functions.append(covariance_function)

        return covariance_functions


    # ------------------------------------------------------------
    
    def build_mean_function(self, output_idx: int, param: dict):
        """Build a mean function for a given output.

        Parameters
        ----------
        output_idx : int
            Output index.
        param : dict
            Additional parameters for the mean function.

        Returns
        -------
        (callable, int)
            The mean function and the number of mean parameters.
        """
        raise NotImplementedError("This method should be implemented by subclasses")

    def build_covariance(self, output_idx: int, param: dict):
        """Build a covariance function for a given output.

        Parameters
        ----------
        output_idx : int
            Output index.
        param : dict
            Additional parameters for the covariance function.

        Returns
        -------
        callable
            A covariance function.
        """
        raise NotImplementedError("This method should be implemented by subclasses")

    
    # ------------------------------------------------------------
    # Parameter selection
    # ------------------------------------------------------------
    
    def set_parameters_initial_guess_procedures(self, initial_guess_procedures=None, build_params=None):
        """Set initial-guess procedures for each output.

        Parameters
        ----------
        initial_guess_procedures : list[callable] or callable or None
            List (one per output) or a single callable applied to all.
        build_params : dict or list[dict], optional
            Builder params per output if procedures are constructed here.

        Returns
        -------
        list[callable]
            One callable per output.

        Raises
        ------
        ValueError
            If lengths do not match output_dim.
        """
        if not isinstance(build_params, list):
            build_params = [build_params] * self.output_dim
        if len(build_params) != self.output_dim:
            raise ValueError("Length of params must match output_dim")

        if initial_guess_procedures is None:
            procs = [self.build_parameters_initial_guess_procedure(i, **(bp or {})) for i, bp in enumerate(build_params)]
        elif callable(initial_guess_procedures):
            procs = [initial_guess_procedures] * self.output_dim
        else:
            if len(initial_guess_procedures) != self.output_dim:
                raise ValueError("initial_guess_procedures must be a list of length output_dim")
            procs = initial_guess_procedures
        return procs

    def set_selection_criteria(self, selection_criteria=None, build_params=None):
        """Set selection criteria for each output.

        Parameters
        ----------
        selection_criteria : list[callable] or callable or None
            List (one per output) or a single callable applied to all.
        build_params : dict or list[dict], optional
            Builder params per output if criteria are constructed here.

        Returns
        -------
        list[callable]
            One callable per output.

        Raises
        ------
        ValueError
            If lengths do not match output_dim.
        """
        if not isinstance(build_params, list):
            build_params = [build_params] * self.output_dim
        if len(build_params) != self.output_dim:
            raise ValueError("Length of params must match output_dim")

        if selection_criteria is None:
            crits = [self.build_selection_criterion(i, **(bp or {})) for i, bp in enumerate(build_params)]
        elif callable(selection_criteria):
            crits = [selection_criteria] * self.output_dim
        else:
            if len(selection_criteria) != self.output_dim:
                raise ValueError("selection_criteria must be a list of length output_dim")
            crits = selection_criteria
        return crits

    def make_selection_criterion_with_gradient(self, model, xi_, zi_):
        """Wrap the model's selection criterion with gradient support."""
        selection_criterion = model["selection_criterion"]
        mean_paramlength = model["mean_paramlength"]

        if mean_paramlength > 0:
            # with mean parameters
            def crit_(param, xi, zi):
                meanparam = param[:mean_paramlength]
                covparam = param[mean_paramlength:]
                return selection_criterion(model["model"], meanparam, covparam, xi, zi)
        else:
            # covariance-only
            def crit_(covparam, xi, zi):
                return selection_criterion(model["model"], covparam, xi, zi)

        crit = gnp.DifferentiableSelectionCriterion(crit_, xi_, zi_)
        return crit.evaluate, crit.gradient, crit.evaluate_no_grad

    def build_parameters_initial_guess_procedure(self, output_idx: int, **build_param):
        """Build an initial guess procedure for anisotropic parameters.

        Parameters
        ----------
        output_idx : int
            Output index for which the procedure is built.

        Returns
        -------
        callable
            A function that returns initial guesses for parameters.
        """
        raise NotImplementedError("This method should be implemented by subclasses")

    def build_selection_criterion(self, output_idx: int, **build_params):
        """Build the selection criterion for parameter estimation."""
        raise NotImplementedError("This method should be implemented by subclasses")

    
    # --------------------------------------------------------------------------
    # Parameter selection / diagnosis / prediction
    # --------------------------------------------------------------------------

    def select_params(self, xi, zi, force_param_initial_guess=True, param0=None):
        """Parameter selection with optional initial parameter guess.

        Parameters
        ----------
        xi : array_like
            Input data points.
        zi : array_like
            Output data values.
        force_param_initial_guess : bool, optional
            If True, forces the calculation of initial guess using the procedure.
        param0 : array or list of arrays, optional
            If provided, used as the initial guess for parameters.
            - When output_dim > 1, it must be a list of arrays (one per output).
            - When output_dim == 1, it can be a single array or a list with one array.
            Each array concatenates mean and covariance parameters.
        """
        xi_ = gnp.asarray(xi)
        zi_ = gnp.asarray(zi)
        if zi_.ndim == 1:
            zi_ = zi_.reshape(-1, 1)

        if param0 is not None and self.output_dim == 1 and not isinstance(param0, list):
            param0 = [param0]

        for i in range(self.output_dim):
            tic = time.time()

            model = self.models[i]
            mpl = model["mean_paramlength"]

            if param0 is not None:
                param0_model = param0[i]
                meanparam0 = param0_model[:mpl]
                covparam0 = param0_model[mpl:]
            else:
                if model["model"].covparam is None or force_param_initial_guess:
                    initial_guess_procedure = model["parameters_initial_guess_procedure"]
                    if mpl == 0:
                        meanparam0 = gnp.array([])
                        covparam0 = initial_guess_procedure(model["model"], xi_, zi_[:, i])
                    else:
                        meanparam0, covparam0 = initial_guess_procedure(model["model"], xi_, zi_[:, i])
                else:
                    meanparam0 = model["model"].meanparam
                    covparam0 = model["model"].covparam

            param0_init = gnp.concatenate((meanparam0, covparam0))

            crit, dcrit, crit_nograd = self.make_selection_criterion_with_gradient(model, xi_, zi_[:, i])
            param, info = gp.kernel.autoselect_parameters(param0_init, crit, dcrit, silent=True, info=True)

            model["model"].meanparam = gnp.asarray(param[:mpl])
            model["model"].covparam = gnp.asarray(param[mpl:])
            model["info"] = info
            model["info"]["meanparam0"] = meanparam0
            model["info"]["covparam0"] = covparam0
            model["info"]["param0"] = param0_init
            model["info"]["meanparam"] = model["model"].meanparam
            model["info"]["covparam"] = model["model"].covparam
            model["info"]["param"] = param
            model["info"]["selection_criterion"] = crit
            model["info"]["selection_criterion_nograd"] = crit_nograd
            model["info"]["time"] = time.time() - tic


    def predict(self, xi, zi, xt, convert_in=True, convert_out=True):
        """Predict method."""
        xi, zi, xt = self._ensure_shapes_and_type(xi=xi, zi=zi, xt=xt, convert=convert_in)

        zpm_ = gnp.empty((xt.shape[0], self.output_dim))
        zpv_ = gnp.empty((xt.shape[0], self.output_dim))

        for i in range(self.output_dim):
            model_predict = self.models[i]["model"].predict
            zpm_i, zpv_i = model_predict(xi, zi[:, i], xt, convert_in=convert_in, convert_out=False)
            zpm_ = gnp.set_col_2d(zpm_, i, zpm_i)
            zpv_ = gnp.set_col_2d(zpv_, i, zpv_i)

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

        Parameters
        ----------
        xi : ndarray
            Input data points, rows = data points.
        zi : ndarray
            Output values corresponding to xi, shape (n_samples, output_dim).
        convert_in : bool, optional
            Convert input data to backend type before processing (default True).
        convert_out : bool, optional
            Convert outputs to numpy arrays before returning (default True).

        Returns
        -------
        (zloo, sigma2loo, eloo)
        """
        xi, zi, _ = self._ensure_shapes_and_type(xi=xi, zi=zi, convert=convert_in)

        zloo_ = gnp.empty((xi.shape[0], self.output_dim))
        sigma2loo_ = gnp.empty((xi.shape[0], self.output_dim))
        eloo_ = gnp.empty((xi.shape[0], self.output_dim))

        for i in range(self.output_dim):
            model_loo = self.models[i]["model"].loo
            zloo_i, sigma2loo_i, eloo_i = model_loo(xi, zi[:, i], convert_in=convert_in, convert_out=False)
            zloo_ = gnp.set_col_2d(zloo_, i, zloo_i)
            sigma2loo_ = gnp.set_col_2d(sigma2loo_, i, sigma2loo_i)
            eloo_ = gnp.set_col_2d(eloo_, i, eloo_i)

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
            Number of sample paths to generate (default 1).
        type : str, optional
            'intersection' (xi and xt may overlap) or 'disjoint' (xi and xt must be disjoint).
        method : str, optional
            Method to draw unconditional sample paths: 'svd' or 'chol' (default 'svd').

        Returns
        -------
        ndarray
            (nt, n_samplepaths) for single output, (nt, n_samplepaths, output_dim) for multi-output.
        """
        xi_, zi_, xt_ = self._ensure_shapes_and_type(xi=xi, zi=zi, xt=xt, convert=convert_in)

        compute_zsim = True  # placeholder for caching later
        if compute_zsim:
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
            else:
                raise ValueError("type must be 'intersection' or 'disjoint'")

            # sample paths on xtsim
            zsim = gnp.empty((n, n_samplepaths, self.output_dim))
            for i in range(self.output_dim):
                zsim_i = self.models[i]["model"].sample_paths(xtsim, n_samplepaths, method=method)
                zsim = gnp.set_col_3d(zsim, i, zsim_i)

        # conditional sample paths
        zpsim = gnp.empty((nt, n_samplepaths, self.output_dim))
        for i in range(self.output_dim):
            zpm, zpv, lambda_t = self.models[i]["model"].predict(
                xi_, zi_[:, i], xtsim[xtsim_xt_ind], return_lambdas=True, convert_in=False, convert_out=False
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
                zpsim_i = self.models[i]["model"].conditional_sample_paths_parameterized_mean(
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
                raise ValueError(f"gpmp.core.Model.meantype {self.models[i]['model'].meantype} not implemented")

            zpsim = gnp.set_col_3d(zpsim, i, zpsim_i)

        if self.output_dim == 1:
            zpsim = zpsim.reshape((zpsim.shape[0], zpsim.shape[1]))

        if convert_out:
            zpsim = gnp.to_np(zpsim)

        return zpsim


    # --------------------------------------------------------------------------
    # Performance diagnostics (wrappers around gp.modeldiagnosis / plotutils)
    # --------------------------------------------------------------------------

    def diagnosis(self, xi, zi):
        for i in range(self.output_dim):
            gp.modeldiagnosis.diag(self.models[i]["model"], self.models[i]["info"], xi, zi)
            print("\n")
            gp.modeldiagnosis.perf(self.models[i]["model"], xi, zi, loo=True)

    
    def compute_performance_metrics(
        self,
        xi,
        zi,
        *,
        xt=None,
        zt=None,
        loo: bool = True,
        compute_pit: bool = False,
        convert_in: bool = True,
    ):
        """
        Return dict of performance metrics per output (no plotting).

        Parameters
        ----------
        xi, zi : arrays
            Training inputs/targets. zi may be (n,) or (n, output_dim).
        xt, zt : arrays, optional
            Test inputs/targets for test-set metrics. If provided and zt is 2D,
            the i-th column is used for output i.
        loo : bool
            Compute LOO metrics.
        compute_pit : bool
            Also compute PIT vectors.
        convert_in : bool
            Convert inputs to backend type.

        Returns
        -------
        dict
            {i: perf_dict_i} keyed by output index.
        """
        xi_ = gnp.asarray(xi) if convert_in else xi
        zi_ = gnp.asarray(zi) if convert_in else zi
        if zi_.ndim == 1:
            if self.output_dim != 1:
                raise ValueError("zi is 1D but model is multi-output.")
            zi_ = zi_.reshape(-1, 1)

        # Optional test set
        xt_ = zt_ = None
        has_test = xt is not None and zt is not None
        if has_test:
            xt_ = gnp.asarray(xt) if convert_in else xt
            zt_ = gnp.asarray(zt) if convert_in else zt
            if zt_.ndim == 1 and self.output_dim != 1:
                raise ValueError("zt is 1D but model is multi-output.")
            if zt_.ndim == 1:
                zt_ = zt_.reshape(-1, 1)

        out = {}
        for i in range(self.output_dim):
            m = self.models[i]["model"]
            zi_i = zi_[:, i]
            # LOO (compute once to reuse)
            zloom = zloov = eloo = None
            if loo:
                zloom, zloov, eloo = m.loo(xi_, zi_i, convert_in=False, convert_out=False)

            # Test predictions (if any)
            zpm = zpv = None
            if has_test:
                zt_i = zt_[:, i]
                zpm, zpv = m.predict(xi_, zi_i, xt_, convert_in=False, convert_out=False)

            # Metrics
            perf_i = gp.modeldiagnosis.compute_performance(
                m,
                xi_,
                zi_i,
                loo=loo,
                loo_res=None if (zloom is None) else (zloom, zloov, eloo),
                xtzt=None if not has_test else (xt_, zt_i),
                zpmzpv=None if (zpm is None) else (zpm, zpv),
                compute_pit=compute_pit,
            )
            out[i] = perf_i
        return out

    def performance_diagnostics(
        self,
        xi,
        zi,
        *,
        xt=None,
        zt=None,
        box=None,
        cross_sections: bool = False,
        ind_i=(0, 1),
        ind_dim=None,
        plot_loo: bool = True,
        compute_pit: bool = False,
        convert_in: bool = True,
    ):
        """
        End-to-end diagnostics per output: diag() + LOO + (optional) test perf + plots.

        Mirrors:
          - gp.modeldiagnosis.diag(...)
          - model.predict / visualize_predictions (you provide zt)
          - model.loo / plot_loo
          - plotutils.crosssections(...)

        Parameters
        ----------
        xi, zi : arrays
        xt, zt : arrays, optional
            If provided, will compute test predictions and include in perf().
        box : array-like, optional
            Input domain (2,d) for cross-sections plotting.
        cross_sections : bool
            If True, call gp.plot.crosssections for each output.
        ind_i : tuple[int,int]
            Indices of two training points used by crosssections for 1D slices.
        ind_dim : list[int] or None
            Dimensions to slice in crosssections; defaults to range(d).
        plot_loo : bool
            Plot LOO diagnostics via gp.plot.plot_loo.
        compute_pit : bool
            Also compute PIT in performance metrics.
        convert_in : bool
            Convert inputs to backend type.

        Returns
        -------
        dict
            Per-output dict with keys: 'loo', 'test_pred', 'perf'.
        """
        xi_ = gnp.asarray(xi) if convert_in else xi
        zi_ = gnp.asarray(zi) if convert_in else zi
        if zi_.ndim == 1:
            if self.output_dim != 1:
                raise ValueError("zi is 1D but model is multi-output.")
            zi_ = zi_.reshape(-1, 1)

        xt_ = zt_ = None
        has_test = xt is not None and zt is not None
        if has_test:
            xt_ = gnp.asarray(xt) if convert_in else xt
            zt_ = gnp.asarray(zt) if convert_in else zt
            if zt_.ndim == 1 and self.output_dim != 1:
                raise ValueError("zt is 1D but model is multi-output.")
            if zt_.ndim == 1:
                zt_ = zt_.reshape(-1, 1)

        results = {}
        for i in range(self.output_dim):
            model = self.models[i]["model"]
            info = self.models[i]["info"]
            if info is None:
                print(f"[Warning] Output {i}: missing selection info; run select_params() first.")

            zi_i = zi_[:, i]

            # 1) diagnosis summary (param selection + data stats)
            if info is not None:
                gp.modeldiagnosis.diag(model, info, xi_, zi_i)

            # 2) predictions on test set
            zpm = zpv = None
            if has_test:
                zt_i = zt_[:, i]
                zpm, zpv = model.predict(xi_, zi_i, xt_, convert_in=False, convert_out=False)

            # 3) LOO
            zloom, zloov, eloo = model.loo(xi_, zi_i, convert_in=False, convert_out=False)
            if plot_loo:
                gp.plot.plot_loo(zi_i, zloom, zloov)

            # 4) Cross-sections of the predictive surface (optional)
            if cross_sections:
                if box is None:
                    raise ValueError("cross_sections=True requires 'box' (2,d) to be provided.")
                d = xi_.shape[1]
                indd = list(range(d)) if ind_dim is None else list(ind_dim)
                gp.plot.crosssections(model, xi_, zi_i, box, ind_i=list(ind_i), ind_dim=indd)

            # 5) Performance printout
            gp.modeldiagnosis.perf(
                model,
                xi_,
                zi_i,
                loo=True,
                loo_res=(zloom, zloov, eloo),
                xtzt=None if not has_test else (xt_, zt_i),
                zpmzpv=None if (zpm is None) else (zpm, zpv),
            )

            results[i] = {
                "loo": (zloom, zloov, eloo),
                "test_pred": None if not has_test else (zpm, zpv),
                "perf": gp.modeldiagnosis.compute_performance(
                    model,
                    xi_,
                    zi_i,
                    loo=True,
                    loo_res=(zloom, zloov, eloo),
                    xtzt=None if not has_test else (xt_, zt_i),
                    zpmzpv=None if (zpm is None) else (zpm, zpv),
                    compute_pit=compute_pit,
                ),
            }
        return results

    def plot_selection_criterion_crosssections(
        self,
        i: int,
        *,
        param_box=None,
        n_points: int = 100,
        param_names=None,
        delta: float = 5.0,
        pooled: bool = False,
        ind=None,
        ind_pooled=None,
    ):
        """
        Wrapper for gp.modeldiagnosis.plot_selection_criterion_crossections
        for output i.
        """
        info = self.models[i]["info"]
        if info is None:
            raise ValueError("Run select_params() first to populate info.")
        kw = dict(
            info=info,
            selection_criterion=None,     # use from info
            selection_criteria=None,
            covparam=None,                # use from info
            n_points=n_points,
            param_names=param_names,
            criterion_name="selection criterion",
            criterion_names=None,
            criterion_name_full="Cross sections for negative log restricted likelihood",
            ind=ind if not pooled else None,
            ind_pooled=ind_pooled if pooled else None,
            param_box=param_box if not pooled else None,
            param_box_pooled=param_box if pooled else None,
            delta=delta,
        )
        gp.modeldiagnosis.plot_selection_criterion_crossections(**kw)

    def plot_selection_criterion_profile_2d(self, i: int, param_indices=(0, 1), param_names=None, criterion_name="selection criterion"):
        """
        Wrapper for gp.modeldiagnosis.plot_selection_criterion_2d for output i.
        """
        info = self.models[i]["info"]
        if info is None:
            raise ValueError("Run select_params() first to populate info.")
        gp.modeldiagnosis.plot_selection_criterion_2d(
            self.models[i]["model"],
            info,
            param_indices=param_indices,
            param_names=param_names,
            criterion_name=criterion_name,
        )

    def selection_criterion_stats_fast(self, i: int, xi, *, ind=None, param_box=None, delta=5.0, n_points=250, verbose=False):
        """
        Wrapper for gp.modeldiagnosis.selection_criterion_statistics_fast (per output).
        """
        info = self.models[i]["info"]
        if info is None:
            raise ValueError("Run select_params() first to populate info.")
        return gp.modeldiagnosis.selection_criterion_statistics_fast(
            info=info,
            model=self.models[i]["model"],
            xi=gnp.asarray(xi),
            ind=ind,
            param_box=param_box,
            delta=delta,
            n_points=n_points,
            verbose=verbose,
        )

    def selection_criterion_stats(self, i: int, xi, *, ind=None, param_box=None, delta=5.0, verbose=False):
        """
        Wrapper for gp.modeldiagnosis.selection_criterion_statistics (per output).
        """
        info = self.models[i]["info"]
        if info is None:
            raise ValueError("Run select_params() first to populate info.")
        return gp.modeldiagnosis.selection_criterion_statistics(
            info=info,
            model=self.models[i]["model"],
            xi=gnp.asarray(xi),
            ind=ind,
            param_box=param_box,
            delta=delta,
            verbose=verbose,
        )

    
    # ------------------------------------------------------------


    
    def sample_parameters(self, model_indices=None, **kwargs):
        """Run MCMC sampling for GP model parameters from posterior distribution.

        If model_indices is not provided, all models are processed.

        Parameters
        ----------
        model_indices : list of int, optional
            Indices of models to sample. Defaults to all models.
        **kwargs
            Extra arguments passed to
            `gpmp.misc.param_posterior.sample_from_selection_criterion`.

        Returns
        -------
        dict
            {index: {'samples': ndarray, 'mh': MetropolisHastings}}
        """
        from gpmp.misc.param_posterior import sample_from_selection_criterion

        if model_indices is None:
            model_indices = list(range(self.output_dim))

        results = {}
        for idx in model_indices:
            model_info = self.models[idx].get("info")
            if model_info is None:
                raise ValueError(f"Model {idx} missing 'info'. Run select_params() first.")
            samples, mh = sample_from_selection_criterion(model_info, **kwargs)
            results[idx] = {"samples": samples, "mh": mh}
        return results

    def sample_parameters_smc(self, init_box, model_indices=None, **kwargs):
        """Run SMC sampling for GP model parameters from the posterior distribution.

        If model_indices is not provided, all models are processed.

        Parameters
        ----------
        init_box : array-like
            Initialization box for SMC sampling in parameter space.
        model_indices : list of int, optional
            Indices of models to sample. Defaults to all models.
        **kwargs
            Extra arguments passed to
            `gpmp.misc.param_posterior.sample_from_selection_criterion_smc`, e.g.:
              - n_particles: int
              - initial_temperature: float
              - final_temperature: float
              - min_ess_ratio: float
              - max_stages: int
              - debug: bool
              - plot: bool

        Returns
        -------
        dict
            {index: {'particles': ndarray, 'smc': SMC}}
        """
        from gpmp.misc.param_posterior import sample_from_selection_criterion_smc

        if model_indices is None:
            model_indices = list(range(self.output_dim))

        results = {}
        for idx in model_indices:
            model_info = self.models[idx].get("info")
            if model_info is None:
                raise ValueError(f"Model {idx} missing 'info'. Run select_params() first.")
            particles, smc_instance = sample_from_selection_criterion_smc(
                info=model_info, init_box=init_box, **kwargs
            )
            results[idx] = {"particles": particles, "smc": smc_instance}
        return results


    

    # --- Serialization methods for model state ---

    def _filter_info(self, info):
        """Remove non-pickleable entries from the info dict."""
        if info is None:
            return None
        info_copy = info.copy()
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
                    "meanparam": (gnp.copy(m["model"].meanparam) if m["model"].meanparam is not None else None),
                    "covparam": (gnp.copy(m["model"].covparam) if m["model"].covparam is not None else None),
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

    # --------------------------------------------------------------------------
    # Helpers
    # --------------------------------------------------------------------------

    def _ensure_shapes_and_type(self, xi=None, zi=None, xt=None, convert=True):
        """Validate and adjust shapes/types of input arrays.

        Parameters
        ----------
        xi : array_like, optional
            Observation points (n, d).
        zi : array_like, optional
            Observed values (n,) or (n, output_dim). Converted to 2D.
        xt : array_like, optional
            Prediction points (m, d).
        convert : bool, optional
            If True, convert arrays to the backend type (default True).

        Returns
        -------
        (xi, zi, xt)
            With proper shapes and backend types.

        Raises
        ------
        ValueError
            If input arrays do not meet required shape conditions.
        """
        if xi is not None:
            xi = gnp.asarray(xi) if convert else xi
            if getattr(xi, "ndim", None) != 2:
                raise ValueError("xi must be a 2D array")

        if zi is not None:
            zi = gnp.asarray(zi) if convert else zi
            if zi.ndim == 1:
                if self.output_dim != 1:
                    raise ValueError("zi provided as 1D array, but output_dim != 1")
                zi = zi.reshape(-1, 1)
            else:
                if zi.ndim != 2:
                    raise ValueError("zi must be a 1D or 2D array")
                if zi.shape[1] != self.output_dim:
                    raise ValueError("zi must have output_dim columns")

        if xt is not None:
            xt = gnp.asarray(xt) if convert else xt
            if getattr(xt, "ndim", None) != 2:
                raise ValueError("xt must be a 2D array")

        if xi is not None and zi is not None and xi.shape[0] != zi.shape[0]:
            raise ValueError("Number of rows in xi must equal number of rows in zi")
        if xi is not None and xt is not None and xi.shape[1] != xt.shape[1]:
            raise ValueError("xi and xt must have the same number of columns")

        return xi, zi, xt


# ==============================================================================
# Mean Functions Section (examples)
# ==============================================================================

def mean_parameterized_constant(x, param):
    return param * gnp.ones((x.shape[0], 1))


def mean_linpred_constant(x, param):
    """Constant mean function, linear predictor type."""
    return gnp.ones((x.shape[0], 1))


def mean_linpred_linear(x, param):
    """Linear mean function, linear predictor type.

    Returns matrix:
        [1, x[1,1], ..., x[1,d];
         1, x[2,1], ..., x[2,d];
         ...
         1, x[n,1], ..., x[n,d]]
    """
    return gnp.hstack((gnp.ones((x.shape[0], 1)), gnp.asarray(x)))
