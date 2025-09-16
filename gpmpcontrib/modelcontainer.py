"""
Gaussian Process Model Container

This module defines a `ModelContainer` object as a wrapper around the
`Model` object in `gpmp`, simplifying the creation and management of
Gaussian Process models. It provides tools for choosing the mean and
covariance functions, and accommodates multiple outputs. The module
supports parameter selection using user-provided methods (ML, REML...).
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
        # Return a representation that lists only the keys.
        return f"AttrDict({list(self.keys())})"


# ==============================================================================
# ModelContainer
# ==============================================================================


class ModelContainer:
    """
    Container for one or more `gpmp.core.Model` instances (multi-output GP).

    This class streamlines the creation and management of Gaussian Process
    models with possibly different mean/covariance choices per output. It also
    wires in parameter selection (e.g., ML/REML), prediction/LOO utilities,
    optional conditional simulations, light-weight diagnostics wrappers, and
    basic serialization.

    Parameters
    ----------
    name : str
        Model name (free text).
    output_dim : int
        Number of outputs (i.e., number of independent GP models managed).
    parameterized_mean : bool
        If True, the GP uses a *parameterized* mean (you provide the number of
        mean parameters via `param_length`). If False, it uses a *linear
        predictor* mean (design matrix) and the mean parameter vector is
        empty/implicit.
    mean_params : dict | list[dict]
        Mean specification(s). Either a single dict applied to all outputs, or
        a list of dicts (one per output). Each dict may be either
        - {"function": callable, "param_length": int} to use a provided mean
          function; or
        - free-form arguments consumed by your subclass implementation of
          `build_mean_function(output_idx, param)` which must return
          `(mean_callable, param_length)`.
    covariance_params : dict | list[dict]
        Covariance specification(s). Same broadcasting rules as `mean_params`.
        Each dict may be either
        - {"function": callable} to use a provided covariance function; or
        - free-form arguments consumed by your subclass implementation of
          `build_covariance(output_idx, param)` which must return a callable.
    initial_guess_procedures : callable | list[callable] | None
        Initial-guess procedure(s) for parameter optimization. Either a single
        callable applied to all outputs or a list (one per output). If None,
        they are built via `build_parameters_initial_guess_procedure(...)`.
        A procedure must accept `(model, xi, zi)` and return either
        `(meanparam0, covparam0)` (parameterized mean) or `covparam0`.
    selection_criteria : callable | list[callable] | None
        Selection criterion(ia) used during parameter estimation. Either a
        single callable applied to all outputs or a list (one per output). If
        None, they are built via `build_selection_criterion(...)`. A criterion
        must match the signature expected by `make_selection_criterion_with_gradient`.

    Attributes
    ----------
    name : str
        Model name.
    output_dim : int
        Number of outputs.
    parameterized_mean : bool
        Whether a parameterized mean is used.
    mean_functions : list[callable]
        One mean function per output.
    mean_functions_info : list[dict]
        Per-output dicts with keys `"description"` and `"param_length"`.
    covariance_functions : list[callable]
        One covariance function per output.
    models : list[AttrDict]
        For each output, an `AttrDict` with keys:
          - "output_name" : str
          - "model"       : gpmp.core.Model
          - "mean_fname"  : str
          - "mean_paramlength" : int
          - "covariance_fname" : str
          - "parameters_initial_guess_procedure" : callable | None
          - "selection_criterion"                : callable | None
          - "info" : object | None (optimizer/selection report; filled by `select_params`)

    Key Methods (overview)
    ----------------------
    Construction / config
        set_mean_functions, set_covariance_functions
        build_mean_function (override in subclass)
        build_covariance   (override in subclass)

    Parameter selection
        set_parameters_initial_guess_procedures, set_selection_criteria
        make_selection_criterion_with_gradient
        select_params

    Inference
        predict           : joint predictions over all outputs
        loo               : leave-one-out over all outputs
        compute_conditional_simulations

    Diagnostics (wrappers)
        run_diag          : per-output wrapper of `gp.misc.modeldiagnosis.diag`
        run_perf          : per-output wrapper of `gp.misc.modeldiagnosis.perf`
        plot_selection_criterion_crosssections
        plot_selection_criterion_profile_2d
        selection_criterion_stats_fast
        selection_criterion_stats

    Sampling
        sample_parameters, sample_parameters_smc

    Serialization
        get_state, set_state, save_state, load_state

    Notes
    -----
    * For multi-output problems, all arrays passed to `predict`, `loo`, and
      wrappers should have shape `(n, output_dim)` for targets (unless you pick
      a specific output via the wrapper’s `output_ind`/`model_index`).
    * Subclasses must implement `build_mean_function` and `build_covariance`
      (and usually the two `build_*` selection helpers) when you don’t pass
      explicit callables in `*_params`.
    """

    # --------------------------------------------------------------------------
    # Construction & representation
    # --------------------------------------------------------------------------
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
        """Create a multi-output GP container with per-output mean/covariance."""
        self.name = name
        self.output_dim = output_dim

        self.parameterized_mean = parameterized_mean
        mean_type = "parameterized" if self.parameterized_mean else "linear_predictor"

        self.mean_functions, self.mean_functions_info = self.set_mean_functions(
            mean_params
        )
        self.covariance_functions = self.set_covariance_functions(covariance_params)

        # Build per-output models
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
                        "covariance_fname": getattr(
                            model.covariance,
                            "__name__",
                            type(model.covariance).__name__,
                        ),
                        "parameters_initial_guess_procedure": None,
                        "selection_criterion": None,
                        "info": None,
                    }
                )
            )

        # Attach intial guess procedures / selection criteria
        parameters_initial_guess_procedures = (
            self.set_parameters_initial_guess_procedures(initial_guess_procedures)
        )
        selection_criteria = self.set_selection_criteria(selection_criteria)
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
        """Human-readable description of all sub-models."""
        s = [f"Model Name: {self.name}, Output Dimension: {self.output_dim}"]
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
            s.append(
                "\n".join(
                    [
                        f"\nGaussian process {i}:",
                        f"  Output Name: {model['output_name']}",
                        f"  Mean: {mean_descr}",
                        f"  Mean Type: {mean_type}",
                        f"  Mean Parameters: {mean_params}",
                        f"  Covariance: {covariance}",
                        f"  Covariance Parameters: {cov_params}",
                        f"  Initial Guess Procedure: {ig_name}",
                        f"  Selection Criterion: {sc_name}",
                    ]
                )
            )
        return "\n".join(s)

    # --------------------------------------------------------------------------
    # Mean / covariance configuration (builders included)
    # --------------------------------------------------------------------------
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
                        "'param_length' is required for callable mean functions"
                    )
                param_length = param["param_length"]
            else:
                mean_function, param_length = self.build_mean_function(i, param)

            mean_functions.append(mean_function)
            desc = getattr(mean_function, "__name__", type(mean_function).__name__)
            mean_functions_info.append(
                {"description": desc, "param_length": param_length}
            )
        return mean_functions, mean_functions_info

    def set_covariance_functions(self, covariance_params):
        """
        Set one covariance function per output.

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
                "covariance_params must be a dict or a list of dicts of length output_dim"
            )

        covariance_functions = []
        for i, param in enumerate(covariance_params):
            if "function" in param and callable(param["function"]):
                covariance_function = param["function"]
            else:
                covariance_function = self.build_covariance(i, param)
            covariance_functions.append(covariance_function)
        return covariance_functions

    def build_mean_function(self, output_idx: int, param: dict):
        """Build and return `(mean_callable, n_mean_params)` for the given output.

        Override in subclasses.
        
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
        raise NotImplementedError

    def build_covariance(self, output_idx: int, param: dict):
        """
        Build and return a covariance callable for the given output.

        Override in subclasses.

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
        raise NotImplementedError

    # --------------------------------------------------------------------------
    # Selection-criterion plumbing (builders + wrappers)
    # --------------------------------------------------------------------------
    def set_parameters_initial_guess_procedures(
        self, initial_guess_procedures=None, build_params=None
    ):
        """
        Attach or build initial-guess procedures (one per output).

        `initial_guess_procedures` may be a single callable or a list. If None,
        procedures are built via `build_parameters_initial_guess_procedure`.
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
        elif callable(initial_guess_procedures):
            intial_guess_procedures = [initial_guess_procedures] * self.output_dim
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
        elif callable(selection_criteria):
            selection_criteria = [selection_criteria] * self.output_dim
        elif (
            isinstance(selection_criteria, list)
            and len(selection_criteria) != self.output_dim
        ):
            raise ValueError("selection_criteria must be a list of length output_dim")

        return selection_criteria

    def make_selection_criterion_with_gradient(self, model, xi_, zi_):
        """
        Wrap the (per-output) selection criterion into a differentiable object
        compatible with `gp.kernel.autoselect_parameters`.
        """
        selection_criterion = model["selection_criterion"]
        mean_paramlength = model["mean_paramlength"]

        if mean_paramlength > 0:
            # make a selection criterion with mean parameter
            def crit_(param, xi, zi):
                meanparam = param[:mean_paramlength]
                covparam = param[mean_paramlength:]
                return selection_criterion(model["model"], meanparam, covparam, xi, zi)

        else:
            # make a selection criterion without mean parameter
            def crit_(covparam, xi, zi):
                return selection_criterion(model["model"], covparam, xi, zi)

        crit = gnp.DifferentiableSelectionCriterion(crit_, xi_, zi_)
        return crit.evaluate, crit.gradient, crit.evaluate_no_grad

    def build_parameters_initial_guess_procedure(self, output_idx: int, **build_param):
        """Build an initial guess procedure for anisotropic parameters.

        Override in subclass.

        Returns
        -------
        function
            A function to compute initial guesses for anisotropic parameters.
        """
        raise NotImplementedError

    def build_selection_criterion(self, output_idx: int, **build_params):
        """Override in subclass to construct a selection criterion."""
        raise NotImplementedError

    # --------------------------------------------------------------------------
    # Training / parameter selection
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
            - When output_dim == 1, it can be either a single array or a list with one array.
            In both cases, each array should be the concatenation of mean parameters and covariance parameters.
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
                p0 = param0[i]
                meanparam0, covparam0 = p0[:mpl], p0[mpl:]
            else:
                if model["model"].covparam is None or force_param_initial_guess:
                    initial_guess_procedure = model[
                        "parameters_initial_guess_procedure"
                    ]
                    if mpl == 0:
                        meanparam0 = gnp.array([])
                        covparam0 = initial_guess_procedure(
                            model["model"], xi_, zi_[:, i]
                        )
                    else:
                        meanparam0, covparam0 = initial_guess_procedure(
                            model["model"], xi_, zi_[:, i]
                        )
                else:
                    meanparam0 = model["model"].meanparam
                    covparam0 = model["model"].covparam

            param0_init = gnp.concatenate((meanparam0, covparam0))
            crit, dcrit, crit_nograd = self.make_selection_criterion_with_gradient(
                model, xi_, zi_[:, i]
            )

            param, info = gp.kernel.autoselect_parameters(
                param0_init, crit, dcrit, silent=True, info=True
            )

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

    # --------------------------------------------------------------------------
    # Inference / prediction APIs
    # --------------------------------------------------------------------------
    def predict(self, xi, zi, xt, convert_in=True, convert_out=True):
        """
        Predictions over all outputs.

        Returns
        -------
        (zpm, zpv) :
            Mean and variance arrays of shape (m, output_dim).
        """
        xi, zi, xt = self._ensure_shapes_and_type(
            xi=xi, zi=zi, xt=xt, convert=convert_in
        )

        zpm_ = gnp.empty((xt.shape[0], self.output_dim))
        zpv_ = gnp.empty((xt.shape[0], self.output_dim))
        for i in range(self.output_dim):
            zpm_i, zpv_i = self.models[i]["model"].predict(
                xi, zi[:, i], xt, convert_in=convert_in, convert_out=False
            )
            zpm_ = gnp.set_col_2d(zpm_, i, zpm_i)
            zpv_ = gnp.set_col_2d(zpv_, i, zpv_i)

        if convert_out:
            return gnp.to_np(zpm_), gnp.to_np(zpv_)
        return zpm_, zpv_

    def loo(self, xi, zi, convert_in=True, convert_out=True):
        """
        Perform leave-one-out (LOO) cross-validation for each model within the container.

        Parameters
        ----------
        xi : ndarray
            Input data points, where each row represents a data point.
        zi : ndarray
            Output values corresponding to xi. Should have a shape of (n_samples, output_dim) where
            n_samples is the number of data points and output_dim is the number of outputs.
        convert_in : bool, optional
            If True, converts input data to the appropriate type for processing (default is True).
        convert_out : bool, optional
            If True, converts output data to numpy array type before returning (default is True).

        Returns
        -------
        (zloo, sigma2loo, eloo) :
            Arrays of shape (n, output_dim).
        """
        xi, zi, _ = self._ensure_shapes_and_type(xi=xi, zi=zi, convert=convert_in)

        zloo_ = gnp.empty((xi.shape[0], self.output_dim))
        sigma2loo_ = gnp.empty((xi.shape[0], self.output_dim))
        eloo_ = gnp.empty((xi.shape[0], self.output_dim))
        for i in range(self.output_dim):
            zloo_i, sigma2loo_i, eloo_i = self.models[i]["model"].loo(
                xi, zi[:, i], convert_in=convert_in, convert_out=False
            )
            zloo_ = gnp.set_col_2d(zloo_, i, zloo_i)
            sigma2loo_ = gnp.set_col_2d(sigma2loo_, i, sigma2loo_i)
            eloo_ = gnp.set_col_2d(eloo_, i, eloo_i)

        if convert_out:
            return gnp.to_np(zloo_), gnp.to_np(sigma2loo_), gnp.to_np(eloo_)
        return zloo_, sigma2loo_, eloo_

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

        FIXME
        -----
            Allow an optional zsim argument.
        """

        xi_, zi_, xt_ = self._ensure_shapes_and_type(
            xi=xi, zi=zi, xt=xt, convert=convert_in
        )

        ni, nt = xi_.shape[0], xt_.shape[0]
        xtsim = gnp.vstack((xi_, xt_))
        if type == "intersection":
            xtsim, indices = gnp.unique(xtsim, return_inverse=True, axis=0)
            xtsim_xi_ind = indices[0:ni]
            xtsim_xt_ind = indices[ni : (ni + nt)]
        elif type == "disjoint":
            xtsim_xi_ind = gnp.arange(ni)
            xtsim_xt_ind = gnp.arange(nt) + ni
        else:
            raise ValueError("type must be 'intersection' or 'disjoint'")

        # unconditional sample paths
        n = xtsim.shape[0]
        zsim = gnp.empty((n, n_samplepaths, self.output_dim))
        for i in range(self.output_dim):
            zsim_i = self.models[i]["model"].sample_paths(
                xtsim, n_samplepaths, method=method
            )
            zsim = gnp.set_col_3d(zsim, i, zsim_i)

        # conditional
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
                    f"Unsupported meantype {self.models[i]['model'].meantype}"
                )

            zpsim = gnp.set_col_3d(zpsim, i, zpsim_i)

        if self.output_dim == 1:
            zpsim = zpsim.reshape((zpsim.shape[0], zpsim.shape[1]))
        if convert_out:
            zpsim = gnp.to_np(zpsim)
        return zpsim

    # --------------------------------------------------------------------------
    # Diagnostics
    # --------------------------------------------------------------------------
    def _indices_or_single(self, output_ind):
        """Yield all output indices, or a single index if provided."""
        return range(self.output_dim) if output_ind is None else [output_ind]

    def _validate_and_slice_targets(self, z, output_ind):
        """
        Ensure targets have the right shape and return a column-picker.

        Returns
        -------
        (z_, take_col_fn)
            `z_` is the validated array. `take_col_fn(arr, k)` returns the
            1D target for output k (no-op if `z_` is already 1D).
        """
        z_ = gnp.asarray(z)
        if z_.ndim == 1:
            # ok for single-output, or for multi-output when a specific output_ind is given
            if self.output_dim != 1 and output_ind is None:
                raise ValueError(
                    "zi is 1D but model is multi-output; provide output_ind or a 2D zi."
                )
            take_col_fn = lambda arr, k: arr  # no-op
        else:
            if z_.shape[1] != self.output_dim:
                raise ValueError(
                    f"zi must have {self.output_dim} columns; got {z_.shape[1]}."
                )
            take_col_fn = lambda arr, k: arr[:, k]
        return z_, take_col_fn

    def run_diag(
        self, xi, zi, *, output_ind: int | None = None, convert_in: bool = True
    ):
        """
        Wrapper around `gp.misc.modeldiagnosis.diag`.

        If `output_ind` is None, runs for all outputs; otherwise only for the
        specified output.
        """
        xi_ = gnp.asarray(xi) if convert_in else xi
        zi_, take_col_fn = self._validate_and_slice_targets(zi, output_ind)

        for k in self._indices_or_single(output_ind):
            info = self.models[k]["info"]
            if info is None:
                raise ValueError(
                    f"Output {k}: no selection info. Run select_params() first."
                )
            zi_k = zi_ if zi_.ndim == 1 else take_col_fn(zi_, k)
            gp.misc.modeldiagnosis.diag(self.models[k]["model"], info, xi_, zi_k)

    def run_perf(
        self,
        xi,
        zi,
        *,
        output_ind: int | None = None,
        loo: bool = True,
        loo_res=None,  # (zloom, zloov, eloo) — 1D or 2D; slice per k if 2D
        xtzt=None,  # (xt, zt)            — slice zt per k if 2D
        zpmzpv=None,  # (zpm, zpv)          — slice BOTH per k if 2D
        convert_in: bool = True,
    ):
        """
        Wrapper around `gp.misc.modeldiagnosis.perf`.

        If `output_ind` is None, runs for all outputs; otherwise only for the
        specified output. Supports providing precomputed tuples (`loo_res`,
        `xtzt`, `zpmzpv`) either as 1D or 2D arrays.
        """
        xi_ = gnp.asarray(xi) if convert_in else xi
        zi_, take_col_fn = self._validate_and_slice_targets(zi, output_ind)

        # Optional test tuples (convert xt/zt if provided)
        if xtzt is not None:
            xt, zt = xtzt
            xt_ = gnp.asarray(xt) if convert_in else xt
            zt_ = gnp.asarray(zt) if convert_in else zt
        else:
            xt_ = zt_ = None

        # Slicers ----------------------------------------------------
        def _slice_test(tup, k):
            """For (xt, zt): xt shared, zt sliced if 2D."""
            if tup is None:
                return None
            a, b = tup
            b_k = b if getattr(b, "ndim", 1) == 1 else b[:, k]
            return (a, b_k)

        def _slice_pred(tup, k):
            """For (zpm, zpv): slice BOTH if 2D."""
            if tup is None:
                return None
            a, b = tup
            s = lambda arr: arr if getattr(arr, "ndim", 1) == 1 else arr[:, k]
            return (s(a), s(b))

        def _slice_loo(tup, k):
            if tup is None:
                return None
            zloom, zloov, eloo = tup
            s = lambda arr: arr if getattr(arr, "ndim", 1) == 1 else arr[:, k]
            return (s(zloom), s(zloov), s(eloo))

        # ------------------------------------------------------------

        for k in self._indices_or_single(output_ind):
            zi_k = zi_ if zi_.ndim == 1 else take_col_fn(zi_, k)
            xtzt_k = _slice_test((xt_, zt_), k) if xt_ is not None else None
            zpmzpv_k = _slice_pred(zpmzpv, k) if zpmzpv is not None else None
            loo_k = _slice_loo(loo_res, k) if loo_res is not None else None

            gp.misc.modeldiagnosis.perf(
                self.models[k]["model"],
                xi_,
                zi_k,
                loo=loo,
                loo_res=loo_k,
                xtzt=xtzt_k,
                zpmzpv=zpmzpv_k,
            )

    # --------------------------------------------------------------------------
    # Selection-criterion visualization & stats (per-output wrappers)
    # --------------------------------------------------------------------------
    def plot_selection_criterion_crosssections(
        self,
        *,
        model_index: int = 0,
        param_box=None,
        n_points: int = 100,
        param_names=None,
        delta: float = 5.0,
        pooled: bool = False,
        ind=None,
        ind_pooled=None,
    ):
        """Wrapper around `gp.misc.modeldiagnosis.plot_selection_criterion_crossections`."""
        info = self.models[model_index]["info"]
        if info is None:
            raise ValueError("Run select_params() first to populate info.")
        gp.misc.modeldiagnosis.plot_selection_criterion_crossections(
            info=info,
            selection_criterion=None,
            selection_criteria=None,
            covparam=None,
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

    def plot_selection_criterion_profile_2d(
        self,
        model_index: int = 0,
        param_indices=(0, 1),
        param_names=None,
        criterion_name="selection criterion",
    ):
        """Wrapper around `gp.misc.modeldiagnosis.plot_selection_criterion_2d`."""
        info = self.models[model_index]["info"]
        if info is None:
            raise ValueError("Run select_params() first to populate info.")
        gp.misc.modeldiagnosis.plot_selection_criterion_2d(
            self.models[model_index]["model"],
            info,
            param_indices=param_indices,
            param_names=param_names,
            criterion_name=criterion_name,
        )

    def selection_criterion_stats_fast(
        self,
        xi,
        *,
        model_index: int = 0,
        ind=None,
        param_box=None,
        delta=5.0,
        n_points=250,
        verbose=False,
    ):
        """Wrapper around `gp.misc.modeldiagnosis.selection_criterion_statistics_fast`."""
        info = self.models[model_index]["info"]
        if info is None:
            raise ValueError("Run select_params() first to populate info.")
        return gp.misc.modeldiagnosis.selection_criterion_statistics_fast(
            info=info,
            model=self.models[model_index]["model"],
            xi=gnp.asarray(xi),
            ind=ind,
            param_box=param_box,
            delta=delta,
            n_points=n_points,
            verbose=verbose,
        )

    def selection_criterion_stats(
        self,
        xi,
        *,
        model_index: int = 0,
        ind=None,
        param_box=None,
        delta=5.0,
        verbose=False,
    ):
        """Wrapper around `gp.misc.modeldiagnosis.selection_criterion_statistics`."""
        info = self.models[input]["info"]
        if info is None:
            raise ValueError("Run select_params() first to populate info.")
        return gp.misc.modeldiagnosis.selection_criterion_statistics(
            info=info,
            model=self.models[model_index]["model"],
            xi=gnp.asarray(xi),
            ind=ind,
            param_box=param_box,
            delta=delta,
            verbose=verbose,
        )

    # --------------------------------------------------------------------------
    # Parameter posterior sampling
    # --------------------------------------------------------------------------
    def sample_parameters(self, model_indices=None, **kwargs):
        """
        Run MCMC over parameters using the stored selection criterion.

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

    def sample_parameters_smc(self, init_box, model_indices=None, **kwargs):
        """Run SMC sampling for GP model parameters from the posterior distribution.

        If model_indices is not provided, all models are processed.

        Parameters
        ----------
        model_indices : list of int, optional
            Indices of models to sample. Defaults to all models.
        **kwargs
            Extra arguments passed to sample_from_selection_criterion_smc.
            Expected keywords include:
              - n_particles: int, optional
              - initial_temperature: float, optional
              - final_temperature: float, optional
              - min_ess_ratio: float, optional
              - max_stages: int, optional
              - debug: bool, optional
              - plot: bool, optional
            (See the documentation of
            gpmp.misc.param_posterior.sample_from_selection_criterion_smc
            for details.)

        Returns
        -------
        dict
            Dictionary mapping model index to:
              - 'particles': Final particle positions (np.ndarray).
              - 'smc': SMC instance containing additional logs and diagnostics.
        """
        from gpmp.misc.param_posterior import sample_from_selection_criterion_smc

        if model_indices is None:
            model_indices = list(range(self.output_dim))

        results = {}
        for idx in model_indices:
            model_info = self.models[idx].get("info")
            if model_info is None:
                raise ValueError(
                    f"Model {idx} missing 'info'. Run select_params() first."
                )
            
            # NB: sample_from_selection_criterion_smc uses the negative log-posterior contained in
            # info.selection_criterion_nograd and the domain box info.box to define a tempered logpdf.
            particles, smc_instance = sample_from_selection_criterion_smc(
                info=model_info, init_box=init_box, **kwargs
            )
            results[idx] = {"particles": particles, "smc": smc_instance}
        return results

    # --------------------------------------------------------------------------
    # Serialization
    # --------------------------------------------------------------------------
    def _filter_info(self, info):
        """Drop non-pickleable entries from an `info` dict/object clone."""
        if info is None:
            return None
        info_copy = info.copy()
        # Remove keys that are functions or other non-pickleable objects.
        for key in list(info_copy.keys()):
            if callable(info_copy[key]):
                del info_copy[key]
        return info_copy

    def get_state(self):
        """Return a serializable snapshot of parameters and (filtered) info."""
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
        """Restore parameters and info from `get_state()` output."""
        for m, m_state in zip(self.models, state.get("models", [])):
            if m_state["meanparam"] is not None:
                m["model"].meanparam = m_state["meanparam"]
            if m_state["covparam"] is not None:
                m["model"].covparam = m_state["covparam"]
            m["info"] = m_state.get("info", None)

    def save_state(self, filename):
        """Serialize model state to disk (pickle)."""
        from pickle import dump

        state = self.get_state()
        with open(filename, "wb") as f:
            dump(state, f)

    def load_state(self, filename):
        """Load model state from disk (pickle)."""
        from pickle import load

        with open(filename, "rb") as f:
            state = load(f)
        self.set_state(state)

    # --------------------------------------------------------------------------
    # Helpers
    # --------------------------------------------------------------------------
    def _ensure_shapes_and_type(self, xi=None, zi=None, xt=None, convert=True):
        """
        Validate shapes and (optionally) convert arrays to backend type.

        Returns
        -------
        (xi, zi, xt) : arrays with correct shapes/types.
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
# Mean functions
# ==============================================================================


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
