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
from gpmp.misc.param import Param


class AttrDict(dict):
    """Dict with attribute access to keys (d.k <-> d['k']).

    Missing keys raise AttributeError.
    """    
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
    mean_specification : dict | list[dict]
        Mean specification(s). Either a single dict applied to all outputs, or
        a list of dicts (one per output). Each dict may be either
        - {"function": callable, "param_length": int} to use a provided mean
          function; or
        - free-form arguments consumed by your subclass implementation of
          `build_mean_function(output_idx, param)` which must return
          `(mean_callable, param_length)`.
    covariance_specification : dict | list[dict]
        Covariance specification(s). Same broadcasting rules as `mean_specification`.
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
        run_diag          : per-output wrapper of `gp.modeldiagnosis.diag`
        run_perf          : per-output wrapper of `gp.modeldiagnosis.perf`
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
        mean_specification,
        covariance_specification,
        initial_guess_procedures=None,
        selection_criteria=None,
    ):
        """Create a multi-output GP container with per-output mean/covariance."""
        self.name = name
        self.output_dim = output_dim

        self.parameterized_mean = parameterized_mean
        mean_type = "parameterized" if self.parameterized_mean else "linear_predictor"

        # mean_functions : list[callable] -> one mean function per
        # mean_functions_info : list[dict] -> per-output dicts
        #   with keys `"description"` and `"param_length"`
        # covariance_functions : list[callable] -> one covariance function per output.
        mean_functions, mean_functions_info = self.make_mean_functions(
            mean_specification
        )
        covariance_functions = self.make_covariance_functions(covariance_specification)

        # Build per-output models
        self.models = []
        for i in range(output_dim):
            model = gp.core.Model(
                mean_functions[i],
                covariance_functions[i],
                meanparam=None,
                covparam=None,
                meantype=mean_type,
            )
            self.models.append(
                AttrDict(
                    {
                        "output_name": f"output{i}",
                        "model": model,
                        "mean_fname": mean_functions_info[i]["description"],
                        "mean_paramlength": mean_functions_info[i]["param_length"],
                        "covariance_fname": getattr(
                            model.covariance,
                            "__name__",
                            type(model.covariance).__name__,
                        ),
                        "param": None,  # a Param instance from gpmp.misc.param
                        "parameters_initial_guess_procedure": None,
                        "selection_criterion": None,
                        "info": None,
                    }
                )
            )

        # Install guess parameters procedures / selection criteria
        parameters_initial_guess_procedures = (
            self.make_parameters_initial_guess_procedures(initial_guess_procedures)
        )
        selection_criteria = self.make_selection_criteria(selection_criteria)

        for i in range(self.output_dim):
            self.models[i]["parameters_initial_guess_procedure"] = (
                parameters_initial_guess_procedures[i]
            )
            self.models[i]["selection_criterion"] = selection_criteria[i]

        # Install get_param / apply_param on each models[i]
        param_procedures = self.make_param_object_procedures(
            param_procedures=None,
            auxiliary_params=[
                {"name_prefix": f"z{i}_"} for i in range(self.output_dim)
            ],
        )
        for i, (pf, pt) in enumerate(param_procedures):
            model_entry = self.models[i]
            model_entry["param_from_vectors"] = pf
            model_entry["param_to_vectors"] = pt
            model_entry["param"] = None

            def _get_param(pf=pf, model_entry=model_entry):
                m = model_entry.model
                P = pf(m.meanparam, m.covparam)
                model_entry["param"] = P
                return P

            def _apply_param(P, pt=pt, model_entry=model_entry):
                m = model_entry.model
                mean_vec, cov_vec = pt(P)
                m.meanparam = mean_vec
                m.covparam = cov_vec
                model_entry["param"] = P

            # expose as methods on the per-output record
            model_entry["get_param"] = _get_param
            model_entry["apply_param"] = _apply_param

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
            # mean_descr = self.mean_functions_info[i]["description"]  #FIXME
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
                        # f"  Mean: {mean_descr}",
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
    def make_mean_functions(self, mean_specification):
        """Make mean functions.

        This function sets mean functions based on the parameters
        provided in mean_specification.  Each entry in mean_specification should
        specify a mean function and its associated parameters.

        Parameters
        ----------
        mean_specification : list of dict or dict
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
            If mean_specification are not correctly specified or do not match
            output_dim.
        """
        if isinstance(mean_specification, dict):
            mean_specification = [mean_specification] * self.output_dim
        elif (
            not isinstance(mean_specification, list)
            or len(mean_specification) != self.output_dim
        ):
            raise ValueError(
                "mean_specification must be a dict or a list of dicts of length output_dim"
            )

        mean_functions = []
        mean_functions_info = []
        for i, param in enumerate(mean_specification):
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

    def make_covariance_functions(self, covariance_specification):
        """
        Make one covariance function per output.

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
        if isinstance(covariance_specification, dict):
            covariance_specification = [covariance_specification] * self.output_dim
        elif (
            not isinstance(covariance_specification, list)
            or len(covariance_specification) != self.output_dim
        ):
            raise ValueError(
                "covariance_specification must be a dict or a list of dicts of length output_dim"
            )

        covariance_functions = []
        for i, param in enumerate(covariance_specification):
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
    # Param / Selection-criterion plumbing (builders + wrappers)
    # --------------------------------------------------------------------------
    def make_param_object_procedures(self, param_procedures=None, auxiliary_params=None):
        """
        Return per-output Param adapter procedures.

        This procedure normalizes user input (None, a single (pf, pt) pair, or a
        per-output list of pairs) into a validated list of (pf, pt) pairs, so the
        rest of the code can index procedures[i] unconditionally.

        If param_procedures is None:
            build via build_param_procedures(i, **bp) per output.
        If a single pair (pf, pt) is passed:
            broadcast to all outputs.
        If a list of pairs is passed:
            length must equal output_dim.

        pf, pt meanings
        ---------------
        pf ("param_from_vectors"):
            Callable that packs model parameter vectors into a Param object:
                pf(meanparam, covparam) -> Param
        pt ("param_to_vectors"):
            Callable that unpacks a Param object into model parameter vectors:
                pt(Param) -> (meanparam, covparam)
        """
        # broadcast build_params
        if not isinstance(auxiliary_params, list):
            auxiliary_params = [auxiliary_params] * self.output_dim
        if len(auxiliary_params) != self.output_dim:
            raise ValueError("Length of auxiliary_params must match output_dim")

        # Case 1: build
        if param_procedures is None:
            return [
                self.build_param_procedures(i, **(bp or {}))
                for i, bp in enumerate(auxiliary_params)
            ]
        # Case 2: single pair - broadcast
        if (
            isinstance(param_procedures, tuple)
            and len(param_procedures) == 2
            and callable(param_procedures[0])
            and callable(param_procedures[1])
        ):
            return [param_procedures] * self.output_dim
        # Case 3: list of pairs -> validate
        if isinstance(param_procedures, list):
            if len(param_procedures) != self.output_dim:
                raise ValueError("param_procedures must have length output_dim")
            for j, pair in enumerate(param_procedures):
                if (
                    not isinstance(pair, tuple)
                    or len(pair) != 2
                    or not callable(pair[0])
                    or not callable(pair[1])
                ):
                    raise TypeError(
                        f"param_procedures[{j}] must be a (callable, callable) tuple"
                    )
            return param_procedures

        raise TypeError(
            "param_procedures must be None, a (callable, callable) pair, or a list of such pairs"
        )

    def make_parameters_initial_guess_procedures(
        self, initial_guess_procedures=None, auxiliary_params=None
    ):
        """
        Return initial-guess procedures (one per output).

        This procedure normalizes user input (None, a single callable, or a
        per-output list) into a validated list of per-output procedures, so the
        rest of the code can index procedures[i] unconditionally.

        initial_guess_procedures may be:
          - None: build per output via build_parameters_initial_guess_procedure(...)
          - callable: broadcast to all outputs
          - list[callable]: one per output (length must be output_dim)

        auxiliary_params may be:
          - None: each builder gets {}
          - dict: broadcast to all outputs
          - list[dict|None]: one per output (length must be output_dim)
        """
        # Normalize auxiliary_params -> list of dicts (or {})
        if auxiliary_params is None:
            build_params = [{} for _ in range(self.output_dim)]
        elif isinstance(auxiliary_params, dict):
            build_params = [auxiliary_params] * self.output_dim
        elif isinstance(auxiliary_params, list):
            if len(auxiliary_params) != self.output_dim:
                raise ValueError("Length of auxiliary_params must match output_dim")
            build_params = [(p or {}) for p in auxiliary_params]
        else:
            raise TypeError("auxiliary_params must be None, a dict, or a list of dicts")

        # Normalize initial_guess_procedures -> list of callables
        if initial_guess_procedures is None:
            procedures = [
                self.build_parameters_initial_guess_procedure(i, **build_params[i])
                for i in range(self.output_dim)
            ]
        elif callable(initial_guess_procedures):
            procedures = [initial_guess_procedures] * self.output_dim
        elif isinstance(initial_guess_procedures, list):
            if len(initial_guess_procedures) != self.output_dim:
                raise ValueError("initial_guess_procedures must have length output_dim")
            if not all(callable(p) for p in initial_guess_procedures):
                raise TypeError("initial_guess_procedures must contain only callables")
            procedures = initial_guess_procedures
        else:
            raise TypeError(
                "initial_guess_procedures must be None, a callable, or a list of callables"
            )

        return procedures

    def make_selection_criteria(self, selection_criteria=None, auxiliary_params=None):
        """
        Return selection criteria (one per output).

        This procedure normalizes user input (None, a single callable, or a
        per-output list) into a validated list of per-output criteria, so the rest
        of the code can index criteria[i] unconditionally.

        Parameters
        ----------
        selection_criteria : callable | list[callable] | None
            Criterion procedure(s). If None, criteria are built via
            build_selection_criterion(...). If a callable is provided, it is
            broadcast to all outputs. If a list is provided, its length must match
            output_dim.
        auxiliary_params : dict | list[dict] | None
            Per-output parameters passed to build_selection_criterion when
            selection_criteria is None. If a dict is provided, it is broadcast to
            all outputs. If a list is provided, its length must match output_dim.

        Returns
        -------
        list[callable]
            A list of selection criterion callables, one per output.
        """
        # Normalize auxiliary_params -> list of dicts
        if auxiliary_params is None:
            build_params = [{} for _ in range(self.output_dim)]
        elif isinstance(auxiliary_params, dict):
            build_params = [auxiliary_params] * self.output_dim
        elif isinstance(auxiliary_params, list):
            if len(auxiliary_params) != self.output_dim:
                raise ValueError("Length of auxiliary_params must match output_dim")
            build_params = [(p or {}) for p in auxiliary_params]
        else:
            raise TypeError("auxiliary_params must be None, a dict, or a list of dicts")

        # Normalize selection_criteria -> list of callables
        if selection_criteria is None:
            criteria = [
                self.build_selection_criterion(i, **build_params[i])
                for i in range(self.output_dim)
            ]
        elif callable(selection_criteria):
            criteria = [selection_criteria] * self.output_dim
        elif isinstance(selection_criteria, list):
            if len(selection_criteria) != self.output_dim:
                raise ValueError("selection_criteria must have length output_dim")
            if not all(callable(c) for c in selection_criteria):
                raise TypeError("selection_criteria must contain only callables")
            criteria = selection_criteria
        else:
            raise TypeError(
                "selection_criteria must be None, a callable, or a list of callables"
            )

        return criteria

    # def make_selection_criterion_with_gradient(
    #         self,
    #         model,
    #         xi_=None,
    #         zi_=None,
    #         dataloader=None,
    #         batches_per_eval=0,
    #         parameterized_mean=False
    # ):
    #     """
    #     Wrap the (per-output) selection criterion into a differentiable object
    #     compatible with `gp.kernel.autoselect_parameters`.
    #     """
    #     selection_criterion = model["selection_criterion"]
    #     meanparam_len = model["mean_paramlength"]
    #     data_source = gp.kernel.parameter_selection.check_xi_zi_or_loader(xi_, zi_, dataloader)
        
    #     if meanparam_len > 0:
    #         parameterized_mean = True

    #     if parameterized_mean:
    #         # make a selection criterion with mean parameter
    #         def crit_(param, xi, zi):
    #             meanparam = param[:meanparam_len]
    #             covparam = param[meanparam_len:]
    #             return selection_criterion(model["model"], meanparam, covparam, xi, zi)
    #     else:
    #         # make a selection criterion without mean parameter
    #         def crit_(covparam, xi, zi):
    #             return selection_criterion(model["model"], covparam, xi, zi)

    #     if data_source == "arrays":
    #         crit = gnp.DifferentiableSelectionCriterion(crit_, xi_, zi_)
    #     else:
    #         crit = gnp.BatchDifferentiableSelectionCriterion(
    #             crit_, dataloader, batches_per_eval=batches_per_eval
    #         )
    
    #     return crit.evaluate, crit.gradient, crit.evaluate_no_grad

    def build_param_procedures(self, output_idx: int, **kwargs):
        """Return Param procedures"""
        raise NotImplementedError

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
    # Parameter selection
    # --------------------------------------------------------------------------
    def select_params(
        self,
        xi,
        zi,
        *,
        force_param_initial_guess=True,
        param0=None,
        use_bounds_from_param_obj=True,
        bounds=None,
        bounds_factory=None,
        bounds_delta=None,
    ):
        """
        Parameter selection (per output).

        Parameters
        ----------
        xi : array_like
            Observation points, shape (n, d).
        zi : array_like
            Observed values. Shape (n,) for single-output, or (n, output_dim).
        force_param_initial_guess : bool, default True
            If True, compute an initial guess even when the model already
            has parameters. If False and the model already has parameters,
            those are reused as the starting point.
        param0 : None | array_like | Param | list[ array_like | Param ]
            Optional initializer(s) for optimization. For multi-output:
              - provide a list where param0[i] is either a vector in the
                model's normalized coordinates or a Param for output i.
            For single-output, you may pass a single vector or a Param.
            If a vector is passed, it must be the concatenation of mean
            parameters (length = mean_paramlength) followed by covariance
            parameters. If a Param is passed, the method uses the per-output
            adapter param_to_vectors to extract (mean, cov).
        use_bounds_from_param_obj : bool, default True
            If True and a per-output Param object is available, bounds are
            taken from Param.bounds (normalized coordinates) unless overridden
            by manual bounds or a bounds_factory.
        bounds : None | array_like(n_params, 2) | list[array_like(n_params, 2)]
            Manual bounds in normalized space. If provided for multi-output,
            either supply a single (n_params, 2) array to broadcast to all
            outputs, or a list with one (n_params, 2) array per output.
            These take precedence over bounds from Param or bounds_factory.
        bounds_factory : None | callable
            Optional callback to compute bounds from data and the initializer:
                bounds_factory(model_dict, xi_array, zi_vector, param0_init_vector)
            It must return an array of shape (n_params, 2) or None.
            Used only if bounds is None; takes precedence over Param bounds.
        bounds_delta : None | float | array_like
            Local interval around the initializer param0_init in normalized space.
            If provided, for each parameter theta0 we build [theta0 - delta, theta0 + delta].
            If scalar, the same delta is used for all parameters; if an array, the
            length must match the parameter dimension. This interval is then
            intersected with whichever base bounds were chosen by precedence:
                manual bounds > bounds_factory(...) > Param bounds > None
        """
        xi_ = gnp.asarray(xi)
        zi_ = gnp.asarray(zi)
        if zi_.ndim == 1:
            zi_ = zi_.reshape(-1, 1)

        # Normalize param0 container for multi-output
        if param0 is not None and self.output_dim == 1 and not isinstance(param0, list):
            param0 = [param0]

        for i in range(self.output_dim):
            tic = time.time()
            model = self.models[i]
            mpl = int(model["mean_paramlength"])

            # 1) Initializer (meanparam0, covparam0)
            if param0 is not None:
                p0_i = param0[i]
                meanparam0, covparam0 = self._param_to_vector_pair(p0_i, model, mpl)
            else:
                if model["model"].covparam is None or force_param_initial_guess:
                    igp = model["parameters_initial_guess_procedure"]
                    if mpl == 0:
                        meanparam0 = gnp.array([])
                        covparam0 = igp(model["model"], xi_, zi_[:, i])
                    else:
                        meanparam0, covparam0 = igp(model["model"], xi_, zi_[:, i])
                else:
                    meanparam0 = model["model"].meanparam
                    covparam0 = model["model"].covparam

            param0_init = gnp.concatenate((meanparam0, covparam0))
            n_params = int(param0_init.shape[0])

            # 2) Base bounds by precedence: manual > factory > Param > None
            base_bounds = ModelContainer._extract_bounds_for_output(
                bounds, i, n_params, output_dim=self.output_dim
            )

            if base_bounds is None and callable(bounds_factory):
                base_bounds = bounds_factory(model, xi_, zi_[:, i], param0_init)
                if base_bounds is not None:
                    base_bounds = ModelContainer._as_bounds_array(base_bounds, n_params)

            if base_bounds is None and use_bounds_from_param_obj:
                # Use existing Param if present; otherwise build a temporary one from the initializer.
                P = model.get("param", None)
                if P is None and callable(model.get("param_from_vectors", None)):
                    P = model["param_from_vectors"](meanparam0, covparam0)
                if P is not None:
                    base_bounds = ModelContainer._bounds_from_param_obj(P)

            # 3) Local +-delta interval around param0_init (intersect)
            if bounds_delta is not None:
                local_interval = ModelContainer._interval_around(param0_init, bounds_delta)
                base_bounds = ModelContainer._intersect_bounds(base_bounds, local_interval)

            # 4) Optimize
            selection_criterion = model["selection_criterion"]
            meanparam_len = model["mean_paramlength"]
            crit, crit_pre_grad, crit_no_grad, crit_grad = (
                gp.kernel.make_selection_criterion_with_gradient(
                    model["model"],
                    selection_criterion,
                    xi=xi_,
                    zi=zi_[:, i],
                    dataloader=None,
                    parameterized_mean=True if meanparam_len > 0 else False,
                    meanparam_len=meanparam_len,
                )
            )

            # crit, dcrit, crit_nograd = self.make_selection_criterion_with_gradient(
            #     model, xi_, zi_[:, i]
            # )

            param, info = gp.kernel.autoselect_parameters(
                param0_init,
                crit_pre_grad,
                crit_grad,
                bounds=base_bounds,
                silent=True,
                info=True,
            )

            # 5) Write back params
            model["model"].meanparam = gnp.asarray(param[:mpl])
            model["model"].covparam = gnp.asarray(param[mpl:])

            # 6) Refresh Param object (pf must be pf(meanparam, covparam))
            if callable(model.get("get_param", None)):
                model["param"] = model["get_param"]()
            elif callable(model.get("param_from_vectors", None)):
                model["param"] = model["param_from_vectors"](
                    model["model"].meanparam, model["model"].covparam
                )

            # Mirror final bounds into Param.bounds (normalized space)
            ModelContainer._apply_bounds_to_param(model.get("param", None), base_bounds, mpl)

            # 7) Record info
            model["info"] = info
            model["info"]["meanparam0"] = meanparam0
            model["info"]["covparam0"] = covparam0
            model["info"]["param0"] = param0_init
            model["info"]["meanparam"] = model["model"].meanparam
            model["info"]["covparam"] = model["model"].covparam
            model["info"]["param"] = param
            model["info"]["selection_criterion"] = crit
            model["info"]["selection_criterion_nograd"] = crit_no_grad
            model["info"]["time"] = time.time() - tic

    
    # def select_params(
    #     self,
    #     xi,
    #     zi,
    #     *,
    #     force_param_initial_guess=True,
    #     param0=None,
    #     use_bounds_from_param_obj=True,
    #     bounds=None,
    #     bounds_factory=None,
    #     bounds_delta=None,
    # ):
    #     """
    #     Parameter selection (per output).

    #     Parameters
    #     ----------
    #     xi : array_like
    #         Observation points, shape (n, d).
    #     zi : array_like
    #         Observed values. Shape (n,) for single-output, or (n, output_dim).
    #     force_param_initial_guess : bool, default True
    #         If True, compute an initial guess even when the model already
    #         has parameters. If False and the model already has parameters,
    #         those are reused as the starting point.
    #     param0 : None | array_like | Param | list[ array_like | Param ]
    #         Optional initializer(s) for optimization. For multi-output:
    #           - provide a list where param0[i] is either a vector in the
    #             model's normalized coordinates or a Param for output i.
    #         For single-output, you may pass a single vector or a Param.
    #         If a vector is passed, it must be the concatenation of mean
    #         parameters (length = mean_paramlength) followed by covariance
    #         parameters. If a Param is passed, the method will use the
    #         model's vectors_from_param adapter to extract (mean, cov).
    #     use_bounds_from_param_obj : bool, default True
    #         If True and the per-output model has a non-None Param object,
    #         bounds are taken from that Param (in normalized coordinates)
    #         unless overridden by manual bounds or a bounds_factory.
    #     bounds : None | array_like(n_params, 2) | list[array_like(n_params, 2)]
    #         Manual bounds in normalized space. If provided for multi-output,
    #         either supply a single (n_params, 2) array to broadcast to all
    #         outputs, or a list with one (n_params, 2) array per output.
    #         These take precedence over bounds from Param or bounds_factory.
    #     bounds_factory : None | callable
    #         Optional callback to compute bounds from data and the initializer:
    #             bounds_factory(model_dict, xi_array, zi_vector, param0_init_vector)
    #         It must return an array of shape (n_params, 2) or None.
    #         Used only if `bounds` is None; takes precedence over Param bounds.
    #     bounds_delta : None | float | array_like
    #         Local interval around the initializer param0_init in normalized space.
    #         If provided, for each parameter theta0 we build [theta0 - delta, theta0 + delta].
    #         If scalar, the same delta is used for all parameters; if an array, the
    #         length must match the parameter dimension. This interval is then
    #         intersected with whichever base bounds were chosen by precedence:
    #             manual bounds > bounds_factory(...) > Param bounds > None
    #     """
    #     xi_ = gnp.asarray(xi)
    #     zi_ = gnp.asarray(zi)
    #     if zi_.ndim == 1:
    #         zi_ = zi_.reshape(-1, 1)

    #     # Normalize param0 container for multi-output
    #     if param0 is not None and self.output_dim == 1 and not isinstance(param0, list):
    #         param0 = [param0]

    #     for i in range(self.output_dim):
    #         tic = time.time()
    #         model = self.models[i]
    #         mpl = int(model["mean_paramlength"])

    #         # 1) Initializer (meanparam0, covparam0)
    #         if param0 is not None:
    #             p0_i = param0[i]
    #             meanparam0, covparam0 = self._param_to_vector_pair(p0_i, model, mpl)
    #         else:
    #             if model["model"].covparam is None or force_param_initial_guess:
    #                 igp = model["parameters_initial_guess_procedure"]
    #                 if mpl == 0:
    #                     meanparam0 = gnp.array([])
    #                     covparam0 = igp(model["model"], xi_, zi_[:, i])
    #                 else:
    #                     meanparam0, covparam0 = igp(model["model"], xi_, zi_[:, i])
    #             else:
    #                 meanparam0 = model["model"].meanparam
    #                 covparam0 = model["model"].covparam

    #         param0_init = gnp.concatenate((meanparam0, covparam0))
    #         n_params = param0_init.shape[0]

    #         # 2) Base bounds by precedence: manual > factory > Param > None
    #         base_bounds = ModelContainer._extract_bounds_for_output(bounds, i, n_params)
    #         if base_bounds is None and callable(bounds_factory):
    #             base_bounds = bounds_factory(model, xi_, zi_[:, i], param0_init)
    #             if base_bounds is not None:
    #                 base_bounds = ModelContainer._as_bounds_array(base_bounds, n_params)
    #         if (
    #             base_bounds is None
    #             and use_bounds_from_param_obj
    #             and model.get("param") is not None
    #         ):
    #             base_bounds = ModelContainer._bounds_from_param_obj(model["param"])

    #         # 3) Local +-delta interval around param0_init (intersect)
    #         if bounds_delta is not None:
    #             local_interval = ModelContainer._interval_around(
    #                 param0_init, bounds_delta
    #             )
    #             base_bounds = ModelContainer._intersect_bounds(
    #                 base_bounds, local_interval
    #             )

    #         # 4) Optimize
    #         crit, dcrit, crit_nograd = self.make_selection_criterion_with_gradient(
    #             model, xi_, zi_[:, i]
    #         )
    #         param, info = gp.kernel.autoselect_parameters(
    #             param0_init,
    #             crit,
    #             dcrit,
    #             bounds=base_bounds,
    #             silent=True,
    #             info=True,
    #         )

    #         # 5) Write back params
    #         model["model"].meanparam = gnp.asarray(param[:mpl])
    #         model["model"].covparam = gnp.asarray(param[mpl:])

    #         # 6) Refresh Param object and copy bounds used into it
    #         if callable(model.get("get_param", None)):
    #             model["param"] = model["get_param"]()
    #         elif callable(model.get("param_from_vectors", None)):
    #             model["param"] = model["param_from_vectors"](
    #                 xi_, zi_[:, i], model["model"].meanparam, model["model"].covparam
    #             )

    #         # If we have a Param, mirror the final bounds into Param.bounds (normalized space)
    #         ModelContainer._apply_bounds_to_param(
    #             model.get("param", None), base_bounds, mpl
    #         )

    #         # 7) Record info
    #         model["info"] = info
    #         model["info"]["meanparam0"] = meanparam0
    #         model["info"]["covparam0"] = covparam0
    #         model["info"]["param0"] = param0_init
    #         model["info"]["meanparam"] = model["model"].meanparam
    #         model["info"]["covparam"] = model["model"].covparam
    #         model["info"]["param"] = param
    #         model["info"]["selection_criterion"] = crit
    #         model["info"]["selection_criterion_nograd"] = crit_nograd
    #         model["info"]["time"] = time.time() - tic

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
            zpm_[:, i] = zpm_i
            zpv_[:, i] = zpv_i

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
            zloo_[:, i] = zloo_i
            sigma2loo_[:, i] = sigma2loo_i
            eloo_[:, i] = eloo_i

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
            zsim[:, :, i] = zsim_i

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

            zpsim[:, :, i] = zpsim_i

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
        Wrapper around `gp.modeldiagnosis.diag`.

        If `output_ind` is None, runs for all outputs; otherwise only for the
        specified output.
        """
        xi_ = gnp.asarray(xi) if convert_in else xi
        zi_, take_col_fn = self._validate_and_slice_targets(zi, output_ind)

        for k in self._indices_or_single(output_ind):
            print(f"~ Model [{k}]")
            info = self.models[k]["info"]
            if info is None:
                raise ValueError(
                    f"Output {k}: no selection info. Run select_params() first."
                )
            zi_k = zi_ if zi_.ndim == 1 else take_col_fn(zi_, k)
            gp.modeldiagnosis.diag(self.models[k]["model"], info, xi_, zi_k)

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
        Wrapper around `gp.modeldiagnosis.perf`.

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
            print(f"~ Model [{k}]")
            zi_k = zi_ if zi_.ndim == 1 else take_col_fn(zi_, k)
            xtzt_k = _slice_test((xt_, zt_), k) if xt_ is not None else None
            zpmzpv_k = _slice_pred(zpmzpv, k) if zpmzpv is not None else None
            loo_k = _slice_loo(loo_res, k) if loo_res is not None else None

            gp.modeldiagnosis.perf(
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
        param_box_pooled=None,
        n_points: int = 100,
        param_names=None,
        delta: float = 5.0,
        pooled: bool = False,
        ind=None,
        ind_pooled=None,
    ):
        """Wrapper around `gp.modeldiagnosis.plot_selection_criterion_crosssections`."""
        info = self.models[model_index]["info"]
        if info is None:
            raise ValueError("Run select_params() first to populate info.")
        gp.modeldiagnosis.plot_selection_criterion_crosssections(
            info=info,
            selection_criterion=None,
            selection_criteria=None,
            covparam=None,
            n_points=n_points,
            param_names=param_names,
            criterion_name="selection criterion",
            criterion_names=None,
            criterion_name_full="Cross sections for negative log restricted likelihood",
            ind=ind,
            ind_pooled=ind_pooled if pooled else None,
            param_box=param_box if not pooled else None,
            param_box_pooled=param_box_pooled if pooled else None,
            delta=delta,
        )

    def plot_selection_criterion_profile_2d(
        self,
        model_index: int = 0,
        param_indices=(0, 1),
        param_names=None,
        criterion_name="selection criterion",
    ):
        """Wrapper around `gp.modeldiagnosis.plot_selection_criterion_2d`."""
        info = self.models[model_index]["info"]
        if info is None:
            raise ValueError("Run select_params() first to populate info.")
        gp.modeldiagnosis.plot_selection_criterion_2d(
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
        """Wrapper around `gp.modeldiagnosis.selection_criterion_statistics_fast`."""
        info = self.models[model_index]["info"]
        if info is None:
            raise ValueError("Run select_params() first to populate info.")
        return gp.modeldiagnosis.selection_criterion_statistics_fast(
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
        """Wrapper around `gp.modeldiagnosis.selection_criterion_statistics`."""
        info = self.models[model_index]["info"]
        if info is None:
            raise ValueError("Run select_params() first to populate info.")
        return gp.modeldiagnosis.selection_criterion_statistics(
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
    def sample_parameters(
        self, method="nuts", model_indices=None, init_box=None, sampling_box=None, **kwargs
    ):
        """Sample GP parameters from the stored selection criterion.

        Parameters
        ----------
        method : {"mh", "hmc", "nuts", "smc"}, default "nuts"
            Sampling method.
            - "mh": Metropolis-Hastings sampler
            - "hmc"/"nuts": NUTS-based Hamiltonian sampling
            - "smc": Sequential Monte Carlo sampler
        model_indices : list of int, optional
            Indices of models to sample. Defaults to all models.
        init_box : list, optional
            Initialization box.
            - Required for method="smc" (particle initialization domain).
            - For "mh"/"hmc"/"nuts", used only when random_init=True.
        sampling_box : list, optional
            Domain box used to truncate the target criterion during sampling
            (outside points get log-prob = -inf). If None, sampling is unbounded.
        **kwargs
            Extra arguments passed to the selected gpmp sampler:
            - gpmp.mcmc.param_posterior.sample_from_selection_criterion_mh
            - gpmp.mcmc.param_posterior.sample_from_selection_criterion_nuts
            - gpmp.mcmc.param_posterior.sample_from_selection_criterion_smc

        Returns
        -------
        dict
            Dictionary mapping model index to:
            - method="mh": {"samples": ..., "mh": ...}
            - method="hmc"/"nuts": {"samples": ..., "nuts": ...}
            - method="smc": {"particles": ..., "smc": ...}
        """
        method = str(method).lower()
        if method not in {"mh", "hmc", "nuts", "smc"}:
            raise ValueError("method must be one of: 'mh', 'hmc', 'nuts', 'smc'.")

        if model_indices is None:
            model_indices = list(range(self.output_dim))

        results = {}
        for idx in model_indices:
            model_info = self.models[idx].get("info")
            if model_info is None:
                raise ValueError(
                    f"Model {idx} missing 'info'. Run select_params() first."
                )

            if method == "mh":
                from gpmp.mcmc.param_posterior import sample_from_selection_criterion_mh

                samples, mh = sample_from_selection_criterion_mh(
                    info=model_info,
                    init_box=init_box,
                    sampling_box=sampling_box,
                    **kwargs,
                )
                results[idx] = {"samples": samples, "mh": mh}

            elif method in {"hmc", "nuts"}:
                from gpmp.mcmc.param_posterior import (
                    sample_from_selection_criterion_nuts,
                )

                # Backward-compatible aliases for users migrating from MH-style kwargs.
                nuts_kwargs = dict(kwargs)
                if "n_steps_total" in nuts_kwargs and "num_samples" not in nuts_kwargs:
                    nuts_kwargs["num_samples"] = nuts_kwargs.pop("n_steps_total")
                if "burnin_period" in nuts_kwargs and "num_warmup" not in nuts_kwargs:
                    nuts_kwargs["num_warmup"] = nuts_kwargs.pop("burnin_period")
                if "show_progress" in nuts_kwargs and "progress" not in nuts_kwargs:
                    nuts_kwargs["progress"] = nuts_kwargs.pop("show_progress")
                if "silent" in nuts_kwargs:
                    silent = bool(nuts_kwargs.pop("silent"))
                    nuts_kwargs.setdefault("verbose", 0 if silent else 1)
                    nuts_kwargs.setdefault("progress", not silent)
                nuts_kwargs.pop("n_pool", None)

                samples, nuts_info = sample_from_selection_criterion_nuts(
                    info=model_info,
                    init_box=init_box,
                    sampling_box=sampling_box,
                    **nuts_kwargs,
                )
                key = "hmc" if method == "hmc" else "nuts"
                results[idx] = {"samples": samples, key: nuts_info}

            else:
                from gpmp.mcmc.param_posterior import sample_from_selection_criterion_smc

                if init_box is None:
                    raise ValueError("init_box must be provided when method='smc'.")
                particles, smc_instance = sample_from_selection_criterion_smc(
                    info=model_info,
                    init_box=init_box,
                    sampling_box=sampling_box,
                    **kwargs,
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

    # .................................................... parameters and bounds
    @staticmethod
    def _split_mean_cov_from_vector(p, mpl):
        p = gnp.asarray(p).reshape(-1)
        return p[:mpl], p[mpl:]

    @staticmethod
    def _param_to_vector_pair(p, model_dict, mpl):
        """Accept Param or vector; return (mean0, cov0) in normalized space.

        Parameters
        ----------
        p : Param | array_like
            Either a Param object or a 1D vector containing [mean..., cov...].
        model_dict : dict
            Per-output model record (AttrDict/dict) containing adapters.
            Expected keys:
              - "param_to_vectors": callable, mapping Param -> (meanparam, covparam)
        mpl : int
            Mean-parameter length.

        Returns
        -------
        (mean0, cov0) : (array, array)
            1D arrays in normalized coordinates.
        """
        # Param object path: rely on the container-installed adapter
        if hasattr(p, "get_by_path") and callable(getattr(p, "get_by_path")):
            pt = model_dict.get("param_to_vectors", None)
            if pt is None or not callable(pt):
                raise ValueError(
                    "param_to_vectors adapter is required to use a Param initializer."
                )
            mean0, cov0 = pt(p)
            mean0 = gnp.asarray(mean0).reshape(-1)
            cov0 = gnp.asarray(cov0).reshape(-1)
            if mean0.shape[0] != int(mpl):
                raise ValueError(
                    f"Param initializer produced mean length {mean0.shape[0]}, expected {int(mpl)}."
                )
            return mean0, cov0

        # Vector path: split [mean..., cov...]
        return ModelContainer._split_mean_cov_from_vector(p, mpl)

    @staticmethod
    def _bounds_from_param_obj(P):
        """Build (n_params, 2) array from Param.bounds (normalized)."""
        any_bound = False
        neg_inf = -float("inf")
        pos_inf = float("inf")
        out = []
        for b in P.bounds:
            if b is None:
                out.append((neg_inf, pos_inf))
            else:
                any_bound = True
                out.append((float(b[0]), float(b[1])))
        return None if not any_bound else gnp.asarray(out, dtype=float)

    @staticmethod
    def _as_bounds_array(b, n_params):
        if b is None:
            return None
        b = gnp.asarray(b, dtype=float)
        if b.ndim != 2 or b.shape[1] != 2:
            raise ValueError("bounds must have shape (n_params, 2).")
        if b.shape[0] != n_params:
            raise ValueError(f"bounds has {b.shape[0]} rows, expected {n_params}.")
        return b

    @staticmethod
    def _extract_bounds_for_output(bounds_arg, i, n_params, output_dim=None):
        """
        Extract bounds for output i and validate shape.

        Parameters
        ----------
        bounds_arg : None | array_like(n_params, 2) | list[array_like(n_params, 2)]
            Either a single bounds array to broadcast to all outputs, or a list of
            per-output bounds arrays.
        i : int
            Output index.
        n_params : int
            Expected number of parameters for output i.
        output_dim : int | None
            Required when bounds_arg is a list. If None, no length check is done.

        Returns
        -------
        ndarray | None
            Bounds array of shape (n_params, 2) or None.
        """
        if bounds_arg is None:
            return None

        if isinstance(bounds_arg, list):
            if output_dim is not None and len(bounds_arg) != int(output_dim):
                raise ValueError("When bounds is a list, its length must equal output_dim.")
            return ModelContainer._as_bounds_array(bounds_arg[i], n_params)

        return ModelContainer._as_bounds_array(bounds_arg, n_params)

    @staticmethod
    def _interval_around(p0, delta):
        """Return (n_params, 2) array [p0 - d, p0 + d]."""
        p0 = gnp.asarray(p0, dtype=float).reshape(-1)
        if gnp.isscalar(delta):
            d = gnp.ones_like(p0) * float(delta)
        else:
            d = gnp.asarray(delta, dtype=float).reshape(-1)
            if d.shape[0] != p0.shape[0]:
                raise ValueError("bounds_delta length must match number of parameters.")
        lo = p0 - d
        hi = p0 + d
        return gnp.stack([lo, hi], axis=1)

    @staticmethod
    def _intersect_bounds(a, b):
        """Intersect two (n_params, 2) arrays. If one is None, return the other."""
        if a is None:
            return b
        if b is None:
            return a
        lo = gnp.maximum(a[:, 0], b[:, 0])
        hi = gnp.minimum(a[:, 1], b[:, 1])
        return gnp.stack([lo, hi], axis=1)

    @staticmethod
    def _apply_bounds_to_param(param_obj, bounds_array, mpl):
        """
        Copy optimizer bounds (normalized space) into Param.bounds,
        respecting the ordering [mean..., cov...].
        """
        if bounds_array is None or param_obj is None:
            return
        # Collect indices in Param that correspond to mean then cov
        idx_mean = []
        idx_cov = []
        if hasattr(param_obj, "indices_by_path_prefix"):
            idx_mean = param_obj.indices_by_path_prefix(["meanparam"])
            idx_cov = param_obj.indices_by_path_prefix(["covparam"])
        else:
            # Fallback: assume the Param was built as [mean..., cov...] in order
            idx_mean = list(range(mpl))
            idx_cov = list(range(mpl, mpl + (bounds_array.shape[0] - mpl)))

        seq = idx_mean + idx_cov
        if len(seq) != bounds_array.shape[0]:
            # As a stricter fallback, map first mpl to mean and the rest to cov
            seq = list(range(bounds_array.shape[0]))

        # Ensure bounds list exists and is sized
        if not hasattr(param_obj, "bounds") or len(param_obj.bounds) < len(seq):
            return  # nothing we can safely do

        for pos, idx in enumerate(seq):
            lo, hi = bounds_array[pos]
            # Store as a tuple of floats; if unbounded, they will be +/- inf
            param_obj.bounds[idx] = (float(lo), float(hi))

    # ................................................................ check data
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
