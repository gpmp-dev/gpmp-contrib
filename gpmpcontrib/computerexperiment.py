"""
Multi-output deterministic or stochastic computer experiments

Author: Emmanuel Vazquez <emmanuel.vazquez@centralesupelec.fr>
Copyright (c) 2022-2024, CentraleSupelec
License: GPLv3 (see LICENSE)

"""

import gpmp.num as gnp
import numpy as np

##############################################################################
#                                                                            #
#                         ComputerExperiment Class                           #
#                                                                            #
##############################################################################


class ComputerExperiment:
    """A class representing a multi-output computer experiment, to evaluate
    multiple objectives and constraints simultaneously. This class allows the user to
    specify a set of functions (or a single function) to be evaluated as part of
    a computational experiment.

    The user can provide the functions in a variety of ways, including as individual
    objective or constraint functions, or a combined function that outputs both
    objectives and constraints.

    Parameters
    ----------
    input_dim : int
        Dimension of the input space.
    input_box : list of tuples
        Input domain specified as a list of tuples. Each tuple represents the range
        for a specific input dimension as (min_value, max_value).
    single_function : callable or dict, optional
        A single function that evaluates both objectives and constraints.
        If provided as a dictionary, it must contain a "function" key that specifies
        the function, and optionally, keys such as "output_dim", "type", and "bounds."
        This is useful to define a single function that handles all outputs.
    function_list : list of callable or list of dict, optional
        A list of functions (or dictionaries) to evaluate. Each entry can either be a
        callable function or a dictionary that includes keys such as "function",
        "output_dim", "type", etc.
    single_objective : callable or dict, optional
        A single objective function. If a dictionary is provided, it must contain a
        "function" key, and optionally, "output_dim" and "type."
    objective_list : list of callable or list of dict, optional
        A list of objective functions. Each can either be a function or a dictionary with
        additional information.
    single_constraint : callable or dict, optional
        A single constraint function. If provided as a dictionary, it must include a
        "function" key and a "bounds" key that indicates the constraint bounds.
    constraint_list : list of callable or list of dict, optional
        A list of constraint functions. Each entry can be a function or a dictionary
        containing the constraint and its corresponding bounds.
    constraint_bounds : list of tuples, optional
        A list of constraint bounds (lower_bound, upper_bound) for the constraints.
        This is used if constraints are not provided as dictionaries and need
        global bounds.

    Raises
    ------
    ValueError
        - If both 'function_list'/'single_function' and 'objective_list'/'single_objective'
          or 'constraint_list'/'single_constraint' are provided simultaneously.
        - If a dictionary function does not include a 'function' key.
        - If a constraint dictionary does not include a 'bounds' key.
        - If the number of 'constraint_bounds' does not match the number of constraints.
        - If 'constraint_bounds' elements are not 2-element tuples.

    Attributes
    ----------
    input_dim : int
        Dimension of the input space.
    input_box : list of tuples
        The input domain, representing the ranges of each input variable.
    output_dim : int
        Total number of outputs (sum of output dimensions from all objectives and constraints).
    functions : list of dict
        A list of all functions (both objectives and constraints), where each entry is a dictionary.
    _last_x : tuple
        The last input value that was evaluated.
    _last_result : np.ndarray
        The last result from the evaluation of the functions.

    Example
    -------
    Using the `ComputerExperiment` class to define and evaluate functions:

    1.  **Single objective with two constraint**:

        ```python
        import numpy as np

        def objective(x):
            return (x[:, 0] - 10)**3 + (x[:, 1] - 20)**3

        def constraints(x):
            c1 = - (x[:, 0] - 5)**2 - (x[:, 1] - 5)**2 + 100
            c2 = (x[:, 0] - 6)**2 + (x[:, 1] - 5)**2 - 82.81
            return np.column_stack((c1, c2))

        _pb_dict = {
            "input_dim": 2,
            "input_box": [[13, 0], [100, 100]],
            "single_objective": objective,
            "single_constraint": {'function': constraints,
                                  'output_dim': 2,
                                  'bounds': [[100., np.inf], [-np.inf, 82.81]]}
        }

        pb = ComputerExperiment(
            _pb_dict["input_dim"],
            _pb_dict["input_box"],
            single_objective=_pb_dict["single_objective"],
            single_constraint=_pb_dict["single_constraint"]
        )

        ```

    2.  **Single combined function for objectives and constraints**:

        ```python
        def combined_function(x):
            obj = (x[:, 0] - 10)**3 + (x[:, 1] - 20)**3
            c1 = - (x[:, 0] - 5)**2 - (x[:, 1] - 5)**2 + 100
            c2 = (x[:, 0] - 6)**2 + (x[:, 1] - 5)**2 - 82.81
            return np.column_stack((obj, c1, c2))

        pb = ComputerExperiment(
            input_dim=2,
            input_box=[[0, 100], [0, 100]],
            single_function={
                'function': combined_function,
                'output_dim': 1 + 2,
                'type': ['objective', 'constraint', 'constraint'],
                'bounds': [None, [100., np.inf], [-np.inf, 82.81]]
            }
        )

        ```

    3. **Multiple objective functions and constraints**:

        ```python
        def objective1(x):
            return (x[:, 0] - 10)**3

        def objective2(x):
            return (x[:, 1] - 20)**2

        def constraint1(x):
            return - (x[:, 0] - 5)**2 - (x[:, 1] - 5)**2 + 100

        def constraint2(x):
            return (x[:, 0] - 6)**2 + (x[:, 1] - 5)**2 - 82.81

        pb = ComputerExperiment(
            input_dim=2,
            input_box=[[0, 100], [0, 100]],
            objective_list=[objective1, objective2],
            constraint_list=[
                {'function': constraint1, 'bounds': [100., np.inf]},
                {'function': constraint2, 'bounds': [-np.inf, 82.81]}
            ]
        )
        ```

    4.  **Evaluate functions for objectives and constraints**:


        ```python
        x = np.array([[50.0, 50.0], [80.0, 80.0]])

        # Evaluate both objectives and constraints at once
        results = pb.eval(x)   # or equivalently: results = pb(x)

        # Extract objectives and constraints separately
        objectives_results = pb.eval_objectives(x)
        constraints_results = pb.eval_constraints(x)
        ```
        Note: The functions are not re-evaluated if the same x is provided
        again. The class uses caching to store the result of the last
        evaluated input. If the input x remains unchanged, the cached
        result is returned, improving performance for repeated
        evaluations.

    Usage Notes:
    ------------
    - The `single_function` approach is useful when the user wants to
      provide a single callable that returns both objectives and
      constraints at once, which is useful when the objectives and
      constraints are tightly coupled.
    - The `objective_list` and `constraint_list` approaches are more
      modular and are recommended for distinct and separate
      functions for each objective and constraint.
    - The `bounds` field in constraints specifies constraint
      violations (i.e., values falling outside the provided
      bounds). If bounds are not provided, the class raises an error
      for constraint functions.
    - The `eval_objectives` and `eval_constraints` methods can be used
      to evaluate objectives and constraints separately, even when a
      combined function is used.

    """

    def __init__(
        self,
        input_dim,
        input_box,
        single_function=None,
        function_list=None,
        single_objective=None,
        objective_list=None,
        single_constraint=None,
        constraint_list=None,
        constraint_bounds=None,
    ):
        # Verify input_box has correct structure and dimensions
        if len(input_box) != 2:
            raise ValueError(
                f"Input box must have exactly 2 elements (lower and upper bounds), but got {len(input_box)}."
            )

        if len(input_box[0]) != input_dim or len(input_box[1]) != input_dim:
            raise ValueError(
                f"Each bound in input_box must have {input_dim} dimensions, "
                + f"but got {len(input_box[0])} for lower bounds and {len(input_box[1])} for upper bounds."
            )

        self.input_dim = input_dim
        self.input_box = input_box
        self.functions = []

        self._last_x = None
        self._last_result = None

        self._validate_inputs(
            single_function,
            function_list,
            single_objective,
            objective_list,
            single_constraint,
            constraint_list,
        )

        self._setup_functions(function_list, single_function, "function")
        self._setup_functions(objective_list, single_objective, "objective")
        self._setup_functions(
            constraint_list, single_constraint, "constraint", constraint_bounds
        )
        self._set_output_dim()

    def _validate_inputs(
        self,
        single_function,
        function_list,
        single_objective,
        objective_list,
        single_constraint,
        constraint_list,
    ):
        if (function_list is not None or single_function is not None) and (
            objective_list is not None
            or constraint_list is not None
            or single_objective is not None
            or single_constraint is not None
        ):
            raise ValueError(
                "Either provide 'function_list'/'single_function' "
                + "or 'objective_list'/'single_objective' "
                + "and 'constraint_list'/'single_constraint', but not both."
            )

    def _setup_functions(self, func_list, single_func, func_type, default_bounds=None):
        funcs = self._to_list(func_list) + self._to_list(single_func)
        bounds_index = 0
        for func in funcs:
            func_dict = self._wrap_in_dict(func, func_type)
            if func_type == "constraint":
                func_dict, bounds_index = self._handle_constraint(
                    func_dict, default_bounds, bounds_index
                )
            self.functions.append(func_dict)

    def _handle_constraint(self, func_dict, default_bounds, bounds_index):
        if "bounds" not in func_dict:
            if default_bounds is not None and bounds_index < len(default_bounds):
                # If the function has multiple outputs, get a slice of the bounds list
                d = func_dict["output_dim"]
                if d > 1:
                    if not all(
                        isinstance(b, (tuple, list)) and len(b) == 2
                        for b in default_bounds[bounds_index: bounds_index + d]
                    ):
                        raise ValueError(
                            "Each set of bounds should be a tuple (lb, ub) of length 2."
                        )
                    func_dict["bounds"] = default_bounds[
                        bounds_index: bounds_index + d
                    ]
                    bounds_index += d
                # If the function has one output, just get the next set of bounds
                else:
                    if (
                        not isinstance(
                            default_bounds[bounds_index], (tuple, list))
                        or len(default_bounds[bounds_index]) != 2
                    ):
                        raise ValueError(
                            "Bounds should be a tuple of length 2.")
                    func_dict["bounds"] = default_bounds[bounds_index]
                    bounds_index += 1
            else:
                raise ValueError("Constraint function must have 'bounds'.")
        return func_dict, bounds_index

    def _to_list(self, item):
        if item is None:
            return []
        elif isinstance(item, list):
            return item
        else:
            return [item]

    def _wrap_in_dict(self, item, default_type):
        if isinstance(item, dict):
            if "function" not in item:
                raise ValueError(
                    "The 'function' key is mandatory in the dictionary.")
            item.setdefault("output_dim", 1)
            item.setdefault("type", [default_type] * item["output_dim"])
            if len(item["type"]) != item["output_dim"]:
                raise ValueError(
                    f"The length of 'type' list {len(item['type'])} "
                    f"should match 'output_dim' {item['output_dim']}"
                )
        else:
            item = {"function": item, "output_dim": 1, "type": [default_type]}
        return item

    def _set_output_dim(self):
        self.output_dim = sum(func["output_dim"] for func in self.functions)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        details = [
            f"Computer Experiment object",
            f"      Input Dimension : {self.input_dim}",
            f"            Input Box : {self.input_box}",
            f"     Output Dimension : {self.output_dim}\n",
        ]
        for i, func in enumerate(self.functions, 1):
            details.extend(
                [
                    f"     ** Function {i} **",
                    f"                 Type : {func['type']}",
                    f"             Function : {func['function'].__name__ if callable(func['function']) else func['function']}",
                    f"     Output Dimension : {func['output_dim']}",
                    (
                        f"               Bounds : {func['bounds']}"
                        if "bounds" in func
                        else ""
                    ),
                ]
            )

        return "\n".join(details)

    def __call__(self, x):
        """
        Allows the instance to be called like a function, which internally calls the eval method.

        Parameters
        ----------
        x : array_like
            The input values at which to evaluate the functions.

        Returns
        -------
        ndarray
            The evaluated results from the function or functions.
        """
        return self.eval(x)

    def get_constraint_bounds(self):
        return np.array(
            [func["bounds"]
                for func in self.functions if func["type"] == "constraint"]
        )

    def eval(self, x):
        x_tuple = tuple(x) if x.ndim == 1 else tuple(map(tuple, x))
        if self._last_x is not None and self._last_x == x_tuple:
            return self._last_result
        else:
            result = self._eval_functions(self.functions, x)
            self._last_x = x_tuple
            self._last_result = result
            return result

    def eval_objectives(self, x):
        results = self.eval(x)
        all_types = [t for func in self.functions for t in func["type"]]
        return results[:, [i for i, t in enumerate(all_types) if t == "objective"]]

    def eval_constraints(self, x):
        results = self.eval(x)
        all_types = [t for func in self.functions for t in func["type"]]
        return results[:, [i for i, t in enumerate(all_types) if t == "constraint"]]

    def _eval_functions(self, function_dicts, x):
        # Check if x has the correct number of dimensions
        if x.ndim != 2 or x.shape[1] != self.input_dim:
            raise ValueError(
                f"Input x must be a 2D array with shape (n, {self.input_dim}), "
                f"but got shape {x.shape}."
            )

        results = []
        for func in function_dicts:
            current_function = func["function"]
            current_output_dim = func["output_dim"]

            result_temp = current_function(x)

            if result_temp.ndim == 1:
                result_temp = result_temp[:, np.newaxis]
            elif result_temp.ndim == 2 and result_temp.shape[1] != current_output_dim:
                raise ValueError(
                    f"The function output dimension {result_temp.shape[1]} "
                    f"does not match the expected output dimension {current_output_dim}."
                )

            results.append(result_temp)

        return np.concatenate(results, axis=1)


##############################################################################
#                                                                            #
#                 StochasticComputerExperiment Class                         #
#                                                                            #
##############################################################################


class StochasticComputerExperiment(ComputerExperiment):
    """
    A class representing stochastic computer experiments, extending
    the ComputerExperiment class for stochastic functions. (Gaussian)
    noise can be added on outputs. This class also enables evaluations
    by batch at a point to get multiple realizations in a single call.

    Attributes
    ----------
    simulated_noise_variance : array-like
        The variance of the noise to be simulated for each function.

    Methods
    -------
    __init__(input_dim, input_box,
             single_function=None, function_list=None,
             single_objective=None, objective_list=None,
             single_constraint=None, constraint_list=None,
             simulated_noise_variance=None)
        Constructs an instance of StochasticComputerExperiment and
        initializes its attributes.

    eval(x, simulated_noise_variance=True, batch_size=1)
        Evaluates all functions (objectives and constraints), with
        simulated noise if required.  If a batch_size is provided, it
        returns an array of size n x output_dim x batch_size.  If
        batch_size is 1, it returns a matrix of size n x
        output_dim. The simulated_noise_variance parameter can be set
        to False to perform noise-free evaluations to get ground
        noise-free evaluations when noise is simulated.

    eval_objectives(x, simulated_noise_variance=True, batch_size=1)
        Evaluates only the objectives for the given input with simulated noise.

    eval_constraints(x, simulated_noise_variance=True, batch_size=1)
        Evaluates only the constraints for the given input with simulated noise.

    Notes
    -----
    The simulated_noise_variance property describes the variance of
    simulated noise.  This is useful when the user wants add simulated
    noise on outputs to assess the performance of an algorithm in
    presence of stochastic evaluations.

    - If simulated_noise_variance is an array of variances, Gaussian
      noise is added on each output with noise variance specified by
      the elements of simulated_noise_variance.

    - If simulated_noise_variance is None, the user should provide the
      functions to be evaluated as dictionaries with a key
      "simulated_noise_variance". If the key is absent, no simulated noise
      is added.

    """

    def __init__(
        self,
        input_dim,
        input_box,
        single_function=None,
        function_list=None,
        single_objective=None,
        objective_list=None,
        single_constraint=None,
        constraint_list=None,
        simulated_noise_variance=None,
    ):
        """
        Initialize a StochasticComputerExperiment instance.

        Parameters
        ----------
        input_dim : int
            The dimension of the input space.
        input_box : list of tuples
            The input domain specified as a list of tuples.
        single_function : callable function or dict, optional
            A single function to be evaluated.
        function_list : list of callable functions or dicts, optional
            List of functions to be evaluated.
        single_objective : function or dict, optional
            A single objective function to be evaluated.
        objective_list : list of functions or dicts, optional
            List of objective functions to be evaluated.
        single_constraint : function or dict, optional
            A single constraint function to be evaluated.
        constraint_list : list of functions or dicts, optional
            List of constraint functions to be evaluated.
        simulated_noise_variance : array-like or None, optional
            The variance of the noise to be simulated for each function.

        """
        # problem setting
        super().__init__(
            input_dim,
            input_box,
            single_function=single_function,
            function_list=function_list,
            single_objective=single_objective,
            objective_list=objective_list,
            single_constraint=single_constraint,
            constraint_list=constraint_list,
        )

        # Initialize noise variance
        self.initialize_noise_variance(
            self.functions, simulated_noise_variance)

    def initialize_noise_variance(self, function_dicts, simulated_noise_variance):
        """
        Initialize noise variance from provided values or function dictionaries.

        Parameters
        ----------
        function_dicts : list of dict

            Functions to be evaluated. Each dictionary must contain
            'function' and 'output_dim' keys. Each dictionary may have a 'simulated_noise_variance' key.

        simulated_noise_variance : array-like, scalar, or None
            Variances of the noise added on the outputs. If a scalar,
            it's expanded to an array. If None, the variance will be
            extracted from function dictionaries. If no
            'simulated_noise_variance' key is provided in the
            dictionaries, no simulated noise is added on the outputs.

        """
        # Convert scalar to array
        if np.isscalar(simulated_noise_variance):
            total_output_dim = sum(f["output_dim"] for f in function_dicts)
            simulated_noise_variance = [
                simulated_noise_variance] * total_output_dim

        if simulated_noise_variance is not None:
            # Ensure that simulated_noise_variance has output_dim components
            assert len(simulated_noise_variance) == sum(
                [f["output_dim"] for f in function_dicts]
            ), "Total length of 'simulated_noise_variance' should match total 'output_dim'."
            start = 0
            for f in function_dicts:
                end = start + f["output_dim"]
                f["simulated_noise_variance"] = simulated_noise_variance[start:end]
                start = end
        else:
            for f in function_dicts:
                if not "simulated_noise_variance" in f:
                    f["simulated_noise_variance"] = [0.0] * f["output_dim"]

    def __str__(self):
        details = [
            f"Stochastic Computer Experiment:",
            f"  Input Dimension: {self.input_dim}",
            f"  Input Box: {self.input_box}",
            f"  Output Dimension: {self.output_dim}",
            f"  Functions:",
        ]
        for i, func in enumerate(self.functions, 1):
            details.extend(
                [
                    f"    * Function {i}:",
                    f"      Type: {func['type']}",
                    f"      Function: {func['function'].__name__ if callable(func['function']) else func['function']}",
                    f"      Output Dimension: {func['output_dim']}",
                    f"      Simulated Noise Variance: {func['simulated_noise_variance']}",
                    f"      Bounds: {func['bounds']}" if "bounds" in func else "",
                ]
            )

        return "\n".join(details)

    import numpy as np

    def __call__(self, x, simulate_noise=True, batch_size=1):
        """
        Allows the instance to be called like a function, which
        internally calls the eval method.

        """
        return self.eval(x, simulate_noise, batch_size)

    @property
    def simulated_noise_variance(self):
        """
        Returns the simulated noise variances for each function.

        Returns
        -------
        numpy.ndarray
            The simulated noise variances for each function.
        """
        return self.get_simulated_noise_variances()

    def get_simulated_noise_variances(self):
        """
        Returns the simulated noise variances for each function.

        Returns
        -------
        numpy.ndarray
            The simulated noise variances for each function.
        """
        return np.concatenate(
            [func["simulated_noise_variance"] for func in self.functions]
        )

    def eval(self, x, simulate_noise=True, batch_size=1):
        """
        Evaluate all functions (objectives and constraints) for the given input.
        Include simulated noise if specified.

        Parameters
        ----------
        x : array-like
            Input values.
        simulate_noise : bool
            If True, use the internal simulated_noise_variance.
            If False, no noise is added on the outputs.
            By default False.
        batch_size : int
            The number of batches to evaluate.
            If 1, the result will have shape (n, output_dim).
            If greater than 1, the result will have shape (n, output_dim, batch_size).

        Returns
        -------
        array-like
            Function values for the given input, with optional noise.
        """

        return self._eval_batch(self.functions, x, simulate_noise, batch_size)

    def eval_objectives(self, x):
        raise NotImplementedError(
            "eval_objectives method is not supported in StochasticComputerExperiment."
        )

    def eval_constraints(self, x):
        raise NotImplementedError(
            "eval_constraints method is not supported in StochasticComputerExperiment."
        )

    def _eval_batch(self, function_dicts, x, simulate_noise, batch_size):
        """
        Evaluate all functions (objectives and constraints) for the given input.
        Include simulated noise if specified.

        Parameters
        ----------
        x : array-like
            Input values.
        simulate_noise : bool
            See eval()
        batch_size : int
            The number of batches to evaluate.
            If 1, the result will have shape (n, output_dim).
            If greater than 1, the result will have shape (n, output_dim, batch_size).

        Returns
        -------
        array-like
            Function values for the given input, with optional noise.
        """

        assert batch_size > 0, "Batch size must be a positive integer."

        results = []
        for _ in range(batch_size):
            results.append(self._eval_functions(
                function_dicts, x, simulate_noise))

        if batch_size == 1:
            return results[0]
        else:
            return np.dstack(results)

    def _eval_functions(self, function_dicts, x, simulate_noise):
        """Evaluate the provided functions for the given input and
        add simulated noise if specified.

        Parameters
        ----------
        function_dicts : list of dict
            Functions to be evaluated. The list contains dictionaries with 'function' key
            containing the function to be evaluated.
        x : array-like
            Input values.
        simulate_noise : bool
            If True, add Gaussian noise using the internal 'simulated_noise_variance' of
            each function
            If False, no noise is added

        Returns
        -------
        array-like
            Function values for the given input, with simulated noise added.

        """
        z_ = []

        for func in function_dicts:
            current_function = func["function"]
            current_output_dim = func["output_dim"]

            z_temp = current_function(x)

            # Ensure that the output has the correct shape (n, output_dim)
            if z_temp.ndim == 1:
                z_temp = z_temp[:, np.newaxis]
            elif z_temp.ndim == 2 and z_temp.shape[1] != current_output_dim:
                z_temp = z_temp.reshape((-1, current_output_dim))

            # Add simulated noise
            if simulate_noise:
                for i in range(current_output_dim):
                    if func["simulated_noise_variance"][i] > 0.0:
                        z_temp[:, i] += np.random.normal(
                            0.0,
                            np.sqrt(func["simulated_noise_variance"][i]),
                            size=z_temp[:, i].shape,
                        )

            z_.append(z_temp)

        z = np.hstack(z_)

        return z
