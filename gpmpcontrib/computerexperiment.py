"""
Multi-output deterministic or stochastic computer experiments

Author: Emmanuel Vazquez <emmanuel.vazquez@centralesupelec.fr>
Copyright (c) 2022-2023, CentraleSupelec
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
    """A class representing a computer experiment problem, allowing the user to
    specify and evaluate functions as either objectives or constraints.

    Parameters
    ----------
    input_dim : int
        Dimension of the input space.
    input_box : list of tuples
        Input domain specified as a list of tuples. Each tuple represents
        the range for a specific dimension as (min_value, max_value).
    single_function : callable function or dict, optional
        A single function to be evaluated. If a dictionary is
        provided, it must contain a "function" key. Other optional keys
        include "output_dim", "type", "bounds" or user-defined keys.
    function_list : list of callable functions or list of dicts, optional
        List of functions to be evaluated. Each function can be a
        dictionary with the key "function" and optional keys
        "output_dim", "type"...
    single_objective : function or dict, optional
        A single objective function to be evaluated. If a dictionary
        is provided, it must contain a "function" key. Other optional keys
        include "output_dim" and "type".
    objective_list : list of functions or list of dicts, optional
        List of objective functions to be evaluated. Each function can
        be a dictionary with the key "function" and optional keys
        "output_dim", "type"...
    single_constraint : function or dict, optional
        A single constraint function to be evaluated. If a dictionary
        is provided, it must contain a "function" key and a "bounds" key
        indicating the constraint bounds.
    constraint_list : list of functions or list of dicts, optional
        List of constraint functions to be evaluated. Each function
        can be a dictionary with the key "function" and a "bounds" key
        indicating the constraint bounds.
    constraint_bounds : list of tuples, optional
        List of bounds for constraints. Each bound is a 2-element tuple: (lower_bound, upper_bound).
        The number of constraint bounds should match the number of constraints.
        Only used if constraints are not dictionaries and a global constraint bound is needed.

    Raises
    ------
    ValueError
        If both 'function_list'/'single_function' and either
        'objective_list'/'single_objective' or
        'constraint_list'/'single_constraint' are provided.

        If a dictionary does not include a 'function' key.

        If a constraint dictionary does not include a 'bounds' key.

        If the number of constraint bounds does not match the number of constraints.

        If each element in 'constraint_bounds' is not a 2-element tuple.

    Attributes
    ----------
    input_dim : int
        Dimension of the input space.
    input_box : list of tuples
        Input domain.
    output_dim : int
        Total number of function outputs (sum of output dimensions of objectives and constraints).
    functions : list of dicts
        List of all function dictionaries (objectives and constraints).
    _last_x : tuple
        Last input value evaluated.
    _last_result : np.ndarray
        Last result computed from the evaluation.

    Example
    -------
    Here's how you can use the ComputerExperiment class:

    ```python
    import numpy as np

    def _g6_objective(x):
        return (x[:, 0] - 10)**3 + (x[:, 1] - 20)**3

    def _g6_constraints(x):
        c1 = - (x[:, 0] - 5)**2 - (x[:, 1] - 5)**2 + 100
        c2 = (x[:, 0] - 6)**2 + (x[:, 1] - 5)**2 - 82.81
        return np.column_stack((c1, c2))

    _g6_dict = {
        "input_dim": 2,
        "input_box": [[13, 0], [100, 100]],
        "single_objective": _g6_objective,
        "single_constraint": {'function': _g6_constraints, 'output_dim': 2, 'bounds': [[100., np.inf], [-np.inf, 82.81]]}
    }

    g6 = ComputerExperiment(
        _g6_dict["input_dim"],
        _g6_dict["input_box"],
        single_objective=_g6_dict["single_objective"],
        single_constraint=_g6_dict["single_constraint"]
    )

    # alternative definition

    def _g6_evaluation(x):
        return np.column_stack((_g6_objective(x), _g6_constraints(x)))

    g6 = ComputerExperiment(
        _g6_dict["input_dim"],
        _g6_dict["input_box"],
        single_function={
            'function': _g6_evaluation,
            'output_dim': 1+ 2,
            'type': ["objective"] + ["constraint"] * 2,
            'bounds': [None] + [[100., np.inf], [-np.inf, 82.81]]
        }
    )
    print(g6)
    x = np.array([[50.0, 50.0], [80., 80.]])
    g6.eval(x)
    g6.eval_constraints(x) # Note that this will use the previous computation
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
                        for b in default_bounds[bounds_index : bounds_index + d]
                    ):
                        raise ValueError(
                            "Each set of bounds should be a tuple (lb, ub) of length 2."
                        )
                    func_dict["bounds"] = default_bounds[
                        bounds_index : bounds_index + d
                    ]
                    bounds_index += d
                # If the function has one output, just get the next set of bounds
                else:
                    if (
                        not isinstance(default_bounds[bounds_index], (tuple, list))
                        or len(default_bounds[bounds_index]) != 2
                    ):
                        raise ValueError("Bounds should be a tuple of length 2.")
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
                raise ValueError("The 'function' key is mandatory in the dictionary.")
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
            f"Computer Experiment:",
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
                    f"      Bounds: {func['bounds']}" if "bounds" in func else "",
                ]
            )

        return "\n".join(details)

    def get_constraint_bounds(self):
        return np.array(
            [func["bounds"] for func in self.functions if func["type"] == "constraint"]
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
    A class representing stochastic computer experiments. It is a subclass of the ComputerExperiment class
    and extends its functionality to include simulation of noise.

    Attributes
    ----------
    simulated_noise_variance : array-like
        The variance of the noise to be simulated.

    Methods
    -------
    __init__(input_dim, input_box, single_function=None, function_list=None, single_objective=None,
             objective_list=None, single_constraint=None, constraint_list=None, simulated_noise_variance=None)
        Constructs an instance of StochasticComputerExperiment and initializes the attributes.

    eval(x, simulated_noise_variance, batch_size)
        Evaluates all functions (objectives and constraints) for the given input with simulated noise.
        If a batch_size is provided, it returns a tensor of size n x output_dim x batch_size.
        If batch_size is 1, it returns a matrix of size n x output_dim.

    eval_objectives(x, simulated_noise_variance, batch_size)
        Evaluates only the objectives for the given input with simulated noise.

    eval_constraints(x, simulated_noise_variance, batch_size)
        Evaluates only the constraints for the given input with simulated noise.

    Notes
    -----
    If simulated_noise_variance is set to True, the internal simulated_noise_variance is used.
    If simulated_noise_variance is False, the result is noise-free.
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
        self.initialize_noise_variance(self.functions, simulated_noise_variance)

    def initialize_noise_variance(self, function_dicts, simulated_noise_variance):
        """
        Initialize noise variance from provided values or function dictionaries.

        Parameters
        ----------
        function_dicts : list of dict
            Functions to be evaluated. Each dictionary must contain 'function', 'output_dim', and 'simulated_variance' keys.
        simulated_noise_variance : array-like or None
            The provided noise variance. If None, the variance will be extracted from function dictionaries.
        """
        if simulated_noise_variance is not None:
            # Ensure that simulated_noise_variance has output_dim components
            assert len(simulated_noise_variance) == sum(
                [f["output_dim"] for f in function_dicts]
            ), "Total length of 'simulated_noise_variance' should match total 'output_dim'."
            start = 0
            for f in function_dicts:
                end = start + f["output_dim"]
                f["simulated_variance"] = simulated_noise_variance[start:end]
                start = end
        else:
            for f in function_dicts:
                if not "simulated_variance" in f:
                    f["simulated_variance"] = [0.0] * f["output_dim"]

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
                    f"      Simulated Variance: {func['simulated_variance']}",
                    f"      Bounds: {func['bounds']}" if "bounds" in func else "",
                ]
            )

        return "\n".join(details)

    import numpy as np


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
        return np.concatenate([func['simulated_variance'] for func in self.functions])

    def eval(self, x, simulated_noise_variance=True, batch_size=1):
        """
        Evaluate all functions (objectives and constraints) for the given input.

        Parameters
        ----------
        x : array-like
            Input values.
        simulated_noise_variance : bool
            If True, use the internal simulated_noise_variance.
            If False or 0.0, the result is noise free.
            By default True.
        batch_size : int
            The number of batches to evaluate.
            If 1, the result will have shape (n, output_dim).
            If greater than 1, the result will have shape (n, output_dim, batch_size).

        Returns
        -------
        array-like
            Function values for the given input, with optional noise.
        """

        return self._eval_batch(self.functions, x, simulated_noise_variance, batch_size)

    def _eval_batch(self, function_dicts, x, simulated_noise_variance, batch_size):
        """
        Evaluate all functions (objectives and constraints) for the given input.
        Include simulated noise if specified.

        Parameters
        ----------
        x : array-like
            Input values.
        simulated_noise_variance : bool
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
            results.append(
                self._eval_functions(function_dicts, x, simulated_noise_variance)
            )

        if batch_size == 1:
            return results[0]
        else:
            return np.dstack(results)

    def _eval_functions(self, function_dicts, x, simulated_noise_variance):
        """
        Evaluate the provided functions for the given input and add simulated noise.

        Parameters
        ----------
        function_dicts : list of dict
            Functions to be evaluated. The list contains dictionaries with 'function' key
            containing the function to be evaluated.
        x : array-like
            Input values.
        simulated_noise_variance : bool
            If True, use the internal 'simulated_variance' of each function.
            If False or 0.0, the result is noise free.

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
            if simulated_noise_variance:
                for i in range(current_output_dim):
                    if func["simulated_variance"][i] > 0.0:
                        z_temp[:, i] += np.random.normal(
                            0.0,
                            np.sqrt(func["simulated_variance"][i]),
                            size=z_temp[:, i].shape,
                        )

            z_.append(z_temp)

        z = np.hstack(z_)

        return z
