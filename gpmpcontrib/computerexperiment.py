"""
Multi-output deterministic or stochastic computer experiments.

Author: Emmanuel Vazquez <emmanuel.vazquez@centralesupelec.fr>
Copyright (c) 2022-2026, CentraleSupelec
License: GPLv3 (see LICENSE)
"""

from __future__ import annotations

from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import gpmp.num as gnp
import numpy as np

ArrayLike = Union[np.ndarray, Sequence[float]]
Bounds = Optional[Tuple[float, float]]
FuncSpec = Union[
    Callable[[np.ndarray], np.ndarray],
    Dict[str, Union[Callable[[np.ndarray], np.ndarray], int, List[str], List[Bounds], List[float]]],
]


##############################################################################
#                                                                            #
#                         ComputerExperiment Class                           #
#                                                                            #
##############################################################################


class ComputerExperiment:
    """
    Represent a multi-output computer experiment evaluating objectives and constraints.

    You can supply either:
      - a combined function (or list of functions) via `single_function` / `function_list`,
        whose outputs are tagged with types; or
      - separate objectives and constraints via `single_objective`/`objective_list`
        and `single_constraint`/`constraint_list`.

    Input domain (`input_box`) is accepted as either shape (2, d) [lower; upper] or (d, 2)
    [(l1, u1), ..., (ld, ud)]. Internally it is stored as (2, d).

    Parameters
    ----------
    input_dim : int
        Dimension of the input space.
    input_box : array-like
        Either (2, d) or (d, 2). Bounds must be finite with upper > lower per dimension.
    single_function, function_list : FuncSpec or list[FuncSpec], optional
        Combined function(s). Dict entries may include:
          - "function": callable(X) -> (n, p_i) or (n,)
          - "output_dim": int (default 1)
          - "type": list[str] of length output_dim with values in {"objective","constraint"}
          - "bounds": list[Bounds] or Bounds for constraints (per output); None for objectives.
    single_objective, objective_list : FuncSpec or list[FuncSpec], optional
        Objective function(s). Dict like above; "type" defaults to "objective".
    single_constraint, constraint_list : FuncSpec or list[FuncSpec], optional
        Constraint function(s). Dict like above; must provide bounds if not set via `constraint_bounds`.
    constraint_bounds : list[Bounds], optional
        Global bounds to attach to constraint functions provided without "bounds".
        Length must match total number of constraint outputs.

    Raises
    ------
    ValueError
        If incompatible inputs are provided or shapes are inconsistent.

    Notes
    -----
    - `normalize_input`: if True, `eval` assumes x ∈ [0,1]^d and scales to the physical box.
      You can also override per-call via `eval(..., normalize_input=True/False)`.
    """

    def __init__(
        self,
        input_dim: int,
        input_box: ArrayLike,
        *,
        single_function: Optional[FuncSpec] = None,
        function_list: Optional[List[FuncSpec]] = None,
        single_objective: Optional[FuncSpec] = None,
        objective_list: Optional[List[FuncSpec]] = None,
        single_constraint: Optional[FuncSpec] = None,
        constraint_list: Optional[List[FuncSpec]] = None,
        constraint_bounds: Optional[List[Bounds]] = None,
    ) -> None:
        # Core dims and domain
        self.input_dim = int(input_dim)
        self._parse_and_set_input_box(self.input_dim, input_box)

        # Normalization (backing field; do NOT call property in __init__)
        self._normalize_input: bool = False
        self.input_box = self.input_box_org  # current box mirrors original when not normalized

        # Storage and cache
        self.functions: List[Dict] = []
        self._last_x: Optional[Tuple] = None           # cache key: (use_norm: bool, tuple(rows))
        self._last_result: Optional[np.ndarray] = None

        # Validate mutually exclusive API paths
        self._validate_inputs(
            single_function,
            function_list,
            single_objective,
            objective_list,
            single_constraint,
            constraint_list,
        )

        # Register functions
        self._setup_functions(function_list, single_function, default_type="function")
        self._setup_functions(objective_list, single_objective, default_type="objective")
        self._setup_functions(
            constraint_list,
            single_constraint,
            default_type="constraint",
            default_bounds=constraint_bounds,
        )

        # Finalize shape metadata
        self._set_output_dim()

    # --------------------------- helpers & validation ---------------------------

    def _parse_and_set_input_box(self, input_dim: int, input_box: ArrayLike) -> None:
        arr = np.asarray(input_box, dtype=float)
        if arr.shape == (2, input_dim):
            lower, upper = arr[0], arr[1]
        elif arr.shape == (input_dim, 2):
            lower, upper = arr[:, 0], arr[:, 1]
        else:
            raise ValueError(
                f"input_box must be shape (2, {input_dim}) or ({input_dim}, 2), got {arr.shape}."
            )
        if not (np.all(np.isfinite(lower)) and np.all(np.isfinite(upper))):
            raise ValueError("input_box bounds must be finite.")
        if np.any(upper <= lower):
            raise ValueError("Each upper bound must be strictly greater than the lower bound.")
        # store as (2, d)
        self.input_box_org = gnp.array(np.vstack([lower, upper]))

    def _validate_inputs(
        self,
        single_function: Optional[FuncSpec],
        function_list: Optional[List[FuncSpec]],
        single_objective: Optional[FuncSpec],
        objective_list: Optional[List[FuncSpec]],
        single_constraint: Optional[FuncSpec],
        constraint_list: Optional[List[FuncSpec]],
    ) -> None:
        has_combined = (function_list is not None) or (single_function is not None)
        has_split = any(
            x is not None for x in (objective_list, single_objective, constraint_list, single_constraint)
        )
        if has_combined and has_split:
            raise ValueError(
                "Provide either combined functions ('function_list'/'single_function') "
                "OR split objectives/constraints, but not both."
            )

    def _to_list(self, item: Optional[Union[FuncSpec, List[FuncSpec]]]) -> List[FuncSpec]:
        if item is None:
            return []
        return item if isinstance(item, list) else [item]

    def _wrap_in_dict(self, item: FuncSpec, default_type: str) -> Dict:
        if isinstance(item, dict):
            if "function" not in item:
                raise ValueError("Dictionary func spec must include key 'function'.")
            item = dict(item)  # shallow copy
            item.setdefault("output_dim", 1)
            item.setdefault("type", [default_type] * int(item["output_dim"]))
            if len(item["type"]) != int(item["output_dim"]):
                raise ValueError(
                    f"Length of 'type' ({len(item['type'])}) must match 'output_dim' ({item['output_dim']})."
                )
        else:
            item = {"function": item, "output_dim": 1, "type": [default_type]}
        return item

    def _handle_constraint_bounds(
        self,
        func_dict: Dict,
        default_bounds: Optional[List[Bounds]],
        bounds_index: int,
    ) -> Tuple[Dict, int]:
        if "bounds" in func_dict:
            return func_dict, bounds_index

        if default_bounds is None:
            raise ValueError("Constraint function must have 'bounds' (or provide 'constraint_bounds').")

        d = int(func_dict["output_dim"])
        if d == 1:
            if bounds_index >= len(default_bounds):
                raise ValueError("Not enough entries in 'constraint_bounds'.")
            b = default_bounds[bounds_index]
            if b is not None and (not isinstance(b, (tuple, list)) or len(b) != 2):
                raise ValueError("Each bounds entry must be a (lb, ub) tuple or None.")
            func_dict["bounds"] = b
            bounds_index += 1
        else:
            if bounds_index + d > len(default_bounds):
                raise ValueError("Not enough entries in 'constraint_bounds' for multi-output constraint.")
            bs = default_bounds[bounds_index : bounds_index + d]
            for b in bs:
                if b is not None and (not isinstance(b, (tuple, list)) or len(b) != 2):
                    raise ValueError("Each bounds entry must be a (lb, ub) tuple or None.")
            func_dict["bounds"] = bs
            bounds_index += d

        return func_dict, bounds_index

    def _setup_functions(
        self,
        func_list: Optional[List[FuncSpec]],
        single_func: Optional[FuncSpec],
        default_type: str,
        default_bounds: Optional[List[Bounds]] = None,
    ) -> None:
        funcs = self._to_list(func_list) + self._to_list(single_func)
        bounds_index = 0
        for f in funcs:
            func_dict = self._wrap_in_dict(f, default_type)
            if default_type == "constraint":
                func_dict, bounds_index = self._handle_constraint_bounds(func_dict, default_bounds, bounds_index)
            self.functions.append(func_dict)

    def _set_output_dim(self) -> None:
        self.output_dim: int = int(sum(int(func["output_dim"]) for func in self.functions))
        
    # ------------------------------ dunder & info ------------------------------
    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        lines: list[str] = []
        lines.append("Computer Experiment object")
        lines.append(f"  Input Dimension     : {self.input_dim}")

        if self._normalize_input:
            lines.append(f"  Input Box (original): {self._format_box(self.input_box_org)}")
            lines.append("  Normalized Inputs   : True")
            lines.append(f"  Input Box (current) : {self._format_box(self.input_box)}")
        else:
            lines.append(f"  Input Box           : {self._format_box(self.input_box_org)}")
            lines.append("  Normalized Inputs   : False")

        lines.append(f"  Output Dimension    : {self.output_dim}\n")

        for i, func in enumerate(self.functions, 1):
            lines.append(f"  ** Function {i} **")
            lines.append(f"       Type           : {func['type']}")
            fn = func["function"]
            lines.append(f"       Function       : {fn.__name__ if callable(fn) else fn}")
            lines.append(f"       Output Dim     : {func['output_dim']}")
            if "bounds" in func:
                lines.append(f"       Bounds         : {func['bounds']}")
            if "simulated_noise_variance" in func:
                lines.append(f"       Noise Var      : {func['simulated_noise_variance']}")
        return "\n".join(lines)

    def _format_box(self, box: np.ndarray) -> str:
        return "[" + ", ".join(f"({lo:g}, {hi:g})" for lo, hi in zip(box[0], box[1])) + "]"

    # ------------------------------ configuration ------------------------------

    @property
    def normalize_input(self) -> bool:
        """Whether inputs are normalized to [0,1]^d before evaluation."""
        return self._normalize_input

    @normalize_input.setter
    def normalize_input(self, flag: bool) -> None:
        """Enable or disable input normalization. Clears evaluation cache."""
        self._normalize_input = bool(flag)
        self.input_box = (
            gnp.array([[0.0] * self.input_dim, [1.0] * self.input_dim])
            if self._normalize_input
            else self.input_box_org
        )
        self._last_x = None
        self._last_result = None

    # ------------------------------ metadata APIs ------------------------------

    def get_output_types(self) -> List[str]:
        """List of output types aligned with columns of `eval` result."""
        return [t for f in self.functions for t in f["type"]]

    def get_output_bounds(self) -> List[Bounds]:
        """List of (lb, ub) or None per output, aligned with columns of `eval` result."""
        out: List[Bounds] = []
        for f in self.functions:
            d = int(f["output_dim"])
            b = f.get("bounds", None)
            if d == 1:
                out.append(b if isinstance(b, (tuple, list)) else None)
            else:
                if b is None:
                    out.extend([None] * d)
                else:
                    out.extend(list(b))
        return out

    def get_constraint_bounds(self) -> np.ndarray:
        """Array of bounds (or None) for outputs whose type == 'constraint'."""
        types = self.get_output_types()
        bounds = self.get_output_bounds()
        return np.asarray([b for t, b in zip(types, bounds) if t == "constraint"], dtype=object)

    # --------------------------------- eval APIs --------------------------------

    def __call__(self, x: ArrayLike, *, normalize_input: Optional[bool] = None) -> np.ndarray:
        return self.eval(x, normalize_input=normalize_input)

    def eval(self, x: ArrayLike, *, normalize_input: Optional[bool] = None) -> np.ndarray:
        """
        Evaluate all outputs at x.

        Parameters
        ----------
        x : array-like
            Shape (n, d) or (d,). If 1D, it is reshaped to (1, d).
        normalize_input : Optional[bool]
            Per-call override. None = use object-level setting.

        Returns
        -------
        ndarray
            Shape (n, output_dim).
        """
        X = np.asarray(x, dtype=float)
        if X.ndim == 1:
            if X.size != self.input_dim:
                raise ValueError(f"Expected vector of length {self.input_dim}, got {X.size}.")
            X = X.reshape(1, -1)
        elif X.ndim == 2 and X.shape[1] == self.input_dim:
            pass
        else:
            raise ValueError(f"Input x must be (n, {self.input_dim}) or ({self.input_dim},), got {X.shape}.")

        use_norm = self.normalize_input if normalize_input is None else bool(normalize_input)

        # cache key includes normalization flag and the exact X values
        x_tuple = tuple(map(tuple, X))
        cache_key = (use_norm, x_tuple)

        if self._last_x is not None and self._last_x == cache_key:
            return self._last_result  # type: ignore[return-value]

        X_eval = X
        if use_norm:
            lower = np.asarray(self.input_box_org[0], dtype=float)
            upper = np.asarray(self.input_box_org[1], dtype=float)
            X_eval = lower + (upper - lower) * X

        result = self._eval_functions(self.functions, X_eval)
        self._last_x = cache_key
        self._last_result = result
        return result

    def eval_objectives(self, x: ArrayLike, *, normalize_input: Optional[bool] = None) -> np.ndarray:
        z = self.eval(x, normalize_input=normalize_input)
        idx = [i for i, t in enumerate(self.get_output_types()) if t == "objective"]
        return z[:, idx] if idx else np.zeros((z.shape[0], 0))

    def eval_constraints(self, x: ArrayLike, *, normalize_input: Optional[bool] = None) -> np.ndarray:
        z = self.eval(x, normalize_input=normalize_input)
        idx = [i for i, t in enumerate(self.get_output_types()) if t == "constraint"]
        return z[:, idx] if idx else np.zeros((z.shape[0], 0))

    # -------------------------- internal function eval --------------------------

    def _eval_functions(self, function_dicts: List[Dict], X: np.ndarray) -> np.ndarray:
        """Evaluate functions and concatenate outputs (n, p_total)."""
        if X.ndim != 2 or X.shape[1] != self.input_dim:
            raise ValueError(f"X must be shape (n, {self.input_dim}), got {X.shape}.")

        outputs: List[np.ndarray] = []
        for func in function_dicts:
            f = func["function"]
            d = int(func["output_dim"])
            z = f(X)  # expected shapes: (n,) or (n, d)
            z = np.asarray(z, dtype=float)

            if z.ndim == 1:
                z = z[:, np.newaxis]
            elif z.ndim == 2 and z.shape[1] != d:
                raise ValueError(
                    f"The function output dimension {z.shape[1]} does not match the expected output dimension {d}."
                )
            elif z.ndim > 2:
                raise ValueError(f"Function output must be 1D or 2D, got shape {z.shape}.")

            outputs.append(z)

        return np.concatenate(outputs, axis=1) if outputs else np.empty((X.shape[0], 0), dtype=float)


##############################################################################
#                                                                            #
#                 StochasticComputerExperiment Class                         #
#                                                                            #
##############################################################################


class StochasticComputerExperiment(ComputerExperiment):
    """
    Extension of `ComputerExperiment` that simulates additive Gaussian noise on outputs.

    Noise configuration:
      - Provide a global `simulated_noise_variance` (scalar or list with length = total outputs),
        or
      - Provide per-function dictionaries with "simulated_noise_variance" (list length = that function's output_dim),
        or
      - Omit / set zeros for noise-free outputs.

    Methods differ from base by adding stochastic evaluation with optional batching.

    Notes
    -----
    - `eval_objectives` / `eval_constraints` are not supported here because repeated
      calls would mix stochastic draws and caching is disabled for stochastic evals.
    """

    def __init__(
        self,
        input_dim: int,
        input_box: ArrayLike,
        *,
        single_function: Optional[FuncSpec] = None,
        function_list: Optional[List[FuncSpec]] = None,
        single_objective: Optional[FuncSpec] = None,
        objective_list: Optional[List[FuncSpec]] = None,
        single_constraint: Optional[FuncSpec] = None,
        constraint_list: Optional[List[FuncSpec]] = None,
        simulated_noise_variance: Optional[Union[float, List[float]]] = None,
    ) -> None:
        super().__init__(
            input_dim=input_dim,
            input_box=input_box,
            single_function=single_function,
            function_list=function_list,
            single_objective=single_objective,
            objective_list=objective_list,
            single_constraint=single_constraint,
            constraint_list=constraint_list,
        )
        self._initialize_noise_variance(self.functions, simulated_noise_variance)

    # ------------------------------- noise config -------------------------------

    def _initialize_noise_variance(
        self, function_dicts: List[Dict], simulated_noise_variance: Optional[Union[float, List[float]]]
    ) -> None:
        # Expand scalar to per-output list
        if np.isscalar(simulated_noise_variance):
            total_dim = sum(int(f["output_dim"]) for f in function_dicts)
            simulated_noise_variance = [float(simulated_noise_variance)] * total_dim

        if simulated_noise_variance is not None:
            total_dim = sum(int(f["output_dim"]) for f in function_dicts)
            if len(simulated_noise_variance) != total_dim:
                raise ValueError(
                    f"Length of 'simulated_noise_variance' ({len(simulated_noise_variance)}) "
                    f"must equal total output_dim ({total_dim})."
                )
            start = 0
            for f in function_dicts:
                d = int(f["output_dim"])
                f["simulated_noise_variance"] = list(np.asarray(simulated_noise_variance[start : start + d], dtype=float))
                start += d
        else:
            for f in function_dicts:
                if "simulated_noise_variance" not in f:
                    f["simulated_noise_variance"] = [0.0] * int(f["output_dim"])

    @property
    def simulated_noise_variance(self) -> np.ndarray:
        return self.get_simulated_noise_variances()

    def get_simulated_noise_variances(self) -> np.ndarray:
        return np.concatenate([np.asarray(func["simulated_noise_variance"], dtype=float) for func in self.functions])

    # ------------------------------ dunder & info ------------------------------
    def __str__(self) -> str:
        lines: list[str] = []
        lines.append("Stochastic Computer Experiment")
        lines.append(f"  Input Dimension     : {self.input_dim}")

        if self._normalize_input:
            lines.append(f"  Input Box (original): {self._format_box(self.input_box_org)}")
            lines.append("  Normalized Inputs   : True")
            lines.append(f"  Input Box (current) : {self._format_box(self.input_box)}")
        else:
            lines.append(f"  Input Box           : {self._format_box(self.input_box_org)}")
            lines.append("  Normalized Inputs   : False")

        lines.append(f"  Output Dimension    : {self.output_dim}\n")

        for i, func in enumerate(self.functions, 1):
            lines.append(f"  ** Function {i} **")
            lines.append(f"       Type           : {func['type']}")
            fn = func["function"]
            lines.append(f"       Function       : {fn.__name__ if callable(fn) else fn}")
            lines.append(f"       Output Dim     : {func['output_dim']}")
            lines.append(f"       Noise Var      : {func['simulated_noise_variance']}")
            if "bounds" in func:
                lines.append(f"       Bounds         : {func['bounds']}")
        return "\n".join(lines)
    
    # --------------------------------- eval APIs --------------------------------

    def __call__(
        self,
        x: ArrayLike,
        *,
        simulate_noise: bool = True,
        batch_size: int = 1,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        return self.eval(x, simulate_noise=simulate_noise, batch_size=batch_size, rng=rng)

    def eval(
        self,
        x: ArrayLike,
        *,
        simulate_noise: bool = True,
        batch_size: int = 1,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        """
        Evaluate all outputs at x with optional simulated Gaussian noise.

        Parameters
        ----------
        x : array-like
            Shape (n, d) or (d,). If 1D, it is reshaped to (1, d).
        simulate_noise : bool, default True
            Whether to add Gaussian noise using configured variances.
        batch_size : int, default 1
            If >1, returns shape (n, p, batch_size). If 1, returns (n, p).
        rng : numpy.random.Generator, optional
            Random generator for reproducibility. If None, a default RNG is used.

        Returns
        -------
        ndarray
            (n, p) if batch_size == 1; else (n, p, batch_size).
        """
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")

        # Input validation & normalization (no caching for stochastic evals)
        X = np.asarray(x, dtype=float)
        if X.ndim == 1:
            if X.size != self.input_dim:
                raise ValueError(f"Expected vector of length {self.input_dim}, got {X.size}.")
            X = X.reshape(1, -1)
        elif X.ndim == 2 and X.shape[1] == self.input_dim:
            pass
        else:
            raise ValueError(f"Input x must be (n, {self.input_dim}) or ({self.input_dim},), got {X.shape}.")

        if self.normalize_input:
            lower = np.asarray(self.input_box_org[0], dtype=float)
            upper = np.asarray(self.input_box_org[1], dtype=float)
            X = lower + (upper - lower) * X

        rng = rng if rng is not None else np.random.default_rng()

        draws = [self._eval_functions_with_noise(self.functions, X, simulate_noise, rng) for _ in range(batch_size)]
        return draws[0] if batch_size == 1 else np.dstack(draws)

    def eval_objectives(self, x: ArrayLike, **kwargs) -> np.ndarray:  # type: ignore[override]
        raise NotImplementedError("eval_objectives is not supported in StochasticComputerExperiment.")

    def eval_constraints(self, x: ArrayLike, **kwargs) -> np.ndarray:  # type: ignore[override]
        raise NotImplementedError("eval_constraints is not supported in StochasticComputerExperiment.")

    # -------------------------- internal function eval --------------------------

    def _eval_functions_with_noise(
        self,
        function_dicts: List[Dict],
        X: np.ndarray,
        simulate_noise: bool,
        rng: np.random.Generator,
    ) -> np.ndarray:
        if X.ndim != 2 or X.shape[1] != self.input_dim:
            raise ValueError(f"X must be shape (n, {self.input_dim}), got {X.shape}.")

        outs: List[np.ndarray] = []
        for func in function_dicts:
            f = func["function"]
            d = int(func["output_dim"])
            z = f(X)
            z = np.asarray(z, dtype=float)

            if z.ndim == 1:
                z = z[:, np.newaxis]
            elif z.ndim == 2 and z.shape[1] != d:
                raise ValueError(f"The function output dimension {z.shape[1]} does not match the expected {d}.")
            elif z.ndim > 2:
                raise ValueError(f"Function output must be 1D or 2D, got shape {z.shape}.")

            if simulate_noise:
                var = np.asarray(func.get("simulated_noise_variance", [0.0] * d), dtype=float)
                if var.shape != (d,):
                    raise ValueError(f"'simulated_noise_variance' must have length {d}.")
                mask = var > 0.0
                if np.any(mask):
                    std = np.sqrt(var[mask])
                    z[:, mask] = z[:, mask] + rng.normal(0.0, std, size=(z.shape[0], mask.sum()))

            outs.append(z)

        return np.hstack(outs) if outs else np.empty((X.shape[0], 0), dtype=float)
