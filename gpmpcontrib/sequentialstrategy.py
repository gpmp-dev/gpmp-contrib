# --------------------------------------------------------------
# Authors: Emmanuel Vazquez <emmanuel.vazquez@centralesupelec.fr>
# Copyright (c) 2023-2025, CentraleSupelec
# License: GPLv3 (see LICENSE)
# --------------------------------------------------------------
import time
import pickle
import gpmp.num as gnp
from gpmpcontrib import SequentialPrediction, SMC


class SequentialStrategy(SequentialPrediction):
    """Sequential decision-making via a sampling criterion.

    This abstract class extends SequentialPrediction to incorporate a
    sampling criterion for sequential decision-making. It selects new points
    based on the criterion and updates the predictive model accordingly.

    Parameters
    ----------
    problem : object
        Problem instance with the objective function.
    model : object
        Predictive model.
    options : dict, optional
        Strategy configuration. Options may include:
            maximize_criterion : bool, default True.
            update_model_at_init : bool, default True.
            update_estimate_at_init : bool, default True.
            update_search_space_at_init : bool, default False.

    Attributes
    ----------
    computer_experiments_problem : object
        Problem instance.
    options : dict
        Strategy options.
    nt : int
        Number of candidates / search points.
    xt : ndarray
        Candidate points.
    zpm, zpv : ndarray or None
        Posterior mean and variance on candidate points.
    current_estimate : None
        Current best estimate.
    sampling_criterion_values : array_like
        Sampling criterion values.
    n_iter : int
        Iteration counter.
    exec_times : dict
        Timing metrics.
    history : dict
        Evaluation history with keys:
            - "estimates": recorded estimates,
            - "criterion_best": best criterion value per iteration,
            - "model_params": snapshots of model parameters.
    maximize_criterion : bool
        If True, the criterion is maximized; if False, minimized.
    """

    def __init__(self, problem, model, options=None):
        super().__init__(model=model)  # Ensure proper parent class initialization
        self.computer_experiments_problem = problem
        self.options = self.set_options(options or {})
        self.xt = self.set_initial_xt()
        self.nt = self.xt.shape[0] if self.xt is not None else None
        self.zpm = None
        self.zpv = None
        self.current_estimate = None
        self.sampling_criterion_values = None
        self.n_iter = 0
        self.exec_times = {}
        self.history = {"estimates": [], "criterion_best": [], "model_params": []}
        self.maximize_criterion = self.options.get("maximize_criterion", True)

    def set_options(self, options):
        default_options = {
            "update_model_at_init": True,
            "update_predictions_at_init": True,
            "update_estimate_at_init": True,
            "update_search_space_at_init": False,
            "maximize_criterion": True,
        }
        default_options.update(options or {})
        return default_options

    def set_initial_xt(self):
        raise NotImplementedError("get_initial_xt must be implemented.")

    def set_initial_design(
        self,
        xi,
        update_model=True,
        update_predictions=True,
        update_estimate=True,
        update_search_space=False,
    ):
        update_model = self.options.get("update_model_at_init", update_model)
        update_estimate = self.options.get("update_estimate_at_init", update_estimate)
        update_search_space = self.options.get(
            "update_search_space_at_init", update_search_space
        )
        tic = time.time()
        zi = self.computer_experiments_problem.eval(xi)
        if update_model:
            super().set_data_with_model_selection(xi, zi)
        else:
            super().set_data(xi, zi)
        if update_predictions:
            self.update_predictions()
        if update_estimate:
            self.current_estimate = self.update_current_estimate()
        if update_search_space:
            self.update_search_space()
        self.exec_times["initial_design"] = time.time() - tic

    def update_predictions(self):
        tic = time.time()
        self.predict(self.xt, convert_out=False)
        self.exec_times["update_predictions"] = time.time() - tic

    def update_current_estimate(self, *args, **kwargs):
        raise NotImplementedError("update_current_estimate must be implemented.")

    def update_search_space(self, *args, **kwargs):
        raise NotImplementedError("update_search_space must be implemented.")

    def sampling_criterion(self, x, zpm, zpv, *args, **kwargs):
        raise NotImplementedError("sampling_criterion must be implemented.")

    def update_sampling_criterion_values(self, x, zpm, zpv):
        self.sampling_criterion_values = self.sampling_criterion(x, zpm, zpv)

    def select_best_index(self, criterion_values):
        arr = gnp.asarray(criterion_values)
        return gnp.argmax(arr) if self.maximize_criterion else gnp.argmin(arr)

    def make_new_eval(self, xnew, update_model=True):
        tic = time.time()
        znew = self.computer_experiments_problem.eval(xnew)
        self.exec_times["new_eval"] = time.time() - tic
        tic = time.time()
        if update_model:
            self.set_new_eval_with_model_selection(xnew, znew)
        else:
            self.set_new_eval(xnew, znew)
        self.exec_times["update_model"] = time.time() - tic

    def get_model_params(self):
        """Return a snapshot of model parameters from the ModelContainer."""
        params = []
        for m in self.model.models:
            mp = m["model"].meanparam
            cp = m["model"].covparam
            # Store copies to avoid later modifications
            params.append(
                {
                    "meanparam": gnp.copy(mp) if mp is not None else None,
                    "covparam": gnp.copy(cp) if cp is not None else None,
                }
            )
        return params

    def get_model_state(self):
        """Return the  pickleable model state."""
        return self.model.get_state()

    def step(self):
        raise NotImplementedError("step must be implemented in the subclass.")

    def save_state(self, filename="state.pkl"):
        state = {
            "n_iter": self.n_iter,
            "history": self.history,
            "exec_times": self.exec_times,
            "current_estimate": self.current_estimate,
        }
        with open(filename, "wb") as f:
            pickle.dump(state, f)

    def load_state(self, filename="state.pkl"):
        with open(filename, "rb") as f:
            state = pickle.load(f)
        self.n_iter = state.get("n_iter", 0)
        self.history = state.get("history", {})
        self.exec_times = state.get("exec_times", {})
        self.current_estimate = state.get("current_estimate", None)


class SequentialStrategyGridSearch(SequentialStrategy):
    """Sequential strategy using a fixed candidate set / point collection (PC).

    Parameters
    ----------
    problem : object
        Problem instance.
    model : object
        Predictive model.
    xt : array_like
        Fixed candidate points.
    options : dict, optional
        Additional options. Defaults for PC:
            update_model_at_init = True,
            update_estimate_at_init = True,
            update_search_space_at_init = False.

    Attributes
    ----------
    """

    def __init__(self, problem, model, xt, options=None):
        if options is None:
            options = {}
        options.setdefault("update_model_at_init", True)
        options.setdefault("update_estimate_at_init", True)
        options.setdefault("update_predictions_at_init", True)
        options.setdefault("update_search_space_at_init", False)
        self._xt = xt  # Store xt in a temporary attribute for set_initial_xt()
        super().__init__(problem, model, options)

    def set_initial_xt(self):
        return self._xt

    def step(self):
        step_start = time.time()
        self.update_sampling_criterion_values(self.xt, self.zpm, self.zpv)
        best_idx = self.select_best_index(self.sampling_criterion_values)
        x_new = self.xt[best_idx].reshape(1, -1)
        self.make_new_eval(x_new, update_model=True)
        self.update_predictions()
        self.current_estimate = self.update_current_estimate()
        self.n_iter += 1
        self.history.setdefault("eval_indices", []).append(int(best_idx))
        self.history.setdefault("criterion_best", []).append(
            float(self.sampling_criterion_values[best_idx])
        )
        # Save a snapshot of the model parameters.
        self.history.setdefault("model_params", []).append(self.get_model_params())
        self.exec_times["step"] = time.time() - step_start


# ==============================================================================
# Sequential Strategy with SMC for Search Space Adaptation
# ==============================================================================


class SequentialStrategySMC(SequentialStrategy):
    """Sequential strategy using SMC to adapt the search space.

    This class augments the base strategy with SMC functionalities. It updates
    the search space using an SMC sampler.

    Parameters
    ----------
    problem : object
        Problem instance.
    model : object
        Predictive model.
    options : dict, optional
        Strategy configuration. Options for SMC default to:
            n_smc = 1000,
            update_model_at_init = True,
            update_estimate_at_init = True,
            update_search_space_at_init = True.

    Attributes
    ----------
    smc : SMC
        SMC instance for adapting the search space.
    smc_log_density_param_initial : any
        Initial parameter for SMC target density.
    smc_log_density_param : any
        Current parameter for SMC target density.
    """

    def __init__(self, problem, model, options=None):
        if options is None:
            options = {}
        options.setdefault("n_smc", 1000)
        options.setdefault("initial_distribution", "randunif")
        options.setdefault("update_model_at_init", True)
        options.setdefault("update_predictions_at_init", True)
        options.setdefault("update_estimate_at_init", True)
        options.setdefault("update_search_space_at_init", True)
        self.computer_experiments_problem = problem
        # Merge SMC-specific options with sequentialstrategy defaults.
        self.options = self.set_options(options)
        self.smc = self.init_smc(
            self.computer_experiments_problem.input_box,
            self.options["n_smc"],
            self.options["initial_distribution"],
        )
        self.smc_log_density_param_initial = None
        self.smc_log_density_param = None
        super().__init__(problem, model, options)

    def init_smc(self, box, n_smc, initial_distribution):
        """Initialize the SMC instance using the problem's input box.

        Notes
        -----
        At initialization, particles are distributed according to the initial distribution
        """
        return SMC(
            box,
            n_smc,
            initial_distribution=initial_distribution,
        )

    def set_initial_xt(self):
        return self.smc.particles.x

    def smc_log_density(self, *args, **kwargs):
        raise NotImplementedError("smc_log_density must be implemented.")

    def update_smc_target_log_density_param(self):
        raise NotImplementedError("update_smc_log_density_param must be implemented.")

    def update_search_space(self, method="step_with_possible_restart"):
        """Update the SMC search space using the specified method."""
        self.update_smc_target_log_density_param()
        if method == "simple_step":
            self.smc.step(
                logpdf_parameterized_function=self.smc_log_density,
                logpdf_param=self.smc_log_density_param,
            )
        elif method == "restart":
            self.smc.restart(
                logpdf_parameterized_function=self.smc_log_density,
                logpdf_initial_param=self.smc_log_density_param_initial,
                logpdf_final_param=self.smc_log_density_param,
                p0=0.8,
                debug=True,
            )
        elif method == "step_with_possible_restart":
            self.smc.step_with_possible_restart(
                logpdf_parameterized_function=self.smc_log_density,
                initial_logpdf_param=self.smc_log_density_param_initial,
                target_logpdf_param=self.smc_log_density_param,
                min_ess_ratio=0.6,
                p0=0.6,
                debug=False,
            )

        self.xt = self.smc.particles.x

    def step(self):
        step_start = time.time()
        # Evaluate the sampling criterion on SMC particles.
        self.update_sampling_criterion_values(self.smc.particles.x, self.zpm, self.zpv)
        best_idx = self.select_best_index(self.sampling_criterion_values)
        x_new = self.xt[best_idx].reshape(1, -1)
        self.make_new_eval(x_new, update_model=True)
        # Update current predictions & estimate for SMC
        self.update_predictions()
        self.current_estimate = self.update_current_estimate()
        # Update the SMC search space.
        self.update_search_space()
        # Update predictions & current estimate at the end of the step
        self.update_predictions()
        self.current_estimate = self.update_current_estimate()

        self.n_iter += 1
        self.history.setdefault("eval_indices", []).append(int(best_idx))
        self.history.setdefault("criterion_best", []).append(
            gnp.to_scalar(self.sampling_criterion_values[best_idx])
        )
        self.history.setdefault("model_params", []).append(self.get_model_params())
        self.exec_times["step"] = time.time() - step_start
