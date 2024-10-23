# --------------------------------------------------------------
# Authors: Emmanuel Vazquez <emmanuel.vazquez@centralesupelec.fr>
# Copyright (c) 2023-2024, CentraleSupelec
# License: GPLv3 (see LICENSE)
# --------------------------------------------------------------
import gpmp.num as gnp
from gpmpcontrib import SequentialPrediction, SMC


class SequentialStrategy(SequentialPrediction):
    """Implements sequential decision-making by maximizing a sampling criterion.

    This class extends SequentialPrediction to use a sampling
    criterion and adapts the search space using a Sequential Monte
    Carlo (SMC) method using a target density.

    Parameters
    ----------
    problem : object
        Instance of the problem containing the function to be optimized.
    model : object
        Predictive model for making predictions.
    options : dict, optional
        Configuration options for SMC and other processes.

    Attributes
    ----------
    computer_experiments_problem : object
        Optimization problem instance.
    options : dict
        Configuration options.
    current_estimate : float
        Best current estimate based on sampling criterion.
    smc : SMC
        Sequential Monte Carlo instance for search space.
    sampling_criterion_values : gnp array
        Array holding the calculated values of the sampling criterion for each particle
        within the SMC's search space. These values are used to guide the selection of
        new points for evaluation, aiming to maximize the information gain during the
        sequential decision-making process.

    Notes
    -----
    This class is derived from the SequentialPrediction class, which
    handles data and a (single- or multi-output) Model object for
    sequential predictions. It also relies on the a ComputerExperiment
    object to specify a set of functions (objectives, contraints...)
    to be evaluated, and an SMC object to sequentiallly adjust the
    search space for the optimization of the sampling criterion (aka
    acquisition function).

    The class requires implementation of the sampling criterion and
    a target density for the SMC, as well as a procedure to estimate the
    quantity of interest.

    """

    def __init__(self, problem, model, options=None):
        # computer experiments problem
        self.computer_experiments_problem = problem

        # model initialization
        super().__init__(model=model)

        # options
        self.options = self.set_options(options)

        # Best current value
        self.current_estimate = None

        # search space
        self.smc = self.init_smc(self.options["n_smc"])
        self.smc_log_density_param_initial = None
        self.smc_log_density_param = None

        # Sampling criterion
        self.sampling_criterion_values = gnp.empty(self.options["n_smc"])

        # Current estimate of the quantity of interest
        self.current_estimate = None

    def set_options(self, options):
        """
        Sets configuration options with defaults.

        Parameters
        ----------
        options : dict
            User-provided configuration options.

        Returns
        -------
        dict
            Options with defaults filled in.
        """
        default_options = {"n_smc": 1000}
        default_options.update(options or {})
        return default_options

    def set_initial_design(self, xi, update_model=True, update_search_space=True):
        """
        Sets initial design and optionally updates the model and search space.

        Parameters
        ----------
        xi : array_like
            Initial design points.
        update_model : bool, optional
            If True, update model with initial design. Default is True.
        update_search_space : bool, optional
            If True, update search space based on initial design. Default is True.
        """
        zi = self.computer_experiments_problem.eval(xi)

        if update_model:
            super().set_data_with_model_selection(xi, zi)
        else:
            super().set_data(xi, zi)

        self.current_estimate = self.update_current_estimate()

        if update_search_space:
            self.update_search_space()

    def init_smc(self, n_smc):
        """Initializes the SMC process for the search space.

        Parameters
        ----------
        n_smc : int
            The number of particles to use in the SMC process.

        Returns
        -------
        SMC
            An instance of the SMC class initialized with the problem's input box and particle count.
        """
        return SMC(self.computer_experiments_problem.input_box, n_smc)

    def smc_log_density(self, *args, **kwargs):
        """
        Defines target log density for SMC. Override this method.

        Raises
        ------
        NotImplementedError
            If method is not implemented.
        """
        raise NotImplementedError(
            "Target density method for SMC must be implemented.")

    def update_smc_log_density_param(self):
        """
        Updates the parameter of the target log density for SMC. Override this method.

        Raises
        ------
        NotImplementedError
            If method is not implemented.
        """
        raise NotImplementedError(
            "Update target density parameter method must be implemented."
        )

    def update_search_space(self, method="step_with_possible_restart"):

        self.update_smc_log_density_param()

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

    def update_current_estimate(self, *args, **kwargs):
        """
        Computes current best estimate. Override this method.

        Raises
        ------
        NotImplementedError
            If method is not implemented.
        """
        raise NotImplementedError(
            "compute_current_estimate method must be implemented."
        )

    def sampling_criterion(self, x, *args, **kwargs):
        """
        Computes sampling criterion values at points x. Override this method.

        Parameters
        ----------
        x : array_like
            Points to evaluate the sampling criterion.

        Returns
        -------
        array_like
            Sampling criterion values.

        Raises
        ------
        NotImplementedError
            If method is not implemented.
        """
        raise NotImplementedError(
            "Sampling criterion method must be implemented.")

    def update_sampling_criterion_values(self, x, zpm, zpv):
        """Updates the sampling criterion values based on current predictions."""
        self.sampling_criterion_values = self.sampling_criterion(x, zpm, zpv)

    def make_new_eval(self, xnew, update_model=True, update_search_space=True):
        """
        Makes a new evaluation at xnew and optionally updates the model and search space.

        Parameters
        ----------
        xnew : array_like
            New evaluation point.
        update_model : bool, optional
            If True, update model with new evaluation. Default is True.
        update_search_space : bool, optional
            If True, update search space based on new evaluation. Default is True.
        """
        znew = self.computer_experiments_problem.eval(xnew)

        if update_model:
            self.set_new_eval_with_model_selection(xnew, znew)
        else:
            self.set_new_eval(xnew, znew)

        self.current_estimate = self.update_current_estimate()

        if update_search_space:
            self.update_search_space()

    def step(self):
        """
        Performs a step in the sequential decision-making process by evaluating the
        sampling criterion, making a new evaluation, and updating the search space.
        """

        # evaluate sampling criterion on the search space
        zpm, zpv = self.predict(self.smc.particles.x, convert_out=False)

        # Update the sampling criterion values
        self.update_sampling_criterion_values(self.smc.particles.x, zpm, zpv)

        # make new evaluation
        x_new = self.smc.particles.x[
            gnp.argmax(gnp.asarray(self.sampling_criterion_values))
        ].reshape(1, -1)

        self.make_new_eval(x_new)
