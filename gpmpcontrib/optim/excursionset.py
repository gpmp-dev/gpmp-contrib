# --------------------------------------------------------------
# Authors: Emmanuel Vazquez <emmanuel.vazquez@centralesupelec.fr>
# Copyright (c) 2023-2025, CentraleSupelec
# License: GPLv3 (see LICENSE)
# --------------------------------------------------------------
import gpmp.num as gnp
import gpmp as gp
import gpmpcontrib.samplingcriteria as sampcrit
from gpmpcontrib import (
    SequentialStrategyGridSearch,
    SequentialStrategySMC,
    SequentialStrategyBSS,
)


class ExcursionSetGridSearch(SequentialStrategyGridSearch):
    """Sequential strategy for estimating the excursion set over a
    finite search space.

    This class refines the estimate of the excursion set Γ = {x ∈ X | ξ(x)> u_target},
    where ξ(x) is a Gaussian process model of the objective function.

    At each iteration, it:

      - Computes excursion (`excursion_probability`) and
        misclassification probabilities
        (`excursion_misclassification_probability`).
      - Updates key metrics:
        - `alpha`: Expected excursion volume (mean probability of exceeding `u_target`).
        - `beta`: Expected misclassification volume (mean probability of classification error).
      - Selects the next evaluation point using `excursion_wMSE`.

    Parameters
    ----------
    problem : object
        Problem instance with the objective function.
    model : object
        Predictive model (e.g., Gaussian process).
    xt : array_like, shape (N, d)
        Finite set of candidate points.
    u_target : float
        Excursion threshold.
    options : dict, optional
        Configuration options.

    Attributes
    ----------
    g : gnp.array, shape (N, 1) or None
        Excursion probabilities.
    tau : gnp.array, shape (N, 1) or None
        Misclassification probabilities.
    alpha : float or None
        Expected excursion volume.
    beta : float or None
        Expected misclassification volume.

    """

    def __init__(self, problem, model, xt, u_target, options=None):
        super().__init__(problem=problem, model=model, xt=xt, options=options)
        # target threshold
        self.u_target = u_target
        # P_n(ξ(x) > u): excursion probability at candidate points
        self.g = None
        # P_n(x ∈ Γ Δ hat(Γ)_n): Misclassification probability
        self.tau = None
        # Expected excursion volume (mean of g)
        self.alpha = None
        # Expected misclassification volume (mean of tau)
        self.beta = None

    def update_current_estimate(self):
        """
        Updates estimates.
        """
        self.g = sampcrit.excursion_probability(self.u_target, self.zpm, self.zpv)
        self.tau = sampcrit.excursion_misclassification_probability(
            self.u_target, self.zpm, self.zpv
        )
        self.alpha = gnp.mean(self.g)
        self.beta = gnp.mean(self.tau)
        print(f"alpha = {self.alpha}, beta = {self.beta}")

    def sampling_criterion(self):
        """
        Computes the selection criterion at given points.
        """
        return sampcrit.excursion_wMSE(self.u_target, self.zpm, self.zpv)


class ExcursionSetBSS(SequentialStrategyBSS):
    """Sequential strategy for excursion set estimation with Bayesian Subset Simulation (BSS).

    This strategy estimates the excursion set:
        Γ = {x ∈ X | ξ(x) > u_target}
    where ξ(x) is a Gaussian process model of the objective function.

    The search space is adapted via an SMC sampler, guided by a sequence of
    intermediate thresholds from `u_init` to `u_target`.

    Parameters
    ----------
    problem : object
        The optimization problem.
    model : object
        The predictive model (e.g., a Gaussian process).
    u_init : float
        Initial threshold for the excursion probability.
    u_target : float
        Final excursion threshold.
    options : dict, optional
        Strategy configuration.

    Attributes
    ----------
    u_target : float
        Target excursion threshold.
    u_init : float
        Initial threshold.
    u_current : float
        Current intermediate threshold.
    mu : float
        Interpolation parameter between `u_init` and `u_target`.
    g : ndarray or None
        Excursion probabilities at candidate points.
    tau : ndarray or None
        Misclassification probabilities at candidate points.
    alpha : float or None
        Expected excursion volume.
    beta : float or None
        Expected misclassification volume.
    """

    def __init__(self, problem, model, u_init, u_target, options=None):
        super().__init__(problem=problem, model=model, options=options)
        self.u_target = u_target
        self.u_init = u_init
        self.u_map_function = lambda mu: (1 - mu) * u_init + mu * u_target
        self._mu = 0.0  # Interpolation factor between u_init and u_target
        self.smc_log_density_param = self._mu
        self.u_current = self.u_map_function(self._mu)

        self.g = None  # Excursion probability at candidate points
        self.tau = None  # Misclassification probability at candidate points
        self.alpha = None  # Expected excursion volume
        self.beta = None  # Expected misclassification volume

    @property
    def mu(self):
        """Interpolation parameter between `u_init` and `u_target`."""
        return self._mu

    @mu.setter
    def mu(self, value):
        """Update `mu` and adjust the current threshold accordingly."""
        self._mu = value
        self.smc_log_density_param = self.mu
        self.u_current = self.u_map_function(value)

    def step_move_particles_with_mu(self, mu):
        self.mu = mu
        super().step_move_particles()

    def restart(self):
        self.mu = 0.0
        super().restart()

    def smc_log_density(self, x, mu):
        """Compute the log probability of an excursion for SMC sampling.

        Parameters
        ----------
        x : ndarray
            Points where the density is evaluated.
        mu : float
            Interpolation parameter defining the current threshold.

        Returns
        -------
        log_prob_excur : ndarray
            Log-probability of excursion at `x`.
        """
        min_threshold = gnp.log(1e-10)

        input_box = gnp.asarray(self.computer_experiments_problem.input_box)
        inside_box = sampcrit.isinbox(input_box, x)

        zpm, zpv = self.predict(x, convert_out=False, use_cache=True)
        u = self.u_map_function(mu)

        log_prob_excur = gnp.where(
            inside_box,
            gnp.maximum(
                min_threshold,
                sampcrit.log_excursion_probability(u, zpm, zpv),
            ).flatten(),
            -gnp.inf,
        )
        return log_prob_excur

    def update_current_estimate(self):
        """Update excursion and misclassification probabilities, as well as volume estimates."""
        self.g = sampcrit.excursion_probability(self.u_current, self.zpm, self.zpv)
        self.tau = sampcrit.excursion_misclassification_probability(
            self.u_current, self.zpm, self.zpv
        )
        self.alpha = gnp.mean(self.g)
        self.beta = gnp.mean(self.tau)
        print(f"alpha = {self.alpha:.4e}, beta = {self.beta:.4e}")

    def sampling_criterion(self):
        """Compute the sampling criterion for point selection."""
        return sampcrit.excursion_wMSE(self.u_current, self.zpm, self.zpv)
