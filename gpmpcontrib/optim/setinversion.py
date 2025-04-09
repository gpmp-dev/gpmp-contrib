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


class SetInversionGridSearch(SequentialStrategyGridSearch):
    """Sequential strategy for estimating the inverse image of a box over a
    finite search space.

    This class refines the estimate of the inversion set Γ = {x ∈ X | ξ(x) ∈ box_target},
    where ξ(x) is a Gaussian process model of the objective function.

    At each iteration, it:
      - Computes set membership probabilities
      - Updates misclassification metrics
      - Selects points using summed wMSE across output dimensions

    Parameters
    ----------
    problem : object
        Problem instance with the objective function.
    model : object
        Predictive model (e.g., Gaussian process).
    xt : array_like, shape (N, d)
        Finite set of candidate points.
    box_target : array_like, shape (2, dim_output)
        Target box bounds [lower, upper] for each dimension.
    options : dict, optional
        Configuration options including:
        - normalization: scaling factors for variance (default 1.0)
        - alpha: misclassification exponent (default 1.0)
        - beta: variance exponent (default 0.5)
    """

    def __init__(self, problem, model, xt, box_target, options=None):
        super().__init__(problem=problem, model=model, xt=xt, options=options)
        self.box_target = gnp.asarray(box_target)
        self.g_prod = None  # (N, 1)
        self.g = None  # (N, dim_output)
        self.tau_prod = None  # (N, 1)
        self.tau = None  # (N, dim_output)
        self.alpha_metric = None  # Expected volume
        self.beta_metric = None  # Expected misclassification

    def update_current_estimate(self):
        """Updates probability estimates and metrics."""
        self.sum_log_g, self.log_g = sampcrit.box_logprobability(
            self.box_target, self.zpm, self.zpv
        )
        self.sum_log_tau, self.log_tau = sampcrit.box_misclassification_logprobability(
            self.box_target, self.zpm, self.zpv
        )
        self.g_prod = gnp.exp(self.sum_log_g)
        self.alpha_metric = gnp.mean(self.g_prod)
        self.beta_metric = gnp.mean(gnp.exp(self.sum_log_tau))
        print(
            f"Expected volume (alpha): {self.alpha_metric:.4f}, "
            f"Misclassification (beta): {self.beta_metric:.4f}"
        )

    def sampling_criterion(self):
        """Computes the summed wMSE across output dimensions for point selection.

        Returns
        -------
        gnp.array, shape (N, 1)
            Summed wMSE values for each candidate point
        """
        wmse_sum, _ = sampcrit.box_wMSE(
            self.box_target,
            self.zpm,
            self.zpv,
            normalization=self.options.get("normalization", 1.0),
            alpha=self.options.get("alpha", 1.0),
            beta=self.options.get("beta", 0.5),
        )
        return wmse_sum


class SetInversionBSS(SequentialStrategyBSS):
    """Sequential strategy for box set inversion with Bayesian Subset Simulation (BSS).

    Estimates the inversion set:
        Γ = {x ∈ X | ξ(x) ∈ box_target}
    where ξ(x) is a Gaussian process model and box_target defines multi-dimensional bounds.

    The search space is adapted via an SMC sampler, guided by interpolation between
    initial and target boxes.

    Parameters
    ----------
    problem : object
        The optimization problem.
    model : object
        The predictive model (e.g., Gaussian process).
    box_init : array_like, shape (2, dim_output)
        Initial box bounds [lower, upper] for warm-up.
    box_target : array_like, shape (2, dim_output)
        Final target box bounds.
    options : dict, optional
        Strategy configuration including:
        - normalization: variance scaling factors (default 1.0)
        - alpha: misclassification exponent (default 1.0)
        - beta: variance exponent (default 0.5)

    Attributes
    ----------
    box_target : gnp.array
        Target box bounds.
    box_current : gnp.array
        Current intermediate box.
    mu : float
        Interpolation parameter between boxes.
    g_prod : gnp.array
        Product of probabilities across dimensions.
    tau_prod : gnp.array
        Product of misclassification probabilities.
    alpha : float
        Expected volume.
    beta : float
        Expected misclassification volume.
    """

    def __init__(self, problem, model, box_init, box_target, options=None):
        super().__init__(problem=problem, model=model, options=options)
        self.box_target = gnp.asarray(box_target)
        self.box_init = gnp.asarray(box_init)
        self._mu = 0.0  # Interpolation factor
        self.smc_log_density_param = self._mu
        self.box_current = self._interpolate_box(self._mu)

        # Probability tracking
        self.g_prod = None    # (n, 1)
        self.g = None         # (n, dim_output)
        self.tau_prod = None  # (n, 1)
        self.tau = None       # (n, dim_output)
        self.alpha = None
        self.beta = None

    def _interpolate_box(self, mu):
        """Interpolate between initial and target boxes."""
        return (1 - mu) * self.box_init + mu * self.box_target

    @property
    def mu(self):
        return self._mu

    @mu.setter
    def mu(self, value):
        self._mu = value
        self.smc_log_density_param = self._mu
        self.box_current = self._interpolate_box(value)

    def step_move_particles_with_mu(self, mu):
        self.mu = mu
        super().step_move_particles()

    def restart(self):
        self.mu = 0.0
        super().restart()

    def smc_log_density(self, x, mu):
        """Compute log probability of falling in current box for SMC sampling."""
        min_log_prob = gnp.log(1e-10)
        input_box = gnp.asarray(self.computer_experiments_problem.input_box)
        inside_box = sampcrit.isinbox(input_box, x)

        zpm, zpv = self.predict(x, convert_out=False, use_cache=True)
        current_box = self._interpolate_box(mu)

        sum_log_probs, _ = sampcrit.box_logprobability(current_box, zpm, zpv)
        
        return gnp.where(
            inside_box,
            gnp.maximum(min_log_prob, sum_log_probs).reshape(-1),
            -gnp.inf
        )

    def update_current_estimate(self):
        """Update probabilities and metrics."""
        # Get probabilities in log space first
        sum_log_probs, log_probs = sampcrit.box_logprobability(
            self.box_current, self.zpm, self.zpv
        )
        sum_log_tau, log_tau = sampcrit.box_misclassification_logprobability(
            self.box_current, self.zpm, self.zpv
        )

        # Convert back for metrics
        self.g_prod = gnp.exp(sum_log_probs)
        self.g = gnp.exp(log_probs)
        self.tau_prod = gnp.exp(sum_log_tau)
        self.tau = gnp.exp(log_tau)
        
        self.alpha = gnp.mean(self.g_prod)
        self.beta = gnp.mean(self.tau_prod)
        
        print(f"alpha = {self.alpha:.4e}, beta = {self.beta:.4e}")

    def sampling_criterion(self):
        """Compute wMSE."""
        wmse_sum, _ = sampcrit.box_wMSE(
            self.box_current,
            self.zpm,
            self.zpv,
            normalization=self.options.get('normalization', 1.0),
            alpha=self.options.get('alpha', 1.0),
            beta=self.options.get('beta', 0.5)
        )
        return wmse_sum
