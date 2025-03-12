# --------------------------------------------------------------
# Authors: Emmanuel Vazquez <emmanuel.vazquez@centralesupelec.fr>
# Copyright (c) 2023-2024, CentraleSupelec
# License: GPLv3 (see LICENSE)
# --------------------------------------------------------------
import gpmp.num as gnp
import gpmp as gp
import gpmpcontrib.samplingcriteria as sampcrit
from gpmpcontrib import SequentialStrategySMC


class ExpectedImprovement(SequentialStrategySMC):
    """
    Implements the Expected Improvement (EI) optimization algorithm
    using a Sequential Monte Carlo (SMC) method to adapt the search space.
    """

    def __init__(self, problem, model, options=None):
        super().__init__(problem=problem, model=model, options=options)

    def smc_log_density(self, x, u):
        """
        Defines the log probability of an excursion, used as a target density for SMC.
        """
        min_threshold = gnp.log(1e-10)
        sigma2_scale_factor = 1.0**2

        input_box = gnp.asarray(self.computer_experiments_problem.input_box)
        b = sampcrit.isinbox(input_box, x)

        zpm, zpv = self.predict(x, convert_out=False)

        log_prob_excur = gnp.where(
            gnp.asarray(b),
            gnp.maximum(
                min_threshold,
                sampcrit.log_probability_excursion(u, -zpm, sigma2_scale_factor * zpv),
            ).flatten(),
            -gnp.inf,
        )

        return log_prob_excur

    def update_smc_log_density_param(self):
        """
        Updates the target log density parameter for SMC based on the current estimate.
        """
        self.smc_log_density_param = -self.current_estimate
        self.smc_log_density_param_initial = -gnp.max(self.zi)

    def update_current_estimate(self):
        """
        Updates the current estimate of the minimum.
        """
        return gnp.min(self.zi)

    def sampling_criterion(self, x, zpm, zpv):
        """
        Computes the Expected Improvement (EI) at given points.
        """
        return sampcrit.expected_improvement(-self.current_estimate, -zpm, zpv)
