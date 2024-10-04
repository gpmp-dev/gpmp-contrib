# --------------------------------------------------------------
# Authors: Emmanuel Vazquez <emmanuel.vazquez@centralesupelec.fr>
# Copyright (c) 2023, CentraleSupelec
# License: GPLv3 (see LICENSE)
# --------------------------------------------------------------
import gpmp.num as gnp
import gpmp as gp
import gpmpcontrib.samplingcriteria as sampcrit
from gpmpcontrib import SequentialStrategy


class ExpectedImprovement(SequentialStrategy):
    """
    Implements the Expected Improvement (EI) optimization algorithm
    using a Sequential Monte Carlo (SMC) method to adapt the search space.
    """

    def __init__(self, problem, model, options=None):
        super().__init__(problem=problem, model=model, options=options)

    def target_log_density(self, x, u):
        """
        Defines the log probability of an excursion, used as a target density for SMC.
        """
        min_threshold = 1e-6
        sigma2_scale_factor = 2.0**2

        input_box = gnp.asarray(self.computer_experiments_problem.input_box)
        b = sampcrit.isinbox(input_box, x)

        zpm, zpv = self.predict(x, convert_out=False)

        log_prob_excur = gnp.where(
            gnp.asarray(b),
            gnp.log(
                gnp.maximum(
                    min_threshold,
                    sampcrit.probability_excursion(
                        u, -zpm, sigma2_scale_factor * zpv),
                )
            ).flatten(),
            -gnp.inf,
        )

        return log_prob_excur

    def update_target_log_density_param(self):
        """
        Updates the target log density parameter for SMC based on the current estimate.
        """
        self.target_log_density_param = -self.current_estimate
        self.target_log_density_param_initial = -gnp.max(self.zi)

    def update_current_estimate(self):
        """
        Updates the current estimate of the objective function.

        Returns
        -------
        float
            The updated estimate value.
        """
        return gnp.min(self.zi)

    def sampling_criterion(self, x, zpm, zpv):
        """
        Computes the Expected Improvement (EI) at given points.
        """
        return sampcrit.expected_improvement(-self.current_estimate, -zpm, zpv)
