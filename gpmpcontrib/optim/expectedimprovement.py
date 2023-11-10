# --------------------------------------------------------------
# Authors: Emmanuel Vazquez <emmanuel.vazquez@centralesupelec.fr>
#          Sébastien Petit
# Copyright (c) 2023, CentraleSupelec
# License: GPLv3 (see LICENSE)
# --------------------------------------------------------------
import gpmp.num as gnp
import gpmp as gp
import gpmpcontrib.sequentialprediction as spred
import gpmpcontrib.samplingcriteria as sampcrit
import gpmpcontrib.smc as gpsmc

class AbortException(Exception):
    pass

class AlreadyVisitedException(AbortException):
    pass

class ExpectedImprovement(spred.SequentialPrediction):

    def __init__(self, problem, model=None, options=None):

        # computer experiments problem
        self.computer_experiments_problem = problem

        # model initialization
        super().__init__(output_dim=1, models=model)

        # options
        self.options = self.set_options(options)

        # search space
        self.smc = self.init_smc(self.options['n_smc'])

        # ei values
        self.ei = None

        # minimum
        self.minimum = None

    def set_options(self, options):
        default_options = {'n_smc': 1000}
        return default_options

    def init_smc(self, n_smc):
        return gpsmc.SMC(self.computer_experiments_problem.input_box, n_smc)

    def build_models(self, models):
        if models is None:
            self.models = [
                {
                    "name": "",
                    "model": None,
                    "parameters_initial_guess_procedure": None,
                    "selection_criterion": None,
                    "info": None,
                }
                for i in range(self.output_dim)
            ]

            for i in range(self.output_dim):
                # TODO:() The mean is estimated and then used by pluggin, is it a good idea to define it here?
                self.models[i]['model'] = gp.core.Model(
                    self.constant_mean, self.default_covariance, None, None, meantype="known"
                )
                self.models[i][
                    "parameters_initial_guess_procedure"
                ] = gp.kernel.anisotropic_parameters_initial_guess_constant_mean
                self.models[i][
                    "selection_criterion"
                ] = self.models[i]['model'].negative_log_likelihood
        else:
            self.models = models

    def constant_mean(self, x, param):
        return param * gnp.ones((x.shape[0], 1))

    def log_prob_excursion(self, x):
        min_threshold = 1e-6
        b = sampcrit.isinbox(self.computer_experiments_problem.input_box, x)

        zpm, zpv = self.predict(x)

        minimum = -gnp.numpy.min(self.zi)

        log_prob_excur = gnp.where(
            gnp.asarray(b),
            gnp.log(
                gnp.maximum(
                    min_threshold,
                    sampcrit.probability_excursion(
                        minimum,
                        -zpm,
                        zpv
                    )
                )
            ).flatten(),
            -gnp.inf
        )

        return gnp.to_np(log_prob_excur)

    def update_search_space(self):
        self.smc.step(self.log_prob_excursion)

    def set_initial_design(self, xi, update_model=True, update_search_space=True):
        zi = self.computer_experiments_problem.eval(xi)

        if update_model:
            super().set_data_with_model_selection(xi, zi)
        else:
            super().set_data(xi, zi)

        self.minimum = gnp.numpy.min(self.zi)

        if update_search_space:
            self.update_search_space()

    def make_new_eval(self, xnew, update_model=True, update_search_space=True):
        znew = self.computer_experiments_problem.eval(xnew)

        if update_model:
            self.set_new_eval_with_model_selection(xnew, znew)
        else:
            self.set_new_eval(xnew, znew)

        self.minimum = gnp.numpy.min(self.zi)

        if update_search_space:
            self.update_search_space()

    def step(self):
        # evaluate ei on the search space
        zpm, zpv = self.predict(self.smc.x)
        self.ei = sampcrit.expected_improvement(-self.minimum, -zpm, zpv)
    
        # make new evaluation
        x_new = self.smc.x[gnp.argmax(gnp.asarray(self.ei))].reshape(1, -1)

        if (x_new == self.xi).all(1).any():
            raise AlreadyVisitedException()

        self.make_new_eval(x_new)
