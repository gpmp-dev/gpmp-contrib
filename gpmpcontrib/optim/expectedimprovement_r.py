# --------------------------------------------------------------
# Author: Sébastien Petit
# Copyright (c) 2023, CentraleSupelec
# License: GPLv3 (see LICENSE)
# --------------------------------------------------------------
import gpmp.num as gnp
import gpmp as gp
import gpmpcontrib.optim.expectedimprovement as ei
import gpmpcontrib.samplingcriteria as sampcrit
import gpmpcontrib.smc as gpsmc
import gpmpcontrib.regp as regp
import numpy as np

t_getters = {
    'Constant': lambda l: lambda xi, zi, n_init: np.quantile(zi[:n_init], l),
    'Concentration': lambda l: lambda xi, zi, n_init: np.quantile(zi, l)
}

class ExpectedImprovementR(ei.ExpectedImprovement):

    @staticmethod
    def build(problem, strategy, l):
        return ExpectedImprovementR(problem=problem, options={'t_getter': t_getters[strategy](l)})
    
    def set_options(self, options):
        if options is None:
            options = {}

        assert 't_getter' in options.keys(), "Options must contain a t_getter. See expectedimprovement-r.t_getters"
        self.get_t = options.pop('t_getter')

        default_options = {'n_smc': 1000, 'G': 10}
        default_options.update(options)
        return default_options

    def predict(self, xt):
        """Prediction"""
        zpm = np.empty((xt.shape[0], self.output_dim))
        zpv = np.empty((xt.shape[0], self.output_dim))
        for i in range(self.output_dim):
            zpm[:, i], zpv[:, i] = self.models[i]['model'].predict(
                self.xi, self.zi_relaxed[:, i], xt
            )
        return zpm, zpv

    def compute_conditional_simulations(
            self,
            compute_zsim=True,
            n_samplepaths=0,
            xt='None',
            type='intersection',
            method='chol'
    ):
        raise NotImplementedError

    def update_params(self):
        """Parameter selection"""

        self.zi_relaxed = self.zi.copy()

        for i in range(self.output_dim):

            covparam0 = self.models[i]['model'].covparam
            if covparam0 is None or self.force_param_initial_guess:
                # This will be used by regp.remodel
                assert self.models[i]["parameters_initial_guess_procedure"] == gp.kernel.anisotropic_parameters_initial_guess
                covparam0 = None
            else:
                covparam0 = gnp.asarray(covparam0)

            t0 = self.get_t(self.xi, self.zi[:, i], self.n_init)

            R = regp.select_optimal_threshold_above_t0(
                self.models[i]['model'], self.xi, gnp.asarray(self.zi[:, i]), t0, G=self.options['G']
            )

            self.models[i]['model'], self.zi_relaxed[:, i], _, info_ret = regp.remodel(
                self.models[i]['model'], self.xi, gnp.asarray(self.zi[:, i]), R, covparam0, True
            )

            self.models[i]['info'] = info_ret

    def set_initial_design(self, xi, **kwargs):
        self.n_init = xi.shape[0]
        super().set_initial_design(xi=xi, **kwargs)
