import gpmp.num as gnp
import gpmp as gp
import gpmpcontrib.sequentialprediction as spred
import gpmpcontrib.samplingcriteria as sampcrit
import gpmpcontrib.smc as gpsmc
import numpy as np
import gpmpcontrib.regp as regp

t_getters = {
    'Constant': lambda xi, zi, n_init, l: np.quantile(zi[:n_init], l),
    'Concentration': lambda xi, zi, n_init, l: np.quantile(zi, l)
}

class ExpectedImprovementR(spred.SequentialPrediction):

    def set_options(self, options):
        default_options = {'n_smc': 1000}
        default_options.update(options)
        return default_options

    def predict(self, xt):
        """Prediction"""

        zpm = np.empty((xt.shape[0], self.output_dim))
        zpv = np.empty((xt.shape[0], self.output_dim))
        for i in range(self.output_dim):
            t_i = self.get_t(self.xi, self.zi[:, i], self.n_init)

            _, (zpm[:, i], zpv[:, i]), _, _ = regp.predict(
                self.models[i]['model'],
                self.xi,
                self.zi[:, i],
                xt,
                gnp.numpy.array([[t_i, gnp.numpy.inf]])
            )

        return zpm, zpv

    def log_prob_excursion(self, x):
        raise NotImplementedError

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
        raise NotImplementedError

    def set_initial_design(self, xi, update_search_space=True):
        self.n_init = xi.shape[0]
        super().set_initial_design(xi, update_model=False, update_search_space=update_search_space)

    def make_new_eval(self, xnew, update_search_space=True):
        super().make_new_eval(xnew, update_model=False, update_search_space=update_search_space)