import gpmp.num as gnp
import gpmp as gp
import gpmpcontrib.sequentialprediction as spred
import gpmpcontrib.samplingcriteria as sampcrit
import gpmpcontrib.smc as gpsmc
import numpy as np
import gpmpcontrib.regp as regp

class ExpectedImprovementR(spred.SequentialPrediction):

    def predict(self, xt):
        """Prediction"""
        zpm = np.empty((xt.shape[0], self.output_dim))
        zpv = np.empty((xt.shape[0], self.output_dim))
        for i in range(self.output_dim):
            _, (zpm[:, i], zpv[:, i]), _, _ = regp.predict(
                self.models[i]['model'], self.xi, self.zi[:, i], xt, self.R[i]
            )

        return zpm, zpv