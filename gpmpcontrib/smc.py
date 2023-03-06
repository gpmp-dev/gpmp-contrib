# --------------------------------------------------------------
# Author: Emmanuel Vazquez <emmanuel.vazquez@centralesupelec.fr>
# Copyright (c) 2023, CentraleSupelec
# License: GPLv3 (see LICENSE)
# --------------------------------------------------------------
import numpy as np
import scipy.stats as stats
import gpmp as gp


class SMC:
    '''SMC object: builds a set of particles targeting a
       distribution given by a log-pdf

    '''

    def __init__(self, box, n=1000):

        self.n = n
        (self.x, self.logpx, self.w) = self.particles_init(box, n)

        self.logpdf = None

        self.param_s = 0.05

    def particles_init(self, box, n):
        dim = len(box[0])
        x = gp.misc.designs.maximinlhs(dim, n, box)
        logpx = np.zeros((n,))
        w = np.full((n, ), 1 / n)

        return (x, logpx, w)

    def set_logpdf(self, logpdf):
        self.logpdf = logpdf

    def reweight(self):
        logpx_new = self.logpdf(self.x)
        self.w = self.w * np.exp(logpx_new - self.logpx)
        self.logpx = logpx_new

    def ess(self):
        '''https://en.wikipedia.org/wiki/Effective_sample_size'''
        return np.sum(self.w)**2 / np.sum(self.w**2)

    def resample(self):
        x_resampled = np.empty(self.x.shape)
        logpx_resampled = np.empty(self.logpx.shape)
        p = self.w / np.sum(self.w)
        try:
            counts = stats.multinomial.rvs(self.n, p)
        except:
            extype, value, tb = __import__("sys").exc_info()
            __import__("traceback").print_exc()
            __import__("pdb").post_mortem(tb)

        i = 0
        j = 0
        while j < self.n:
            while counts[j] > 0:
                x_resampled[i, :] = self.x[j, :]
                logpx_resampled[i] = self.logpx[j]
                counts[j] -= 1
                i += 1
            j += 1

        self.x = x_resampled
        self.logpx = logpx_resampled
        self.w = np.full((self.n, ), 1 / self.n)

    def pertubate(self):
        C = self.param_s * np.cov(self.x.reshape(self.x.shape[0], -1),
                                  rowvar=False)
        eps = stats.multivariate_normal.rvs(cov=C, size=self.n)

        return self.x + eps.reshape(self.n, -1)

    def move(self):
        y = self.pertubate()
        logpy = self.logpdf(y)

        rho = np.minimum(1, np.exp(logpy - self.logpx))

        for i in range(self.n):
            if np.random.rand(1) < rho[i]:
                self.x[i, :] = y[i, :]
                self.logpx[i] = logpy[i]
        debug = False
        if debug:
            import matplotlib.pyplot as plt
            plt.plot(self.x, self.logpx, '.'); plt.show()

    def step(self, logpdf):
        self.set_logpdf(logpdf)
        self.reweight()
        self.resample()
        self.move()
