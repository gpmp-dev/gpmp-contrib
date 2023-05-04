"""
Plot and optimize the restricted negative log-likelihood

Author: Emmanuel Vazquez <emmanuel.vazquez@centralesupelec.fr>
Copyright (c) 2022-2023, CentraleSupelec
License: GPLv3 (see LICENSE)
"""

import gpmp.num as gnp
import gpmp as gp
import matplotlib.pyplot as plt
import gpmpcontrib.regp as regp

def generate_data():
    """
    Data generation.
    
    Returns
    -------
    tuple
        (xt, zt): target data
        (xi, zi): input dataset
    """
    dim = 1
    nt = 200
    box = [[-1], [1]]
    xt = gp.misc.designs.regulargrid(dim, nt, box)
    zt = gnp.exp(3.2 * gp.misc.testfunctions.twobumps(xt))

    ni = 10
    xi = gp.misc.designs.ldrandunif(dim, ni, box)
    zi = gnp.exp(3.2 * gp.misc.testfunctions.twobumps(xi))
   
    return xt, zt, xi, zi


def constant_mean(x, param):
    return gnp.ones((x.shape[0], 1))


def kernel(x, y, covparam, pairwise=False):
    p = 2
    return gp.kernel.maternp_covariance(x, y, p, covparam, pairwise)


def visualize_results(xt, zt, xi, zi, zpm, zpv, fig=None, rgb_hue=[242, 64, 76]):
    """
    Visualize the results using gp.misc.plotutils (a matplotlib wrapper).
    
    Parameters
    ----------
    xt : numpy.ndarray
        Target x values
    zt : numpy.ndarray
        Target z values
    xi : numpy.ndarray
        Input x values
    zi : numpy.ndarray
        Input z values
    zpm : numpy.ndarray
        Posterior mean
    zpv : numpy.ndarray
        Posterior variance
    """
    if fig is None:
        fig = gp.misc.plotutils.Figure(isinteractive=True)
    fig.plot(xt, zt, 'k', linewidth=1, linestyle=(0, (5, 5)))
    fig.plotdata(xi, zi)
    fig.plotgp(xt, zpm, zpv, colorscheme='hue', rgb_hue=rgb_hue)
    fig.xlabel('$x$')
    fig.ylabel('$z$')
    fig.title('Posterior GP with parameters selected by ReML')

    return fig


# def main():
xt, zt, xi, zi = generate_data()

meanparam = None
covparam0 = None
model = gp.core.Model(constant_mean, kernel, meanparam, covparam0)

# Automatic selection of parameters using REML
model, info = gp.kernel.select_parameters_with_reml(model, xi, zi, info=True)
gp.misc.modeldiagnosis.diag(model, info, xi, zi)

# GP prediction
zpm, zpv = model.predict(xi, zi, xt)
fig = visualize_results(xt, zt, xi, zi, zpm, zpv)

# reGP prediction
u = 2.
R = gnp.numpy.array([[u, gnp.numpy.inf]])

(xi_relaxed, zi_relaxed), (zpm, zpv), model, info_ret = regp.predict(model, xi, zi, xt, R)

x_limits = fig.axes[0].get_xlim()
plt.hlines(y=u, xmin=x_limits[0], xmax=x_limits[1], colors='b')
visualize_results(xt, zt, xi_relaxed, zi_relaxed, zpm, zpv, fig, rgb_hue=[128, 128, 255])

plt.show()

# LOO
zloom, zloov, eloo = model.loo(xi_relaxed, zi_relaxed)
gp.misc.plotutils.plot_loo(zi_relaxed, zloom, zloov)

# Threshold selection

G = 20
t = gnp.logspace(gnp.log10(u), gnp.log10(gnp.max(zi)), G + 1)
t = t[:-1]

J = gnp.zeros(G)
for g in range(G):
    Rg = gnp.numpy.array([[t[g], gnp.numpy.inf]])
    (xi_relaxed, zi_relaxed), (zpm, zpv), model, info_ret = regp.predict(model, xi, zi, xt, Rg)
    zloom, zloov, eloo = model.loo(xi_relaxed, zi_relaxed)
    tCRPS = gp.misc.scoringrules.tcrps_gaussian(zloom, gnp.sqrt(zloov), zi_relaxed, a=-gnp.inf, b=u)
    J[g] = gnp.sum(tCRPS)

gopt = gnp.argmin(J)
Rgopt = gnp.numpy.array([[t[gopt], gnp.numpy.inf]])
(xi_relaxed, zi_relaxed), (zpm, zpv), model, info_ret = regp.predict(model, xi, zi, xt, Rgopt)
fig.axes[0].hlines(y=t[gopt], xmin=x_limits[0], xmax=x_limits[1], colors='g')
visualize_results(xt, zt, xi_relaxed, zi_relaxed, zpm, zpv, fig, rgb_hue=[128, 255, 128])

# if __name__ == '__main__':
#     main()
