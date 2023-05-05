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
    s = 3.
    ni = 15
    dim = 1
    nt = 200
    box = [[-1], [1]]
    xt = gp.misc.designs.regulargrid(dim, nt, box)
    zt = gnp.exp(s * gp.misc.testfunctions.twobumps(xt))

    xi = gp.misc.designs.ldrandunif(dim, ni, box)
    zi = gnp.exp(s * gp.misc.testfunctions.twobumps(xi))
   
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

# reGP prediction above u
print('\n===== reGP =====\n')
u = 2.0
R = gnp.numpy.array([[u, gnp.numpy.inf]])

zi_relaxed, (zpm, zpv), model, info_ret = regp.predict(model, xi, zi, xt, R)

gp.misc.modeldiagnosis.diag(model, info, xi, zi_relaxed)

x_limits = fig.axes[0].get_xlim()
plt.hlines(y=u, xmin=x_limits[0], xmax=x_limits[1], colors='b')
visualize_results(xt, zt, xi, zi_relaxed, zpm, zpv, fig, rgb_hue=[128, 128, 255])

# LOO
zloom, zloov, eloo = model.loo(xi, zi_relaxed)
gp.misc.plotutils.plot_loo(zi_relaxed, zloom, zloov)

# Threshold selection
Rgopt = regp.select_optimal_threshold_above_t0(model, xi, zi, u)

zi_relaxed, (zpm, zpv), model, info_ret = regp.predict(model, xi, zi, xt, Rgopt)

fig.axes[0].hlines(y=Rgopt[0], xmin=x_limits[0], xmax=x_limits[1], colors='g')
visualize_results(xt, zt, xi, zi_relaxed, zpm, zpv, fig, rgb_hue=[128, 255, 128])

# if __name__ == '__main__':
#     main()
