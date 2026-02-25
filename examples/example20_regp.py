"""
Example 20: relaxed Gaussian process (reGP) for threshold-oriented modeling.

This example demonstrates how to use reGP to improve predictive behavior in
a range of interest when strict interpolation can be detrimental.

Objective
---------
Compare three modeling configurations on the same 1D dataset:
1. Standard GP fit with REML (baseline).
2. reGP with a fixed relaxation interval above a threshold ``u``.
3. reGP with an automatically selected threshold (above ``u``).

Method overview
---------------
- Generate synthetic positive data from a smooth test function and add
  heteroscedastic noise that increases for high responses.
- Fit a Matérn GP by REML and compute posterior predictions.
- Apply reGP with interval ``R = [[u, +inf]]`` so observations above ``u``
  may be relaxed while preserving behavior in the region of interest.
- Select an improved threshold with
  ``regp.select_optimal_threshold_above_t0`` (tCRPS-based criterion), then
  refit reGP and compare predictions.
- Display diagnostics and LOO summaries for the relaxed fit.

Visual output (Figure 1)
------------------------
- Red curve: baseline REML GP posterior.
- Blue curve: reGP posterior with fixed threshold ``u``.
- Green curve: reGP posterior with optimized threshold.
- Blue horizontal line: threshold ``u``.
- Green horizontal line: optimized threshold.

Reference
---------
S. J. Petit, J. Bect, and E. Vazquez (2025).
"Relaxed Gaussian Process Interpolation: a Goal-Oriented Approach to Bayesian Optimization."
Journal of Machine Learning Research, 26(195):1-70.
https://www.jmlr.org/papers/v26/22-0828.html
"""

import gpmp.num as gnp
import gpmp as gp
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
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
    s = 1.5
    ni = 15
    dim = 1
    nt = 200
    box = [[-1], [1]]
    xt = gp.misc.designs.regulargrid(dim, nt, box)
    zt = gnp.exp(s * gp.misc.testfunctions.twobumps(xt))

    xi = gnp.array([-0.7, -0.6, -0.45]).reshape(-1, 1)
    xi = gnp.vstack((xi, gnp.asarray(gp.misc.designs.ldrandunif(dim, ni, box))))
    zi = gnp.exp(s * gp.misc.testfunctions.twobumps(xi))

    u = 2.0
    noise_variance = 0.01 * gnp.maximum(zi - u, 0.0)
    zi = zi + gnp.sqrt(noise_variance) * gnp.randn(zi.shape)

    return xt, zt, xi, zi


def constant_mean(x, param):
    return gnp.ones((x.shape[0], 1))


def kernel(x, y, covparam, pairwise=False):
    p = 2
    return gp.kernel.maternp_covariance(x, y, p, covparam, pairwise)


def visualize_results(xt, zt, xi, zi, zpm, zpv, fig=None, rgb_hue=[242, 64, 76]):
    """
    Visualize the results using gp.plot (a matplotlib wrapper).

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
        fig = gp.plot.Figure(isinteractive=True)
    fig.plot(xt, zt, "k", linewidth=1, linestyle=(0, (5, 5)))
    fig.plotdata(xi, zi)
    fig.plotgp(xt, zpm, zpv, colorscheme="hue", rgb_hue=rgb_hue)
    fig.xlabel("$x$")
    fig.ylabel("$z$")
    fig.title("Posterior GP")

    return fig


# def main():
xt, zt, xi, zi = generate_data()

meanparam = None
covparam0 = None
model = gp.core.Model(constant_mean, kernel, meanparam, covparam0)

# Automatic selection of parameters using REML
model, info = gp.kernel.select_parameters_with_reml(model, xi, zi, info=True)
gp.modeldiagnosis.diag(model, info, xi, zi)

# GP prediction
zpm, zpv = model.predict(xi, zi, xt)
fig = visualize_results(xt, zt, xi, zi, zpm, zpv)

# reGP prediction above u
print("\n===== reGP =====\n")
u = 2.0
R = gnp.numpy.array([[u, gnp.numpy.inf]])

zi_relaxed, (zpm, zpv), model, info_ret = regp.predict(model, xi, zi, xt, R)

gp.modeldiagnosis.diag(model, info, xi, zi_relaxed)

x_limits = fig.axes[0].get_xlim()
plt.hlines(y=u, xmin=x_limits[0], xmax=x_limits[1], colors="b")
visualize_results(xt, zt, xi, zi_relaxed, zpm, zpv, fig, rgb_hue=[128, 128, 255])

# LOO
zloom, zloov, eloo = model.loo(xi, zi_relaxed)
gp.plot.plot_loo(zi_relaxed, zloom, zloov)

# Threshold selection
Rgopt = regp.select_optimal_threshold_above_t0(model, xi, zi, u)

zi_relaxed, (zpm, zpv), model, info_ret = regp.predict(model, xi, zi, xt, Rgopt)

fig.axes[0].hlines(y=Rgopt[0], xmin=x_limits[0], xmax=x_limits[1], colors="g")
visualize_results(xt, zt, xi, zi_relaxed, zpm, zpv, fig, rgb_hue=[128, 255, 128])

# Legend for posterior GP curves and threshold lines
legend_handles = [
    Line2D([0], [0], color=(242 / 255, 64 / 255, 76 / 255), lw=2, label="REML"),
    Line2D([0], [0], color=(128 / 255, 128 / 255, 255 / 255), lw=2, label=f"reGP (u={u})"),
    Line2D([0], [0], color=(128 / 255, 255 / 255, 128 / 255), lw=2, label="reGP (optimal threshold)"),
    Line2D([0], [0], color="b", lw=1.5, label="threshold u"),
    Line2D([0], [0], color="g", lw=1.5, label="optimal threshold"),
]
fig.axes[0].legend(handles=legend_handles, loc="best")

# if __name__ == '__main__':
#     main()
