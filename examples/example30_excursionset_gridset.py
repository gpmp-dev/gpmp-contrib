"""Implement a sketch of the estimation of an excursion set on a fixed grid

Author: Emmanuel Vazquez <emmanuel.vazquez@centralesupelec.fr>
Copyright (c) 2022-2025, CentraleSupelec
License: GPLv3 (see LICENSE)

"""

import numpy as np
import matplotlib.pyplot as plt
import gpmp as gp
import gpmp.num as gnp
import gpmpcontrib as gpc
import gpmpcontrib.optim.excursionset as es
import gpmpcontrib.samplingcriteria as sampcrit


def test_function(x):
    x = np.asarray(x, dtype=np.float64)
    return (
        (0.4 * x - 0.3) ** 2
        + 1.0 * np.exp(-1 / 2 * (np.abs(x) / 0.2) ** 1.95)
        + np.exp(-1 / 2 * (x - 0.8) ** 2 / 0.1)
    )


# -- define a mono-objective problem


problem = gpc.ComputerExperiment(
    1,  # dim of search space
    [[-2.0], [2.0]],  # search box
    single_function=test_function,  # test function
)


# -- create initial dataset

nt = 2000
xt = gp.misc.designs.regulargrid(problem.input_dim, nt, problem.input_box)
zt = problem(xt)

ni = 4
ind = [i * 20 for i in [20, 46, 60, 80]]
xi = xt[ind]

u_target = 1.02

# -- initialize a model and the ei algorithm
model = gpc.Model_ConstantMean_Maternp_REML(
    "GP1d",
    output_dim=problem.output_dim,
    mean_params={"type": "constant"},
    covariance_params={"p": 2},
)

algo = es.ExcursionSetGridSearch(problem, model, xt, u_target)

algo.set_initial_design(xi)

# -- visualization


def plot(show=True, x=None, z=None):
    zpm, zpv = algo.predict(xt, convert_out=False)
    crit = sampcrit.excursion_wMSE(algo.u_target, zpm, zpv)
    pe = sampcrit.excursion_probability(algo.u_target, zpm, zpv)

    fig = gp.misc.plotutils.Figure(nrows=3, ncols=1, isinteractive=True)
    fig.subplot(1)
    fig.plot(xt, zt, "k", linewidth=0.5)
    fig.plot(fig.xlim(), [u_target] * 2, "k", linewidth=0.5)
    if z is not None:
        fig.plot(x, z, "b", linewidth=0.5)
    fig.plotdata(algo.xi, algo.zi)
    fig.plotgp(xt, gnp.to_np(zpm), gnp.to_np(zpv), colorscheme="simple")
    fig.ylabel("$z$")
    fig.title(f"Posterior GP, ni={algo.xi.shape[0]}")
    fig.subplot(2)
    fig.plot(xt, crit, "k", linewidth=0.5)
    # if plot_xnew:
    #     fig.plot(np.repeat(eialgo.xi[-1], 2), fig.ylim(), color="tab:gray", linewidth=3)
    fig.ylabel("criterion")
    fig.subplot(3)
    fig.plot(xt, pe, "k", linewidth=0.5)
    fig.ylabel("Prob. excursion")
    fig.xlabel("x")
    if show:
        fig.show()

    return fig


plot()

# make n new evaluations
n = 18
for i in range(n):
    print(f"Iteration {i} / {n}")
    algo.step()
    plot(show=True)
    # print model diagnosis
    gp.misc.modeldiagnosis.diag(
        algo.models[0]["model"], algo.models[0]["info"], algo.xi, algo.zi
    )
