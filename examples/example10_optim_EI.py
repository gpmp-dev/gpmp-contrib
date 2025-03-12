"""Implement a sketch of the EI algorithm

Author: Emmanuel Vazquez <emmanuel.vazquez@centralesupelec.fr>
Copyright (c) 2022-2024, CentraleSupelec
License: GPLv3 (see LICENSE)

"""

import numpy as np
import matplotlib.pyplot as plt
import gpmp as gp
import gpmp.num as gnp
import gpmpcontrib as gpc
import gpmpcontrib.optim.expectedimprovement as ei
import gpmpcontrib.samplingcriteria as sampcrit

# -- define a mono-objective problem

problem = gpc.ComputerExperiment(
    1,  # dim of search space
    [[-1], [1]],  # search box
    single_function=gp.misc.testfunctions.twobumps,  # test function
)


# -- create initial dataset

nt = 2000
xt = gp.misc.designs.regulargrid(problem.input_dim, nt, problem.input_box)
zt = problem(xt)

ni = 3
ind = [100, 1000, 1600]
xi = xt[ind]

# -- initialize a model and the ei algorithm
model = gpc.Model_ConstantMean_Maternp_REML(
    "GP1d",
    output_dim=problem.output_dim,
    mean_params={"type": "constant"},
    covariance_params={"p": 2},
)

eialgo = ei.ExpectedImprovement(problem, model)

eialgo.set_initial_design(xi)

# -- visualization


def plot(show=True, x=None, z=None):

    __import__("pdb").set_trace()

    zpm, zpv = eialgo.predict(xt, convert_out=False)
    ei = sampcrit.expected_improvement(-gnp.min(eialgo.zi), -zpm, zpv)
    pe = sampcrit.probability_excursion(-gnp.min(eialgo.zi), -zpm, zpv)

    fig = gp.misc.plotutils.Figure(nrows=3, ncols=1, isinteractive=True)
    fig.subplot(1)
    fig.plot(xt, zt, "k", linewidth=0.5)
    if z is not None:
        fig.plot(x, z, "b", linewidth=0.5)
    fig.plotdata(eialgo.xi, eialgo.zi)
    fig.plotgp(xt, gnp.to_np(zpm), gnp.to_np(zpv), colorscheme="simple")
    fig.ylabel("$z$")
    fig.title(f"Posterior GP, ni={eialgo.xi.shape[0]}")
    fig.subplot(2)
    fig.plot(xt, -ei, "k", linewidth=0.5)
    fig.ylabel("EI")
    fig.subplot(3)
    fig.plot(xt, pe, "k", linewidth=0.5)
    # fig.plot(eialgo.smc.particles.x, np.zeros(eialgo.smc.n), ".")
    fig.plot(eialgo.smc.particles.x, gnp.exp(eialgo.smc.particles.logpx), ".")
    fig.ylabel("Prob. excursion")
    fig.xlabel("x")
    if show:
        fig.show()

    return fig


plot()

# make n new evaluations
n = 6
for i in range(n):
    print(f"Iteration {i} / {n}")
    eialgo.step()
    plot(show=True)
    # print model diagnosis
    gp.misc.modeldiagnosis.diag(
        eialgo.models[0]["model"], eialgo.models[0]["info"], eialgo.xi, eialgo.zi
    )

eialgo.smc.plot_state()
