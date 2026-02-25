"""
Example 10: Expected Improvement optimization on a fixed candidate grid.

This example demonstrates a 1D Bayesian optimization loop on the `twobumps`
function over `[-1, 1]`, using:

- a Matérn GP surrogate with REML parameter selection, and
- an EI strategy over a fixed search set (`ExpectedImprovementGridSearch`).

Objective
---------
Show the baseline sequential EI workflow when the candidate set is fixed
(regular grid), as opposed to adaptive particle-based search.

Workflow
--------
At each iteration:
1. Update the GP model from current observations.
2. Predict posterior mean/variance on the fixed grid `xt`.
3. Compute EI on `xt`.
4. Select the grid point maximizing EI and evaluate the objective there.
5. Update the model and repeat.

Sign convention
---------------
EI is computed as:
`expected_improvement(-z_best, -zpm, zpv)`.
This sign flip follows the convention used in `gpmpcontrib.samplingcriteria`
while targeting improvement relative to the current best observation.

What is plotted
---------------
Three panels are shown at each iteration:
1. Posterior GP (truth, observed points, posterior mean/uncertainty),
2. EI profile on the fixed grid,
3. Excursion probability profile relative to current best value.

Notes
-----
- Initial design size: 3 points.
- Sequential additions: 6 new evaluations.
- `run_diagnosis` is called after each iteration to inspect model updates.

Reference
---------
D. R. Jones, M. Schonlau, and W. J. Welch (1998).
"Efficient Global Optimization of Expensive Black-Box Functions."
Journal of Global Optimization, 13, 455-492.

Author: Emmanuel Vazquez <emmanuel.vazquez@centralesupelec.fr>
Copyright (c) 2022-2026, CentraleSupelec
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
    mean_specification={"type": "constant"},
    covariance_specification={"p": 2},
)

eialgo = ei.ExpectedImprovementGridSearch(problem, model, xt)

eialgo.set_initial_design(xi)

# -- visualization


def plot(show=True, x=None, z=None):
    zpm, zpv = eialgo.predict(xt, convert_out=False)
    ei = sampcrit.expected_improvement(-gnp.min(eialgo.zi), -zpm, zpv)
    pe = sampcrit.excursion_probability(-gnp.min(eialgo.zi), -zpm, zpv)

    fig = gp.plot.Figure(nrows=3, ncols=1, isinteractive=True)
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
    # if plot_xnew:
    #     fig.plot(np.repeat(eialgo.xi[-1], 2), fig.ylim(), color="tab:gray", linewidth=3)
    fig.ylabel("EI")
    fig.subplot(3)
    fig.plot(xt, pe, "k", linewidth=0.5)
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
    eialgo.model.run_diagnosis(eialgo.xi, eialgo.zi)
