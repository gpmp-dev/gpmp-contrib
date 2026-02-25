"""
Example 11: Expected Improvement optimization with SMC-adapted search points.

This example runs a 1D Bayesian optimization loop on the `twobumps` function
over `[-1, 1]` using:

- a Matérn GP surrogate with REML parameter selection, and
- an SMC-based sequential strategy (`ExpectedImprovementSMC`) to adapt where
  candidate points are located before maximizing Expected Improvement (EI).

Objective
---------
Demonstrate how SMC can replace a fixed candidate grid by a particle cloud that
concentrates in promising regions, while EI is still used as the pointwise
decision criterion for choosing the next evaluation.

Workflow
--------
At each iteration:
1. Fit/update GP parameters from observations.
2. Predict posterior mean/variance on current SMC particles.
3. Compute EI on these particles.
4. Select the particle with maximal EI and evaluate the objective there.
5. Update the SMC particle cloud with a new target density derived from the
   updated GP posterior.
6. Repeat.

SMC procedure used here
-----------------------
The strategy uses `SequentialStrategySMC` with
`update_method="step_with_possible_restart"`.

Target log-density
^^^^^^^^^^^^^^^^^^
Particles are targeted toward regions with high posterior probability of
improvement via an excursion-based density:
- threshold parameter `u = -min(zi)` (current best value, with sign convention),
- log-density proportional to `log P(-Y(x) > u | data)`,
- points outside the input box get log-density `-inf`.

Tempering / restart logic
^^^^^^^^^^^^^^^^^^^^^^^^^
To avoid particle collapse when the target is too selective, the algorithm uses:
- an initial easier parameter (`-max(zi)`),
- a target harder parameter (`-min(zi)`),
- ESS-based safeguard (`min_ess_ratio`) and automatic restart,
- intermediate tempering levels selected by the configured SMC rule (`p0` in
  this setup), so overlap between successive targets remains adequate.

Particle update at each SMC stage
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
For each tempering stage:
1. Reweight particles by incremental change in target density.
2. Resample (residual resampling in this setup).
3. Move particles with Gaussian random-walk MH kernels.
4. Adapt proposal scale to keep acceptance in a target range.
5. Perform additional MH moves (`mh_steps`) for better mixing.

In this example, the resulting particle set becomes the new adaptive candidate
set for the next EI maximization step.

Sign convention
---------------
EI is computed using
`expected_improvement(-z_best, -zpm, zpv)`.
This sign flip is consistent with the criterion conventions used in
`gpmpcontrib.samplingcriteria`.

What is plotted
---------------
The figure has three panels:
1. Posterior GP (truth, data, posterior mean/uncertainty),
2. EI profile,
3. Excursion-related profile plus SMC particles with associated target-density
   values.

References
----------
P. Feliot, J. Bect, and E. Vazquez (2017).
"A Bayesian approach to constrained single- and multi-objective optimization."
Journal of Global Optimization.

J. Bect, L. Li, and E. Vazquez (2017).
"Bayesian subset simulation."
SIAM/ASA Journal on Uncertainty Quantification, 5(1), 762-786.

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

eialgo = ei.ExpectedImprovementSMC(problem, model)

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
    eialgo.model.run_diagnosis(eialgo.xi, eialgo.zi)

eialgo.smc.plot_state()
