"""
Example 31: excursion-set estimation with Bayesian Subset Simulation (BSS).

This example illustrates an interactive 1D excursion-set workflow on
`[-2, 2]` using:

- a Matérn GP surrogate with REML parameter selection, and
- a BSS-style SMC strategy (`ExcursionSetBSS`) that adapts particles toward
  exceedance regions.

Goal
----
Estimate the excursion set
    Γ(u_target) = {x : f(x) > u_target}
for `u_target = 1.02`, while progressively adapting the particle cloud and
selecting new evaluations where excursion classification is uncertain.

BSS / SMC procedure used here
-----------------------------
The algorithm introduces an interpolation parameter `mu in [0, 1]` and an
intermediate threshold
    u(mu) = (1 - mu) * u_init + mu * u_target.

SMC particles target a log-density proportional to
    log P(ξ(x) > u(mu) | data)
inside the input box (and `-inf` outside). Increasing `mu` makes the target
event rarer and concentrates particles in more relevant regions.

At each SMC stage, particles are:
1. reweighted with the updated excursion-based target,
2. resampled (residual scheme),
3. moved by MH random-walk kernels.

The helper `compute_next_logpdf_param(..., p0=...)` is used to choose the next
`mu` level with controlled stage-to-stage overlap.

Sequential design criterion
---------------------------
New evaluations are chosen by maximizing the excursion-oriented criterion
`excursion_wMSE(u_current, zpm, zpv)`, i.e. a misclassification/variance-driven
score that prioritizes points informative for excursion-set reconstruction.

Reported diagnostics
--------------------
After updates, the script tracks:
- `g(x) = P(ξ(x) > u_current | data)` (excursion probability),
- `tau(x)` (misclassification probability),
- `alpha = mean(g)` (estimated excursion volume),
- `beta = mean(tau)` (expected misclassification volume).

Interactive commands
--------------------
- `n [p0]`: update `mu` with the p0 rule (threshold progression only).
- `s x`: set `mu` manually.
- `m[n]`: move particles `n` times without changing `mu`.
- `u[n]`: update `mu` then move particles `n` times.
- `e[n]`: perform `n` new objective evaluations (with model update).
- `r`: restart particles / threshold progression.
- `p`: redraw plots.
- `q`: quit.

What is plotted
---------------
1. Posterior GP with observations and current threshold `u_current`.
2. Excursion-set criterion profile (`excursion_wMSE`).
3. Excursion probability profile and current SMC particle cloud.

References
----------
P. Feliot, J. Bect, and E. Vazquez (2017).
"A Bayesian approach to constrained single- and multi-objective optimization."
Journal of Global Optimization.

J. Bect, L. Li, and E. Vazquez (2017).
"Bayesian subset simulation."
SIAM/ASA Journal on Uncertainty Quantification, 5(1), 762-786.

J. Bect, D. Ginsbourger, L. Li, V. Picheny, and E. Vazquez (2012).
"Sequential design of computer experiments for the estimation of a probability
of failure."
Statistics and Computing, 22, 773-793.

Author: Emmanuel Vazquez <emmanuel.vazquez@centralesupelec.fr>
Copyright (c) 2022-2026, CentraleSupelec
License: GPLv3 (see LICENSE)
"""

import re
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

# -- initialize a model and the ei algorithm
model = gpc.Model_ConstantMean_Maternp_REML(
    "GP1d",
    output_dim=problem.output_dim,
    mean_specification={"type": "constant"},
    covariance_specification={"p": 2},
)

u_init = 0.0
u_target = 1.02

algo = es.ExcursionSetBSS(problem, model, u_init, u_target)

algo.set_initial_design(xi)

# -- visualization


def plot(show=True, x=None, z=None):
    zpm, zpv = algo.predict(xt, convert_out=False)
    crit = sampcrit.excursion_wMSE(algo.u_current, zpm, zpv)
    pe = sampcrit.excursion_probability(algo.u_current, zpm, zpv)

    fig = gp.plot.Figure(nrows=3, ncols=1, isinteractive=True, figsize=(12,9))
    fig.subplot(1)
    fig.plot(xt, zt, "k", linewidth=0.5)
    fig.plot(fig.xlim(), [algo.u_current] * 2, "k", linewidth=0.5)
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
    fig.plot(algo.smc.particles.x, gnp.exp(algo.smc.particles.logpx), ".")
    fig.ylabel("Prob. excursion")
    fig.xlabel("x")
    if show:
        fig.show()

    return fig


plot()


def interactive():
    while True:
        print("\nChoose an action:")
        print("(n [p0]) Update threshold with the p0 rule (default: 0.1)")
        print("(s x) Set mu to x (e.g., 's 0.4')")
        print("--")
        print("(m[n]) Move particles without updating threshold [n] times (default: 1)")
        print("(u[n]) Move particles and update threshold [n] times (default: 1)")
        print("--")
        print("(e[n]) Make a new evaluation [n] times (default: 1)")
        print("--")
        print("(r) Restart the particles")
        print("(p) Plot current state")
        print("(q) Quit")

        user_input = input("Enter your choice: ").strip().lower()

        # Extract command and optional number or value (supports "m3", "u2", "s 0.4", "n 0.2", etc.)
        match = re.match(r"([muerpsqn])\s*([\d\.]*)", user_input)

        if not match:
            print("Invalid input. Please enter a valid choice.")
            continue

        user_choice, param_str = match.groups()

        # Handle the "s" option separately (setting mu manually)
        if user_choice == "s":
            try:
                mu_value = float(param_str)
                if not (0 <= mu_value <= 1):
                    print("mu must be between 0 and 1.")
                    continue
                algo.mu = mu_value
                print(f"Set mu to {algo.mu:.4f}")
            except ValueError:
                print("Invalid value for mu. Please enter a number between 0 and 1.")
            continue  # Skip to next iteration after setting mu

        # Default repeat count for commands that take an integer (m, u, e)
        repeat = int(param_str) if param_str.isdigit() else 1

        # Default p0 value for "n" (threshold update)
        if user_choice == "n":
            try:
                p0 = float(param_str) if param_str else 0.1  # Default p0 = 0.1
                mu_next = algo.smc.compute_next_logpdf_param(
                    algo.smc_log_density,
                    current_logpdf_param=algo.mu,
                    target_logpdf_param=1.0,
                    p0=p0,
                    debug=False,
                )
                algo.mu = mu_next
                print(f"Updated mu: {algo.mu:.4f} using p0 = {p0:.2f}")
                plot()
            except ValueError:
                print("Invalid p0 value. Please enter a number (e.g., 'n 0.2').")
            continue

        elif user_choice == "m":
            for _ in range(repeat):
                algo.step_move_particles()  # Move without updating mu
                print(f"Moved particles without updating mu (current: {algo.mu:.4f})")
                plot()

        elif user_choice == "u":
            for _ in range(repeat):
                p0 = 0.1  # Default threshold update probability
                mu_next = algo.smc.compute_next_logpdf_param(
                    algo.smc_log_density,
                    current_logpdf_param=algo.mu,
                    target_logpdf_param=1.0,
                    p0=p0,
                    debug=False,
                )
                algo.mu = mu_next
                print(f"Updated mu: {algo.mu:.4f}")
                algo.step_move_particles_with_mu(mu_next)
                kappa = algo.beta / algo.alpha
                print(f"New kappa: {kappa:.4e}")
                plot()

        elif user_choice == "e":
            for _ in range(repeat):
                algo.step_new_evaluation()
                kappa = algo.beta / algo.alpha
                print(f"New kappa: {kappa:.4e}")
                plot()

        elif user_choice == "r":
            algo.restart()
            print("Particles restarted.")
            plot()

        elif user_choice == "p":
            plot()
            print("Plot updated.")

        elif user_choice == "q":
            print("Exiting the algorithm.")
            break

        else:
            print(
                "Invalid input. Please enter 'm', 'u', 'e', 's x', 'n p0', 'r', 'p', or 'q'."
            )


interactive()
