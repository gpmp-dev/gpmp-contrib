"""Implement a sketch of the BSS algorithm

Author: Emmanuel Vazquez <emmanuel.vazquez@centralesupelec.fr>
Copyright (c) 2022-2025, CentraleSupelec
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
    mean_params={"type": "constant"},
    covariance_params={"p": 2},
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

    fig = gp.misc.plotutils.Figure(nrows=3, ncols=1, isinteractive=True)
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
        print("(m[n]) Move particles and update threshold [n] times (default: 1)")
        print("(e[n]) Make a new evaluation [n] times (default: 1)")
        print("(u[n]) Move particles without updating threshold")
        print("(r) Restart the particles")
        print("(p) Plot current state")
        print("(q) Quit")

        user_input = input("Enter your choice: ").strip().lower()

        # Extract command and optional number (supports "m3", "u2", "m 3", etc.)
        match = re.match(r"([muerpq])\s*(\d*)", user_input)

        if not match:
            print("Invalid input. Please enter a valid choice.")
            continue

        user_choice, repeat_str = match.groups()
        # Default to 1 if empty
        repeat = int(repeat_str) if repeat_str.isdigit() else 1

        if user_choice == "m":
            for _ in range(repeat):
                p0 = 0.1
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

        elif user_choice == "u":
            for _ in range(repeat):
                algo.step_move_particles()  # Move without updating mu
                print(f"Moved particles without updating mu (current: {algo.mu:.4f})")
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
            print("Invalid input. Please enter 'm', 'u', 'e', 'r', 'p', or 'q'.")


interactive()
