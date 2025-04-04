"""Implement a sketch of the EI algorithm

Author: Emmanuel Vazquez <emmanuel.vazquez@centralesupelec.fr>
Copyright (c) 2022-2025, CentraleSupelec
License: GPLv3 (see LICENSE)

"""

import warnings
import numpy as np
import matplotlib.pyplot as plt
import gpmp as gp
import gpmp.num as gnp
import gpmpcontrib as gpc
import gpmpcontrib.optim.expectedimprovement as ei
import gpmpcontrib.plot.visualization as gpv


warnings.simplefilter("error", RuntimeWarning)


def visualize(zt, zpm, zc, zi, zloom, zloov, dark=True):
    """
    Visualize both predictions vs truth and LOO errors on the same graph for each output dimension,
    including a histogram of zpm along the vertical axis placed on the left with its bottom axis inverted
    so it aligns against the right, and points colored by an additional array. The scatter plot will have
    no y-axis labels since they are shown on the histogram, ensuring clarity and avoiding redundancy.

    Args:
    - zt: True values for predictions.
    - zpm: Predicted values from the model.
    - zc: Array for coloring the scatter points.
    - zi: True values for LOO.
    - zloom: LOO model predictions.
    - zloov: Variance of LOO predictions.
    """
    if dark:
        preferred_styles = ['dark_background']
        plt.style.use(next((s for s in preferred_styles if s in plt.style.available), None))
        identity_line_color = "#1f77b4"
    else:
        preferred_styles = ['seaborn-v0_8-paper', 'bmh', 'classic']
        plt.style.use(next((s for s in preferred_styles if s in plt.style.available), None))
        identity_line_color = "#222222"

    num_outputs = zt.shape[1]
    fig = plt.figure(figsize=(6 * num_outputs, 5))
    gs = fig.add_gridspec(
        1, num_outputs * 2, width_ratios=[1, 3] * num_outputs, wspace=0.05
    )

    for i in range(num_outputs):
        ax_hist = fig.add_subplot(gs[0, 2 * i])
        ax_scatter = fig.add_subplot(gs[0, 2 * i + 1], sharey=ax_hist)

        # Remove y-axis labels from the scatter plot to avoid redundancy
        # ax_scatter.set_yticklabels([])

        # Color map for visualization
        cmap = plt.get_cmap("viridis")

        # Plot predictions vs true values, colored by zc
        scatter = ax_scatter.scatter(
            zt[:, i], zpm[:, i], c=zc, cmap=cmap, label="Model Predictions"
        )
        # Add a color bar
        cbar = plt.colorbar(scatter, ax=ax_scatter)
        cbar.set_label("Color by probabilities of excursion")

        # Plot identity line for predictions
        ax_scatter.plot(
            [zt[:, i].min(), zt[:, i].max()],
            [zt[:, i].min(), zt[:, i].max()],
            color=identity_line_color,
            linestyle="--",
        )

        # Plot LOO predictions with error bars
        ax_scatter.errorbar(
            zi[:, i],
            zloom[:, i],
            yerr=1.96 * gnp.sqrt(zloov[:, i]),
            fmt="ro",
            ls="None",
            label="LOO Predictions with 95% CI",
        )

        # Quadrant formed by the minimum of observations
        min_true = min(zi[:, i])
        min_pred = min(zi[:, i])
        ax_scatter.axvline(x=min_true, color=identity_line_color, linestyle="--", linewidth=1)
        ax_scatter.axhline(y=min_pred, color=identity_line_color, linestyle="--", linewidth=1)

        ax_scatter.set_xlabel("True Values")
        ax_scatter.set_title(f"Output {i+1}")
        ax_scatter.legend()
        ax_scatter.grid(True, "major", linestyle="--", linewidth=0.5, alpha=0.7)

        # Rotated histogram along the vertical axis, with its bottom against the right axis
        ax_hist.hist(
            zpm[:, i], bins=20, orientation="horizontal", color="gray", alpha=0.7
        )
        ax_hist.set_title("Histogram of Predictions")
        ax_hist.grid(True, "major", linestyle="--", linewidth=0.5, alpha=0.7)

        # Invert the x-axis to position the histogram's bottom against the right axis
        ax_hist.invert_xaxis()

    # plt.tight_layout()
    plt.show()


# -- definition of a mono-objective problem
dim = 6
problem = gpc.ComputerExperiment(
    dim,  # dim of search space
    [[0] * dim, [1] * dim],  # search box
    single_function=gp.misc.testfunctions.hartmann6,  # test function
)

# -- create initial dataset

nt = 200
xt = gp.misc.designs.ldrandunif(problem.input_dim, nt, problem.input_box)
zt = problem(xt)

ni = 3 * problem.input_dim
ind = np.arange(ni)
xi = xt[ind]
zi = zt[ind]

# gpv.parallel_coordinates_plot(xt, zt, p=zt[:, 0], xi=xi, zi=zi, show_type=True)


# -- initialize a model and the ei algorithm
model = gpc.Model_ConstantMean_Maternp_REMAP(
    "GP6d",
    output_dim=problem.output_dim,
    mean_params={"type": "constant"},
    covariance_params={"p": 2},
)

eialgo = ei.ExpectedImprovementSMC(problem, model)

eialgo.set_initial_design(xi)

# make n new evaluations
n = 20
for i in range(n):
    print(f"Iteration {i} / {n}")
    try:
        eialgo.step()
    except RuntimeWarning:
        __import__("pdb").post_mortem()

    # print model diagnosis
    eialgo.model.diagnosis(eialgo.xi, eialgo.zi)

    particles_x = eialgo.smc.particles.x
    particles_zt = problem(particles_x)
    particles_logpx = eialgo.smc.particles.logpx
    zpm, zpv = eialgo.predict(particles_x)
    zloom, zloov, eloo = eialgo.models[0]["model"].loo(eialgo.xi, eialgo.zi)

    pcp = False
    if pcp:
        gpv.parallel_coordinates_plot(particles_x, zpm, p=particles_logpx, show_p=False)

    # eialgo.smc.plot_particles()

    visualize(
        particles_zt,
        zpm,
        particles_logpx,
        eialgo.zi,
        zloom.reshape(-1, 1),
        zloov.reshape(-1, 1),
    )


gpv.parallel_coordinates_plot(
    particles_x,
    zpm,
    p=particles_logpx,
    show_p=True,
    xi=eialgo.xi,
    zi=eialgo.zi,
    show_type=True,
)
