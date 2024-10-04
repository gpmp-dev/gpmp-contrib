"""Demonstration script for gpmpcontrib.tModel

This script illustrates the use of the Model class in gpmp-contrib for
the approximation of a test function using Gaussian Process
modeling. The process involves setting up the problem, creating an
initial dataset, and making predictions using gpmp and gpmpcontrib
libraries.

Imports:
- numpy for numerical computations.
- gpmp and gpmpcontrib for Gaussian Process modeling and sequential prediction.
- test_functions for predefined test functions.

Author: Emmanuel Vazquez <emmanuel.vazquez@centralesupelec.fr>
Copyright (c) 2022-2023, CentraleSupelec
License: GPLv3 (see LICENSE file)

"""
import numpy as np
import gpmp as gp
import gpmpcontrib as gpc
import matplotlib.pyplot as plt
import test_functions as tf

# Set interactive mode for plotting (set to True if interactive plotting is desired)
interactive = False


def visualize_results_1d(xt, zt, xi, zi, zpm, zpv, zpsim):
    """
    Visualize the results of the predictions and the dataset.
    """
    fig = gp.misc.plotutils.Figure(isinteractive=interactive)
    fig.plot(xt, zt, "k", linewidth=1, linestyle=(0, (5, 5)))
    fig.plot(xt, zpsim[:, 0], "k", linewidth=0.5,
             label="conditional sample paths")
    fig.plot(xt, zpsim[:, 1:], "k", linewidth=0.5)
    fig.plotdata(xi, zi)
    fig.plotgp(xt, zpm, zpv, colorscheme="simple")
    fig.xylabels("$x$", "$z$")
    fig.show(grid=True, xlim=[-1.0, 1.0], legend=True, legend_fontsize=9)


def visualize_truth_vs_prediction(zt, zpm):
    num_outputs = zt.shape[1]
    fig, axs = plt.subplots(1, num_outputs, figsize=(6 * num_outputs, 5))

    for i in range(num_outputs):
        ax = axs[i] if num_outputs > 1 else axs
        ax.scatter(zt[:, i], zpm[:, i])
        ax.plot(
            [zt[:, i].min(), zt[:, i].max()], [
                zt[:, i].min(), zt[:, i].max()], "k--"
        )
        ax.set_xlabel("True Values")
        ax.set_ylabel("Predicted Values")
        ax.set_title(f"Output {i+1}")

    plt.tight_layout()
    plt.show()


# Example 1: Single-output 1d problem
# -----------------------------------

# Define the problem
problem = gpc.ComputerExperiment(
    1,  # Input dimension
    [[-1], [1]],  # Input domain (box)
    single_function=gp.misc.testfunctions.twobumps,  # Test function
)

# Generate dataset
nt = 2000
xt = gp.misc.designs.regulargrid(problem.input_dim, nt, problem.input_box)
zt = problem(xt)

ni = 3
ind = [100, 1000, 1400, 1500, 1600]
xi = xt[ind]
zi = problem(xi)

# Define the model, make predictions and draw conditional sample paths
model_choice = 1

if model_choice == 1:
    model = gpc.Model_ConstantMean_Maternp_REML(
        "1d_noisefree",
        problem.output_dim,
        mean_params={"type": "constant"},
        covariance_params={"p": 4},
    )
elif model_choice == 2:
    model = gpc.Model_ConstantMean_Maternp_ML(
        "1d_noisefree",
        problem.output_dim,
        covariance_params={"p": 4}
    )

model.select_params(xi, zi)
zpm, zpv = model.predict(xi, zi, xt)

zpsim = model.compute_conditional_simulations(xi, zi, xt, n_samplepaths=5)

visualize_results_1d(xt, zt, xi, zi, zpm[:, 0], zpv[:, 0], zpsim)

# Example 2: Two-output 2d problem
# --------------------------------

# Define the problem
pb_dict = {
    "functions": [tf.f1, tf.f2],
    "input_dim": 2,
    "input_box": [[0, 0], [1, 1]],
    "output_dim": 2,
}

problem = gpc.ComputerExperiment(
    pb_dict["input_dim"], pb_dict["input_box"], function_list=pb_dict["functions"]
)

# Generate data
n_test_grid = 21
xt1v, xt2v = np.meshgrid(
    np.linspace(problem.input_box[0][0], problem.input_box[1][0], n_test_grid),
    np.linspace(problem.input_box[0][1], problem.input_box[1][1], n_test_grid),
    indexing="ij",
)
xt = np.hstack((xt1v.reshape(-1, 1), xt2v.reshape(-1, 1)))
zt = problem.eval(xt)

ni = 5
ind = np.random.choice(n_test_grid**2, ni, replace=False)
xi = xt[ind]
zi = zt[ind]

# Define the model and make predictions
model = gpc.Model_ConstantMean_Maternp_REML(
    "2d_noisefree",
    problem.output_dim,
    mean_params={"type": "constant"},
    covariance_params=[
        {"p": 1},
        {"p": 1},
    ],  # alternative form: covariance_params={"p": 1}
)
model.select_params(xi, zi)
zpm, zpv = model.predict(xi, zi, xt)

visualize_truth_vs_prediction(zt, zpm)
