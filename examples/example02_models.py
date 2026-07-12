"""Build gpmp-contrib Matérn models and inspect their predictions.

The examples below use the provided ``Model_ConstantMean_Maternp_*`` classes.
The class name states the model construction and the parameter-selection
criterion used by ``select_params``:

- ``ConstantMean``: one constant trend basis.
- ``Maternp``: fixed-regularity Matérn covariance from ``gpmp.kernel``.
- ``REML``, ``REMAP`` or ``ML``: criterion used for parameter selection.

For these fixed-regularity Matérn classes, covariance parameters follow the
``sigma2_rho`` convention:
``[log(sigma2), -log(rho_0), ..., -log(rho_{d-1})]``.

Author: Emmanuel Vazquez <emmanuel.vazquez@centralesupelec.fr>
Copyright (c) 2022-2026, CentraleSupelec
License: GPLv3 (see LICENSE file)

"""

import numpy as np
import gpmp as gp
import gpmpcontrib as gpc
import test_functions as tf
from gpmpcontrib.plot import (
    plot_1d,
    show_truth_vs_prediction,
    show_loo_errors,
)


# -----------------------------------
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

ind = [100, 1000, 1400, 1500, 1600]
ni = len(ind)
xi = xt[ind]
zi = problem(xi)

# Define the model, select parameters, predict, and draw conditional sample paths.
# Change model_choice to compare REML, REMAP, and ML.
model_choice = 1

if model_choice == 1:
    model = gpc.Model_ConstantMean_Maternp_REML(
        "1d_noisefree",
        problem.output_dim,
        mean_specification={"type": "constant"},
        covariance_specification={"p": 4},
    )
elif model_choice == 2:
    model = gpc.Model_ConstantMean_Maternp_REMAP(
        "1d_noisefree",
        problem.output_dim,
        mean_specification={"type": "constant"},
        covariance_specification={"p": 4},
    )
elif model_choice == 3:
    model = gpc.Model_ConstantMean_Maternp_ML(
        "1d_noisefree", problem.output_dim, covariance_specification={"p": 4}
    )

# model.select_params uses the criterion named by the model class.
model.select_params(xi, zi)
model.run_diagnosis(xi, zi)
zpm, zpv = model.predict(xi, zi, xt)

zpsim = model.compute_conditional_simulations(xi, zi, xt, n_samplepaths=5)

title = f"1D GP model with {ni} observations"
plot_1d(xt, zt, xi, zi, zpm[:, 0], zpv[:, 0], zpsim=zpsim, title=title)

# --------------------------------
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

ni = 10
ind = np.random.choice(n_test_grid**2, ni, replace=False)
xi = xt[ind]
zi = zt[ind]

# Define a two-output REMAP model and make predictions.
# The REMAP class uses restricted likelihood plus prior terms for the
# covariance parameters of each output.
model = gpc.Model_ConstantMean_Maternp_REMAP(
    "2d_noisefree",
    problem.output_dim,
    mean_specification={"type": "constant"},
    covariance_specification=[
        {"p": 1},
        {"p": 1},
    ],  # alternative form: covariance_specification={"p": 1}
)
model.select_params(xi, zi)
model.run_diagnosis(xi, zi)

zpm, zpv = model.predict(xi, zi, xt)
model.run_perf(xi, zi)

# Visualize results
show_truth_vs_prediction(zt, zpm)

zloom, zloov, eloo = model.loo(xi, zi)
show_loo_errors(zi, zloom, zloov)
