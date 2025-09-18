"""Sequential Prediction with Maximum MSE Sampling Strategy in a noisy setting

This script demonstrates the use of the SequentialPrediction class for
the approximation of a test function using a maximum MSE sampling
strategy. It includes steps to set up a problem, create an initial
dataset, make predictions, and iteratively improve the model with new
data points chosen based on maximum MSE.

Imports:
- numpy for numerical operations
- gpmp and gpmpcontrib for Gaussian Process modeling and sequential
  prediction

Author: Emmanuel Vazquez <emmanuel.vazquez@centralesupelec.fr>
Copyright (c) 2022-2025, CentraleSupelec
License: GPLv3 (see LICENSE)

"""

import numpy as np
import gpmp as gp
import gpmpcontrib as gpc
from gpmpcontrib.plot import plot_1d


def append_noise_variance(x, noise_variance, output_dim):
    """
    Appends noise variance information to each data point.

    Parameters:
      x (ndarray): Array of data points.
      noise_variance (float): The noise variance to append.
      output_dim (int): The number of output dimensions.

    Returns:
      ndarray: The data points with noise variance appended.
    """
    if output_dim > 0:
        noise_info = noise_variance * np.ones((x.shape[0], output_dim))
    else:
        noise_info = np.zeros((x.shape[0], output_dim))
    return np.hstack((x, noise_info))


# -- 1. Define a problem --
input_dim = 1
input_box = [[-1], [1]]
output_dim = 1
noise_variance = 0.2**2

problem = gpc.computerexperiment.StochasticComputerExperiment(
    input_dim,
    input_box,
    single_function=gp.misc.testfunctions.twobumps,
    simulated_noise_variance=noise_variance,
)

# -- 2. Create initial dataset --
nt = 2000
xt = gp.misc.designs.regulargrid(problem.input_dim, nt, problem.input_box)
zt = problem(xt, simulate_noise=False)

ind = [100, 1600] + [1000] * 15
xi = xt[ind]
zi = problem(xi)

# -- 3. Create a Model and a SequentialPrediction object --
model = gpc.Model_Noisy_ConstantMean_Maternp_REML(
    "Simple function",
    output_dim=problem.output_dim,
    mean_specification={"type": "constant"},
    covariance_specification={"p": 3},
)

# Using the append_noise_variance function to manage noise data
xi_with_noise_variance = append_noise_variance(xi, noise_variance, problem.output_dim)
xt_with_zero_noise_variance = append_noise_variance(xt, 0, problem.output_dim)

sp = gpc.SequentialPrediction(model)
sp.set_data_with_model_selection(xi_with_noise_variance, zi)

zpm, zpv = sp.predict(xt_with_zero_noise_variance)

title = f"1D GP model with {sp.ni} observations"
plot_1d(xt, zt, xi, zi, zpm, zpv, title=title)

# Function for selecting new data points based on maximum MSE


def mmse_sampling(seqpred, xt, xt_with_zero_noise_variance):
    """
    Select a new data point for evaluation based on maximum MSE.

    Parameters:
      seqpred (SequentialPrediction): The sequential prediction object
      xt (ndarray): Test points

    Returns:
      ndarray: The new data point selected for evaluation
    """
    zpm, zpv = seqpred.predict(xt_with_zero_noise_variance)
    maxmse_ind = np.argmax(zpv)
    xi_new = xt[maxmse_ind]
    return xi_new


# -- Iterative model improvement --
n = 10
for i in range(n):
    xi_new = mmse_sampling(sp, xt, xt_with_zero_noise_variance)
    plot_1d(xt, zt, sp.xi[:, :-output_dim], sp.zi, zpm, zpv, xnew=xi_new)

    zi_new = problem(xi_new)
    xi_new_with_noise = append_noise_variance(
        np.array([xi_new]), noise_variance, problem.output_dim
    )
    sp.set_new_eval_with_model_selection(xi_new_with_noise.flatten(), zi_new)
    zpm, zpv = sp.predict(xt_with_zero_noise_variance)

# Final visualization
zpsim = model.compute_conditional_simulations(
    sp.xi, sp.zi, xt_with_zero_noise_variance, n_samplepaths=3
)
title = f"1D GP model with {sp.ni} observations, conditional sample paths"
plot_1d(xt, zt, sp.xi[:, :-output_dim], sp.zi, zpm, zpv, zpsim=zpsim, title=title)
