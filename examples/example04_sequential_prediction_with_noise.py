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
Copyright (c) 2022-2024, CentraleSupelec
License: GPLv3 (see LICENSE)

"""

import numpy as np
import gpmp as gp
import gpmpcontrib as gpc

# Set interactive mode for plotting (set to True if interactive plotting is desired)
interactive = False


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


def visualize_results(xt, zt, xi, zi, zpm, zpv, xnew=None):
    """
    Visualize the results of the predictions and the dataset.

    Parameters:
      xt (ndarray): Test points
      zt (ndarray): True values at test points
      xi (ndarray): Input data points
      zi (ndarray): Output values at input data points
      zpm (ndarray): Posterior mean values
      zpv (ndarray): Posterior variances
      xnew (ndarray, optional): New data point being added
    """
    fig = gp.misc.plotutils.Figure(isinteractive=interactive)
    fig.plot(xt, zt, "k", linewidth=1, linestyle=(0, (5, 5)))
    fig.plotdata(xi, zi)
    fig.plotgp(xt, zpm, zpv, colorscheme="simple")
    if xnew is not None:
        fig.plot(np.repeat(xnew, 2), fig.ylim(), color="tab:gray", linewidth=2)
    fig.xylabels("$x$", "$z$")
    fig.show(grid=True, xlim=[-1.0, 1.0], legend=True, legend_fontsize=9)


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
    mean_params={"type": "constant"},
    covariance_params={"p": 2},
)

# Using the append_noise_variance function to manage noise data
xi_with_noise_variance = append_noise_variance(
    xi, noise_variance, problem.output_dim)
xt_with_zero_noise_variance = append_noise_variance(xt, 0, problem.output_dim)

sp = gpc.SequentialPrediction(model)
sp.set_data_with_model_selection(xi_with_noise_variance, zi)

zpm, zpv = sp.predict(xt_with_zero_noise_variance)
visualize_results(xt, zt, xi, zi, zpm, zpv)

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
    visualize_results(xt, zt, sp.xi[:, :-output_dim], sp.zi, zpm, zpv, xi_new)

    zi_new = problem(xi_new)
    xi_new_with_noise = append_noise_variance(
        np.array([xi_new]), noise_variance, problem.output_dim
    )
    sp.set_new_eval_with_model_selection(xi_new_with_noise.flatten(), zi_new)
    zpm, zpv = sp.predict(xt_with_zero_noise_variance)

# Final visualization
visualize_results(xt, zt, sp.xi[:, :-output_dim], sp.zi, zpm, zpv)
