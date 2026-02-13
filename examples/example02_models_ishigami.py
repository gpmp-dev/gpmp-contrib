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
Copyright (c) 2022-2025, CentraleSupelec
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

# -- create problem & dataset
problem = gpc.test_problems.ishigami
problem.normalize_input = True
print('-'*60, '\n', problem, '\n', '-'*60)

dim_eff = problem.input_dim
dim_inact = 5
dim_tot = dim_eff + dim_inact
input_box = [[0] * dim_tot, [1] * dim_tot]

nt = 1000
xt = gp.misc.designs.maximinlhs(dim_tot, nt, input_box)
zt = problem(xt[:, :dim_eff])

ni = 30 * dim_eff
xi = gp.misc.designs.maximinlhs(dim_tot, ni, input_box)
zi = problem(xi[:, :dim_eff])

# -- Define the model, make predictions
model_choice = 2
match model_choice:
    case 1:
        model = gpc.Model_ConstantMean_Maternp_REML(
            "1d_noisefree",
            problem.output_dim,
            mean_specification={"type": "constant"},
            covariance_specification={"p": 4},
        )
    case 2:
        model = gpc.Model_ConstantMean_Maternp_REMAP(
            "1d_noisefree",
            problem.output_dim,
            mean_specification={"type": "constant"},
            covariance_specification={"p": 4},
        )
    case 3:
        model = gpc.Model_ConstantMean_Maternp_ML(
            "1d_noisefree", 
            problem.output_dim, 
            covariance_specification={"p": 4},
        )
    case _:
        raise ValueError(f"Unknown model_choice {model_choice}")

bounds = gp.kernel.empirical_bounds_factory(xi, zi)
model.select_params(xi, zi, bounds=bounds)

zpm, zpv = model.predict(xi, zi, xt)

# -- Visualize results
show_truth_vs_prediction(zt, zpm)

zloom, zloov, eloo = model.loo(xi, zi)
show_loo_errors(zi, zloom, zloov)

model.run_diag(xi, zi)
print("----")
model.run_perf(xi, zi, loo=True, xtzt=(xt, zt), zpmzpv=(zpm, zpv))


ind = list(range(0, 1 + dim_eff))
ind_pooled = (
    ([1] + list(range(1 + dim_eff, min(1 + dim_tot, 1 + dim_eff + 20))))
    if dim_inact > 0
    else None
)
model.plot_selection_criterion_crosssections(
    param_box=[[-2] * len(ind), [8] * len(ind)],
    param_box_pooled=[[-8] * len(ind_pooled), [5] * len(ind_pooled)],
    n_points=100,
    param_names=None,
    pooled=True,
    ind=ind,
    ind_pooled=ind_pooled,
)
