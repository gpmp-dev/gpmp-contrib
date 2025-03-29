from scipy.io import savemat
import warnings
import numpy as np
import matplotlib.pyplot as plt
import gpmp as gp
import gpmpcontrib as gpc


warnings.simplefilter("error", RuntimeWarning)

# -- definition of a mono-objective problem

problem = gpc.test_problems.hartman6
# problem = gpc.test_problems.braninPlus
dim = problem.input_dim

# -- create initial dataset

nt = 1000
xt = gp.misc.designs.maximinlhs(problem.input_dim, nt, problem.input_box)
zt = problem(xt)

ni = 5 * dim
xi = gp.misc.designs.maximinlhs(problem.input_dim, ni, problem.input_box)
zi = problem(xi)


# -- initialize a model and make predictions
model = gpc.Model_ConstantMean_Maternp_REMAP(
    "GPnd",
    output_dim=problem.output_dim,
    mean_params={"type": "constant"},
    covariance_params={"p": 2},
)

model.select_params(xi, zi)
model.diagnosis(xi, zi)

zloom, zloov, eloo = model.loo(xi, zi)
gp.misc.plotutils.plot_loo(zi.reshape(-1), zloom.reshape(-1), zloov.reshape(-1))

random_init = True
init_box = [-8, -1]

res = model.sample_parameters(
    n_steps_total=5_000,
    burnin_period=4_000,
    n_chains=6,
    show_progress=True,
    silent=False,
    random_init=random_init,
    init_box=init_box,
)

gp.misc.plotutils.crosssections(
    model, xi, zi, problem.input_box, ind_i=[0, 1], ind_dim=list(range(dim))
)

gp.misc.modeldiagnosis.plot_selection_criterion_crossections(
    model[0].model, model[0].info, delta=5, n_points=200
)

savemat("data_xizi.mat", {"xi": xi, "zi": zi})
