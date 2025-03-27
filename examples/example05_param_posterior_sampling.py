import warnings
import numpy as np
import matplotlib.pyplot as plt
import gpmp as gp
import gpmp.num as gnp
import gpmpcontrib as gpc


warnings.simplefilter("error", RuntimeWarning)

# -- definition of a mono-objective problem
dim = 6
problem = gpc.ComputerExperiment(
    dim,  # dim of search space
    [[0] * dim, [1] * dim],  # search box
    single_function=gp.misc.testfunctions.hartmann6,  # test function
)

# -- create initial dataset

nt = 2000
xt = gp.misc.designs.maximinlhs(problem.input_dim, nt, problem.input_box)
zt = problem(xt)

ni = 200
ind = np.arange(ni)
xi = xt[ind]
zi = zt[ind]


# -- initialize a model and make predictions
model = gpc.Model_ConstantMean_Maternp_REMAP(
    "GP6d",
    output_dim=problem.output_dim,
    mean_params={"type": "constant"},
    covariance_params={"p": 2},
)

model.select_params(xi, zi)
model.diagnosis(xi, zi)

zloom, zloov, eloo = model.loo(xi, zi)
gp.misc.plotutils.plot_loo(zi.reshape(-1), zloom.reshape(-1), zloov.reshape(-1))

res = model.sample_parameters(
    n_steps_total=10_000,
    burnin_period=4_000,
    n_chains=2,
    show_progress=True,
    silent=False,
)

gp.misc.plotutils.crosssections(
    model, xi, zi, problem.input_box, ind_i=[0, 1], ind_dim=list(range(6))
)
