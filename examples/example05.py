import time
import warnings
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import savemat, loadmat
import gpmp as gp
import gpmp.num as gnp
import gpmpcontrib as gpc

def handle_warning(message, category, filename, lineno, file=None, line=None):
    print('A warning occurred:')
    print(message)
    print('Do you wish to continue?')

    while True:
        response = input('y/n: ').lower()
        if response not in {'y', 'n'}:
            print('Not understood.')
        else:
            break

    if response == 'n':
        raise category(message)

# warnings.simplefilter("error", RuntimeWarning)
warnings.showwarning = handle_warning

# -- definition of a mono-objective problem

# problem = gpc.test_problems.hartman6
# problem = gpc.test_problems.braninPlus
# problem = gpc.test_problems.branin
# problem = gpc.test_problems.ackley6
problem = gpc.test_problems.ishigami

problem.normalize_input = True

dim_eff = problem.input_dim

# -- create initial dataset

make_test_data = False
if make_test_data:
    nt = 1000
    xt = gp.misc.designs.maximinlhs(problem.input_dim, nt, problem.input_box)
    zt = problem(xt)

ni = 30 * dim_eff
xi = gp.misc.designs.maximinlhs(problem.input_dim, ni, problem.input_box)
zi = problem(xi)

dim_inact = 2
dim_tot = dim_eff + dim_inact
if dim_inact > 0:
    xi = np.hstack((xi, np.random.rand(ni, dim_inact)))

load_data = False
if load_data:
    res = loadmat("data_xizi_01.mat")
    xi = res["xi"]
    ni, dim_eff = xi.shape
    zi = res["zi"]
    dim_inact = 0
    dim_tot = dim_eff + dim_inact
    if dim_inact > 0:
        xi = np.hstack((xi, np.random.rand(ni, dim_inact)))

# -- initialize a model and make predictions
model = gpc.Model_ConstantMean_Maternp_REMAP_gaussian_logsigma2_and_logrho_prior(
    "GPnd",
    output_dim=problem.output_dim,
    mean_specification={"type": "constant"},
    covariance_specification={"p": 2},
)

tic = time.time()
model.select_params(xi, zi)
model.run_diag(xi, zi)

# -- selection criterion cross sections
ind = list(range(1, 1 + dim_eff))
ind_pooled = (
    ([1] + list(range(1 + dim_eff, min(1 + dim_tot, 1 + dim_eff + 20))))
    if dim_inact > 0
    else None
)
gp.modeldiagnosis.plot_selection_criterion_crosssections(
    info=model[0].info,
    param_box=[[-8] * len(ind), [8] * len(ind)],
    param_box_pooled=[[-8] * len(ind_pooled), [5] * len(ind_pooled)],
    n_points=2000,
    ind=ind,
    ind_pooled=ind_pooled,
)
print(f"Exec time: {time.time() - tic:.2f}s")

# -- Fisher information
information_matrix = model[0].model.fisher_information(xi, model[0].info.covparam)
print(f"|I| = {gnp.logdet(information_matrix)}")

ref_prior = False
if ref_prior:
    def neg_log_reference_prior(covparam):
        return -gp.kernel.log_prior_reference(model[0].model, covparam, xi)

    gp.modeldiagnosis.plot_selection_criterion_crosssections(
        selection_criterion=neg_log_reference_prior,
        covparam=model[0].info.covparam,
        ind=ind,
        ind_pooled=ind_pooled,
        param_box=[[-7] * 8, [5] * 8],
    )

# def jr_prior(covparam):
#     return gp.kernel.log_prior_jr(model[0].model, covparam, xi)

# gp.modeldiagnosis.plot_selection_criterion_crossections(
#     selection_criterion=jr_prior,
#     covparam=model[0].info.covparam,
#     ind=ind,
#     param_box=[[-4]*8, [1]*8],
#     param_exponentiate=False
# )

# -- Plot LOO predictions
plot_loo = False
if plot_loo:
    zloom, zloov, eloo = model.loo(xi, zi)
    gp.plot.plot_loo(zi.reshape(-1), zloom.reshape(-1), zloov.reshape(-1))


# -- Parameter posterior distribution
mcmc = False
if mcmc:
    n_chains = 6
    random_init = False
    init_box = None
    if hasattr(model[0], "info") and hasattr(model[0].info, "bounds") and model[0].info.bounds is not None:
        bnds = np.asarray(model[0].info.bounds, dtype=float)
        if bnds.ndim == 2 and bnds.shape[1] == 2:
            init_box = [bnds[:, 0].tolist(), bnds[:, 1].tolist()]

    cp_center = gp.num.to_np(model[0].info.covparam).reshape(-1)
    param_initial_states = np.tile(cp_center, (n_chains, 1))
    rng = np.random.default_rng(2026)
    param_initial_states += 0.05 * rng.standard_normal(param_initial_states.shape)
    if init_box is not None:
        lo = np.asarray(init_box[0], dtype=float)
        hi = np.asarray(init_box[1], dtype=float)
        param_initial_states = np.clip(param_initial_states, lo, hi)

    res = model.sample_parameters(
        method="mh",
        n_steps_total=5_000,
        burnin_period=2_000,
        n_chains=n_chains,
        n_pool=2,
        show_progress=True,
        silent=False,
        random_init=random_init,
        param_initial_states=param_initial_states,
        init_box=init_box,
    )

    # res = model.sample_parameters(
    #     n_steps_total=1_000,
    #     burnin_period=500,
    #     n_chains=4,
    #     n_pool=2,
    #     show_progress=True,
    #     silent=False,
    #     random_init=random_init,
    #     init_box=init_box,
    # )

    gp.modeldiagnosis.plot_selection_criterion_crosssections(
        info=model[0].info,
        param_box=[[-5] * len(ind), [10] * len(ind)],
        param_box_pooled=[[-12] * len(ind_pooled), [5] * len(ind_pooled)],
        n_points=200,
        ind=ind,
        ind_pooled=ind_pooled,
        covparam=res[0]["samples"][1, -1, :],
    )

smc = True
if smc:
    from matplotlib import interactive
    interactive(True)
    cp0 = gp.num.to_np(getattr(model[0].info, "covparam0", model[0].info.covparam)).reshape(-1)
    init_half_width = 0.5
    init_box = [
        (cp0 - init_half_width).tolist(),
        (cp0 + init_half_width).tolist(),
    ]
    sampling_box = [[-10.0] * cp0.size, [10.0] * cp0.size]
    res = model.sample_parameters(
        method="smc",
        init_box=init_box,
        sampling_box=sampling_box,
        n_particles=4000,
        min_ess_ratio=0.5,
        mh_steps=20,
        debug=True,
    )
    smc = res[0]['smc']
    x = smc.particles.x
    p = smc.particles.logpx
    smc.plot_empirical_distributions(parameter_indices=ind, parameter_indices_pooled=ind_pooled)
# gp.plot.crosssections(
#     model, xi, zi, problem.input_box, ind_i=[0, 1], ind_dim=list(range(dim))
# )


save_results = False
if save_results:
    savemat("data_xizi.mat", {"xi": xi, "zi": zi})
