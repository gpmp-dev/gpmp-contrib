from scipy.io import savemat
import warnings
import numpy as np
import matplotlib.pyplot as plt
import gpmp as gp
import gpmpcontrib as gpc


warnings.simplefilter("error", RuntimeWarning)

# -- definition of a mono-objective problem

# problem = gpc.test_problems.branin
# problem = gpc.test_problems.braninPlus
problem = gpc.test_problems.ishigami
# problem = gpc.test_problems.hartman6
dim = problem.input_dim

# -- create initial dataset

nt = 1000
xt = gp.misc.designs.maximinlhs(problem.input_dim, nt, problem.input_box)
zt = problem(xt)

ni = 5 * dim
xi = gp.misc.designs.maximinlhs(problem.input_dim, ni, problem.input_box)
zi = problem(xi)


# -- initialize a model and make predictions
model = gpc.Model_ConstantMean_Maternp_REMAP_gaussian_logsigma2_and_logrho_prior(
    "GPnd",
    output_dim=problem.output_dim,
    mean_specification={"type": "constant"},
    covariance_specification={"p": 2},
)

model.select_params(xi, zi)
model.run_diagnosis(xi, zi)

gp.modeldiagnosis.plot_selection_criterion_2d(
    model=model[0].model,
    info=model[0].info,
    param_indices=(1, 2),
    param_names=("rho1 (log10)", "rho2 (log10)"),
    criterion_name="selection criterion (lengthscales)",
    n=120,
    factor=5.0,
    shift_criterion=True,
)
gp.modeldiagnosis.plot_selection_criterion_2d(
    model=model[0].model,
    info=model[0].info,
    param_indices=(0, 1),
    param_names=("sigma (log10)", "rho1 (log10)"),
    criterion_name="selection criterion (sigma vs rho1)",
    n=120,
    factor=8.0,
    shift_criterion=True,
)

gp.plot.crosssections(
    model, xi, zi, problem.input_box, ind_i=[0, 1], ind_dim=list(range(dim))
)

gp.modeldiagnosis.plot_selection_criterion_crosssections(
    info=model[0].info, delta=5, n_points=200
)

# __import__("pdb").set_trace()

zloom, zloov, eloo = model.loo(xi, zi)
gp.plot.plot_loo(zi.reshape(-1), zloom.reshape(-1), zloov.reshape(-1))


method = "mh"
random_init = False
init_box = [-20, 20]
n_chains = 6
cp_center = gp.num.to_np(model[0].info.covparam).reshape(-1)
param_initial_states = np.tile(cp_center, (n_chains, 1))
rng = np.random.default_rng(2026)
if cp_center.size > 1:
    param_initial_states[:, 1:] += 0.8 * rng.standard_normal(
        (n_chains, cp_center.size - 1)
    )
param_initial_states[:, 0] += 0.05 * rng.standard_normal(n_chains)
param_initial_states = np.clip(param_initial_states, init_box[0], init_box[1])

if method == "nuts":
    res = model.sample_parameters(
        method="nuts",
        num_samples=1_000,
        num_warmup=600,
        n_chains=n_chains,
        target_accept=0.7,
        max_depth=12,
        jitter=1e-2,
        show_progress=True,
        silent=False,
        random_init=random_init,
        param_initial_states=param_initial_states,
        init_box=init_box,
    )

    nuts_info = res[0]["nuts"]
    accept = gp.num.to_np(nuts_info["accept_stat"])  # (n_samples, n_chains)
    div = gp.num.to_np(nuts_info["divergent"]).astype(float)  # (n_samples, n_chains)
    print("\nNUTS diagnostics:")
    print(f"  mean accept (all): {accept.mean():.3f}")
    print(f"  div rate (all):    {div.mean():.3f}")
    for c in range(div.shape[1]):
        print(
            f"  chain {c}: accept={accept[:, c].mean():.3f}, div_rate={div[:, c].mean():.3f}"
        )
elif method == "mh":
    res = model.sample_parameters(
        method="mh",
        n_steps_total=6_000,
        burnin_period=3_000,
        n_chains=n_chains,
        n_pool=2,
        show_progress=True,
        silent=False,
        random_init=random_init,
        param_initial_states=param_initial_states,
        init_box=init_box,
    )

    mh = res[0]["mh"]
    accept = gp.num.to_np(mh.accept[:, mh.burnin_period + 1 :]).astype(float)
    print("\nMH diagnostics:")
    print(f"  mean accept (all): {accept.mean():.3f}")
    for c in range(accept.shape[0]):
        print(f"  chain {c}: accept={accept[c, :].mean():.3f}")
else:
    raise ValueError("method must be 'nuts' or 'mh'")

# --------------------------------------------------------------------------
# Posterior sampling visualization (chains + marginals)
# --------------------------------------------------------------------------
samples = gp.num.to_np(res[0]["samples"])  # (n_chains, n_samples, n_params)
n_chains, n_samples, n_params = samples.shape

fig_traces, axes_traces = plt.subplots(
    n_params, 1, figsize=(10, max(3, 2.0 * n_params)), sharex=True
)
if n_params == 1:
    axes_traces = [axes_traces]
for j, ax in enumerate(axes_traces):
    for c in range(n_chains):
        ax.plot(samples[c, :, j], lw=0.8, alpha=0.7)
    ax.set_ylabel(f"$\\theta_{{{j}}}$")
axes_traces[-1].set_xlabel("Sample index")
fig_traces.suptitle(f"{method.upper()} chains")
fig_traces.tight_layout()

samples_flat = samples.reshape(n_chains * n_samples, n_params)
ncols = min(4, n_params)
nrows = int(np.ceil(n_params / ncols))
fig_marg, axes_marg = plt.subplots(
    nrows, ncols, figsize=(3.5 * ncols, 2.8 * nrows), squeeze=False
)
for j in range(n_params):
    r, c = divmod(j, ncols)
    ax = axes_marg[r, c]
    ax.hist(samples_flat[:, j], bins=40, density=True, alpha=0.75)
    ax.set_title(f"$\\theta_{{{j}}}$")
for k in range(n_params, nrows * ncols):
    r, c = divmod(k, ncols)
    axes_marg[r, c].axis("off")
fig_marg.suptitle(f"{method.upper()} posterior marginals")
fig_marg.tight_layout()


# savemat("data_xizi.mat", {"xi": xi, "zi": zi})
