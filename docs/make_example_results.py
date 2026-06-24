"""Generate static result figures for the Sphinx example pages."""

from __future__ import annotations

import io
import os
import sys
from contextlib import redirect_stdout
from pathlib import Path

os.environ.setdefault("GPMP_LOG_LEVEL", "WARNING")
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

DOCS_DIR = Path(__file__).resolve().parent
REPO_ROOT = DOCS_DIR.parent
OUT_DIR = DOCS_DIR / "source" / "_static" / "example_results"

sys.path.insert(0, str(REPO_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import gpmp as gp
import gpmp.num as gnp
import gpmpcontrib as gpc
import gpmpcontrib.optim.excursionset as es
import gpmpcontrib.optim.expectedimprovement as ei
import gpmpcontrib.optim.setinversion as si
import gpmpcontrib.regp as regp
import gpmpcontrib.samplingcriteria as sampcrit


def _to_np(x):
    return np.asarray(gnp.to_np(x))


def _flat(x):
    return _to_np(x).reshape(-1)


def _seed(seed=1234):
    np.random.seed(seed)
    gnp.set_seed(seed)


def _save(fig, filename):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / filename
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path.relative_to(REPO_ROOT)}")


def _save_gpfig(fig, filename):
    fig.fig.tight_layout()
    _save(fig.fig, filename)


def _write_text(filename, text):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / filename
    path.write_text(text.rstrip() + "\n", encoding="utf-8")
    print(f"wrote {path.relative_to(REPO_ROOT)}")


def _quiet_call(func, *args, **kwargs):
    with redirect_stdout(io.StringIO()):
        return func(*args, **kwargs)


def _twobumps_problem():
    return gpc.ComputerExperiment(
        1,
        [[-1], [1]],
        single_function=gp.misc.testfunctions.twobumps,
    )


def _excursion_test_function(x):
    x = np.asarray(gnp.to_np(x), dtype=np.float64)
    return (
        (0.4 * x - 0.3) ** 2
        + np.exp(-0.5 * (np.abs(x) / 0.2) ** 1.95)
        + np.exp(-0.5 * (x - 0.8) ** 2 / 0.1)
    )


def _plot_gp_prediction_1d(
    xt,
    zt,
    xi,
    zi,
    zpm,
    zpv,
    filename,
    title,
    *,
    truth_label="truth",
    data_label="observations",
    xlim=None,
):
    xt = _to_np(xt)
    xi = _to_np(xi)
    zi = _to_np(zi)
    zpm = _to_np(zpm)
    zpv = _to_np(zpv)
    if zt is not None:
        zt = _to_np(zt)

    fig = gp.plot.Figure(isinteractive=False, figsize=(6.5, 4.0))
    if zt is not None:
        fig.plot(xt, zt, "k", linewidth=1.0, label=truth_label)
    fig.plotdata(xi, zi, label=data_label)
    fig.plotgp(xt, zpm, zpv, colorscheme="simple")
    fig.title(title)
    fig.xylabels("$x$", "$z$")
    fig.grid()
    fig.legend(fontsize=8)
    if xlim is not None:
        fig.xlim(xlim)
    _save_gpfig(fig, filename)


def _plot_prediction_diagnostics_1d(
    xt,
    zt,
    xi,
    zi,
    zpm,
    zpv,
    criterion,
    probability,
    filename,
    title,
    *,
    threshold=None,
    box=None,
    particles_x=None,
    particles_height=None,
    criterion_yscale=None,
    probability_label="probability",
):
    xt = _to_np(xt)
    zt = _to_np(zt)
    xi = _to_np(xi)
    zi = _to_np(zi)
    zpm = _to_np(zpm)
    zpv = _to_np(zpv)
    criterion = _to_np(criterion)
    probability = _to_np(probability)

    fig = gp.plot.Figure(nrows=3, ncols=1, isinteractive=False, figsize=(6.5, 7.0))

    fig.subplot(1)
    fig.plot(xt, zt, "k", linewidth=1.0, label="truth")
    fig.plotdata(xi, zi, label="observations")
    fig.plotgp(xt, zpm, zpv, colorscheme="simple")
    if threshold is not None:
        fig.axhline(threshold, color="k", linewidth=0.8, label="threshold")
    if box is not None:
        lower, upper = _flat(box)
        fig.ax.axhspan(lower, upper, color="#BFBFBF", alpha=0.35, label="box")
    fig.title(title)
    fig.ylabel("$z$")
    fig.grid()
    fig.legend(fontsize=8)

    fig.subplot(2)
    fig.plot(xt, criterion, "k", linewidth=1.0)
    fig.ylabel("criterion")
    if criterion_yscale is not None:
        fig.ax.set_yscale(criterion_yscale)
    fig.grid()

    fig.subplot(3)
    fig.plot(xt, probability, "k", linewidth=1.0, label=probability_label)
    if particles_x is not None and particles_height is not None:
        fig.plot(
            particles_x,
            particles_height,
            "rs",
            markerfacecolor="none",
            markersize=4,
            label="particles",
        )
        fig.legend(fontsize=8)
    fig.xylabels("$x$", probability_label)
    fig.grid()

    _save_gpfig(fig, filename)


def example01_branin():
    _seed(101)
    problem = gpc.test_problems.branin
    problem.normalize_input = True

    x1 = np.linspace(problem.input_box[0][0], problem.input_box[1][0], 140)
    x2 = np.linspace(problem.input_box[0][1], problem.input_box[1][1], 140)
    X1, X2 = np.meshgrid(x1, x2)
    X = np.column_stack((X1.ravel(), X2.ravel()))
    Z = _to_np(problem.eval_objectives(X)).reshape(X1.shape)

    fig, ax = plt.subplots(figsize=(6.0, 4.8))
    cf = ax.contourf(X1, X2, Z, levels=40, cmap="Greys")
    cs = ax.contour(X1, X2, Z, levels=14, colors="#35605a", linewidths=0.7)
    ax.clabel(cs, inline=True, fontsize=7, fmt="%.1f")
    fig.colorbar(cf, ax=ax, label="objective value")
    ax.set_title("Branin objective on normalized inputs")
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    _save(fig, "example01_branin_contour.png")


def getting_started_hartmann4_prediction():
    _seed(1234)
    problem = gpc.test_problems.hartmann4
    box = problem.input_box

    xi = gp.misc.designs.ldrandunif(problem.input_dim, 40, box)
    zi = problem(xi)
    xt = gp.misc.designs.ldrandunif(problem.input_dim, 300, box)
    zt = problem(xt)

    model = gpc.Model_ConstantMean_Maternp_REML(
        "hartmann4",
        output_dim=problem.output_dim,
        mean_specification={"type": "constant"},
        covariance_specification={"p": 3},
    )
    _quiet_call(model.select_params, xi, zi, method_options={"maxiter": 80})
    zpm, zpv = model.predict(xi, zi, xt)

    with redirect_stdout(io.StringIO()) as buf:
        model.run_diagnosis(xi, zi)
    _write_text("getting_started_hartmann4_diagnosis.txt", buf.getvalue())

    with redirect_stdout(io.StringIO()) as buf:
        print(model[0].get_param())
    _write_text("getting_started_hartmann4_param.txt", buf.getvalue())

    with redirect_stdout(io.StringIO()) as buf:
        model.run_perf(xi, zi, xtzt=(xt, zt), zpmzpv=(zpm, zpv))
    _write_text("getting_started_hartmann4_perf.txt", buf.getvalue())

    fig = gp.plot.Figure(isinteractive=False, figsize=(5.2, 4.8))
    zt_ = _flat(zt)
    zpm_ = _flat(zpm)
    zmin = min(float(np.min(zt_)), float(np.min(zpm_)))
    zmax = max(float(np.max(zt_)), float(np.max(zpm_)))
    fig.plot(zt_, zpm_, "ko", markersize=3)
    fig.plot([zmin, zmax], [zmin, zmax], "k--", linewidth=1)
    fig.xylabels("reference value", "posterior mean")
    fig.title("Hartmann4 prediction on test points")
    fig.grid()
    _save_gpfig(fig, "getting_started_hartmann4_prediction.png")


def guide_hartmann4_remap_prior():
    _seed(1234)
    problem = gpc.test_problems.hartmann4
    box = problem.input_box

    xi = gp.misc.designs.ldrandunif(problem.input_dim, 40, box)
    zi = problem(xi)

    model = gpc.Model_ConstantMean_Maternp_REMAP(
        "hartmann4",
        output_dim=problem.output_dim,
        mean_specification={"type": "constant"},
        covariance_specification={"p": 3},
    )
    model.set_prior(gamma=1.5, sigma2_coverage=0.95, alpha=1.0, output_idx=0)
    _quiet_call(model.select_params, xi, zi, method_options={"maxiter": 80})

    with redirect_stdout(io.StringIO()) as buf:
        print(model.get_prior(0))
    _write_text("guide_hartmann4_remap_prior.txt", buf.getvalue())


def example02_gp_prediction():
    _seed(102)
    problem = _twobumps_problem()

    nt = 800
    xt = gp.misc.designs.regulargrid(problem.input_dim, nt, problem.input_box)
    zt = problem(xt)
    xi = xt[[40, 400, 560, 600, 640]]
    zi = problem(xi)

    model = gpc.Model_ConstantMean_Maternp_REML(
        "1d_noisefree",
        problem.output_dim,
        mean_specification={"type": "constant"},
        covariance_specification={"p": 4},
    )
    _quiet_call(model.select_params, xi, zi, method_options={"maxiter": 80})
    zpm, zpv = model.predict(xi, zi, xt)
    _plot_gp_prediction_1d(
        xt,
        zt,
        xi,
        zi,
        zpm[:, 0],
        zpv[:, 0],
        "example02_gp_prediction.png",
        "Matérn GP prediction from five observations",
        xlim=[-1.0, 1.0],
    )


def example04_noisy_sequential():
    _seed(104)
    problem = gpc.computerexperiment.StochasticComputerExperiment(
        1,
        [[-1], [1]],
        single_function=gp.misc.testfunctions.twobumps,
        simulated_noise_variance=0.2**2,
    )

    nt = 700
    xt = gp.misc.designs.regulargrid(problem.input_dim, nt, problem.input_box)
    zt = problem(xt, simulate_noise=False)
    xi = xt[[45, 120, 210, 300, 350, 350, 350, 430, 520, 610]]
    zi = problem(xi)

    def append_noise_variance(x, noise_variance):
        x = np.asarray(gnp.to_np(x))
        return np.hstack((x, noise_variance * np.ones((x.shape[0], 1))))

    model = gpc.Model_Noisy_ConstantMean_Maternp_REML(
        "noisy_1d",
        output_dim=problem.output_dim,
        mean_specification={"type": "constant"},
        covariance_specification={"p": 3},
    )
    xi_noise = append_noise_variance(xi, 0.2**2)
    xt_noise = append_noise_variance(xt, 0.0)
    _quiet_call(model.select_params, xi_noise, zi, method_options={"maxiter": 80})
    zpm, zpv = model.predict(xi_noise, zi, xt_noise)
    _plot_gp_prediction_1d(
        xt,
        zt,
        xi,
        zi,
        zpm[:, 0],
        zpv[:, 0],
        "example04_noisy_prediction.png",
        "Noisy Matérn GP with replicated observations",
        truth_label="noise-free function",
        data_label="noisy observations",
        xlim=[-1.0, 1.0],
    )


def example10_ei_grid():
    _seed(110)
    problem = _twobumps_problem()
    nt = 600
    xt = gp.misc.designs.regulargrid(problem.input_dim, nt, problem.input_box)
    zt = problem(xt)
    xi = xt[[30, 300, 480]]

    model = gpc.Model_ConstantMean_Maternp_REML(
        "GP1d",
        output_dim=problem.output_dim,
        mean_specification={"type": "constant"},
        covariance_specification={"p": 2},
    )
    algo = ei.ExpectedImprovementGridSearch(problem, model, xt)
    _quiet_call(algo.set_initial_design, xi)
    for _ in range(2):
        _quiet_call(algo.step)

    zpm, zpv = algo.predict(xt, convert_out=False)
    ei_values = sampcrit.expected_improvement(-gnp.min(algo.zi), -zpm, zpv)
    ei_max = gnp.max(ei_values)
    ei_values_plot = gnp.maximum(ei_values, ei_max * 1e-8)
    pe = sampcrit.excursion_probability(-gnp.min(algo.zi), -zpm, zpv)
    _plot_prediction_diagnostics_1d(
        xt,
        zt,
        algo.xi,
        algo.zi,
        zpm,
        zpv,
        ei_values_plot,
        pe,
        "example10_ei_grid.png",
        "Fixed-grid EI after two sequential additions",
        criterion_yscale="log",
        probability_label="excursion probability",
    )


def example11_ei_smc():
    _seed(111)
    problem = _twobumps_problem()
    nt = 600
    xt = gp.misc.designs.regulargrid(problem.input_dim, nt, problem.input_box)
    zt = problem(xt)
    xi = xt[[30, 300, 480]]

    model = gpc.Model_ConstantMean_Maternp_REML(
        "GP1d",
        output_dim=problem.output_dim,
        mean_specification={"type": "constant"},
        covariance_specification={"p": 2},
    )
    algo = ei.ExpectedImprovementSMC(
        problem,
        model,
        options={"n_smc": 160, "update_search_space_at_init": False},
    )
    _quiet_call(algo.set_initial_design, xi)
    _quiet_call(algo.update_search_space)

    zpm, zpv = algo.predict(xt, convert_out=False)
    ei_values = sampcrit.expected_improvement(-gnp.min(algo.zi), -zpm, zpv)
    pe = sampcrit.excursion_probability(-gnp.min(algo.zi), -zpm, zpv)
    xp = _flat(algo.smc.particles.x)
    logpx = _flat(algo.smc.particles.logpx)
    particle_height = np.exp(logpx - np.nanmax(logpx))

    _plot_prediction_diagnostics_1d(
        xt,
        zt,
        algo.xi,
        algo.zi,
        zpm,
        zpv,
        ei_values,
        pe,
        "example11_ei_smc.png",
        "EI with SMC-adapted candidate particles",
        particles_x=xp,
        particles_height=particle_height,
        probability_label="relative target",
    )


def _regp_data():
    s = 1.5
    ni = 15
    dim = 1
    nt = 300
    box = [[-1], [1]]
    xt = gp.misc.designs.regulargrid(dim, nt, box)
    zt = gnp.exp(s * gp.misc.testfunctions.twobumps(xt))
    xi = gnp.array([-0.7, -0.6, -0.45]).reshape(-1, 1)
    xi = gnp.vstack((xi, gnp.asarray(gp.misc.designs.ldrandunif(dim, ni, box))))
    zi = gnp.exp(s * gp.misc.testfunctions.twobumps(xi))
    u = 2.0
    noise_variance = 0.01 * gnp.maximum(zi - u, 0.0)
    zi = zi + gnp.sqrt(noise_variance) * gnp.randn(*zi.shape)
    return xt, zt, xi, zi


def _regp_constant_mean(x, param):
    return gnp.ones((x.shape[0], 1))


def _regp_kernel(x, y, covparam, pairwise=False):
    return gp.kernel.maternp_covariance(x, y, 2, covparam, pairwise)


def example20_regp():
    _seed(120)
    xt, zt, xi, zi = _regp_data()
    model = gp.core.Model(_regp_constant_mean, _regp_kernel, None, None)
    model, _info = _quiet_call(
        gp.kernel.select_parameters_with_reml, model, xi, zi, info=True
    )
    zpm0, _zpv0 = model.predict(xi, zi, xt)

    u = 2.0
    R = gnp.numpy.array([[u, gnp.numpy.inf]])
    zi_relaxed, (zpm1, _zpv1), model1, _info1 = _quiet_call(
        regp.predict, model, xi, zi, xt, R
    )
    Ropt = _quiet_call(regp.select_optimal_threshold_above_t0, model1, xi, zi, u)
    u_opt = float(np.asarray(gnp.to_np(Ropt)).reshape(-1)[0])
    zi_relaxed2, (zpm2, _zpv2), _model2, _info2 = _quiet_call(regp.predict, model, xi, zi, xt, Ropt)

    fig = gp.plot.Figure(isinteractive=False, figsize=(6.5, 4.2))
    fig.plot(xt, zt, "k", linestyle="--", linewidth=1.0, label="truth")
    fig.plotdata(xi, zi, label="observations")
    fig.plot(
        xi,
        zi_relaxed2,
        "gs",
        markerfacecolor="none",
        markersize=5,
        label="relaxed values",
    )
    fig.plot(xt, zpm0, "r", linewidth=1.5, label="REML")
    fig.plot(xt, zpm1, "b", linewidth=1.5, label=f"reGP u={u:.1f}")
    fig.plot(xt, zpm2, "g", linewidth=1.5, label="reGP selected threshold")
    fig.axhline(u, color="b", linewidth=0.9)
    fig.axhline(u_opt, color="g", linewidth=0.9)
    fig.title("Relaxed GP predictions and threshold lines")
    fig.xylabels("$x$", "$z$")
    fig.grid()
    fig.legend(fontsize=8)
    _save_gpfig(fig, "example20_regp.png")


def _excursion_problem():
    return gpc.ComputerExperiment(
        1,
        [[-2.0], [2.0]],
        single_function=_excursion_test_function,
    )


def example30_excursion_grid():
    _seed(130)
    problem = _excursion_problem()
    nt = 700
    xt = gp.misc.designs.regulargrid(problem.input_dim, nt, problem.input_box)
    zt = problem(xt)
    xi = xt[[140, 322, 420, 560]]
    u_target = 1.02

    model = gpc.Model_ConstantMean_Maternp_REMAP(
        "GP1d",
        output_dim=problem.output_dim,
        mean_specification={"type": "constant"},
        covariance_specification={"p": 2},
    )
    algo = es.ExcursionSetGridSearch(problem, model, xt, u_target)
    _quiet_call(algo.set_initial_design, xi)
    for _ in range(5):
        _quiet_call(algo.step)

    zpm, zpv = algo.predict(xt, convert_out=False)
    crit = sampcrit.excursion_wMSE(algo.u_target, zpm, zpv)
    pe = sampcrit.excursion_probability(algo.u_target, zpm, zpv)
    _plot_prediction_diagnostics_1d(
        xt,
        zt,
        algo.xi,
        algo.zi,
        zpm,
        zpv,
        crit,
        pe,
        "example30_excursion_grid.png",
        "Excursion-set design on a fixed grid",
        threshold=u_target,
        probability_label="excursion probability",
    )


def example31_excursion_bss():
    _seed(131)
    problem = _excursion_problem()
    nt = 700
    xt = gp.misc.designs.regulargrid(problem.input_dim, nt, problem.input_box)
    zt = problem(xt)
    xi = xt[[140, 322, 420, 560]]
    u_init = 0.0
    u_target = 1.02

    model = gpc.Model_ConstantMean_Maternp_REML(
        "GP1d",
        output_dim=problem.output_dim,
        mean_specification={"type": "constant"},
        covariance_specification={"p": 2},
    )
    algo = es.ExcursionSetBSS(
        problem,
        model,
        u_init,
        u_target,
        options={"n_smc": 160, "update_search_space_at_init": False},
    )
    _quiet_call(algo.set_initial_design, xi)
    _quiet_call(algo.step_move_particles_with_mu, 0.85)

    zpm, zpv = algo.predict(xt, convert_out=False)
    crit = sampcrit.excursion_wMSE(algo.u_current, zpm, zpv)
    pe = sampcrit.excursion_probability(algo.u_current, zpm, zpv)
    xp = _flat(algo.smc.particles.x)
    logpx = _flat(algo.smc.particles.logpx)
    particle_height = np.exp(logpx - np.nanmax(logpx))

    _plot_prediction_diagnostics_1d(
        xt,
        zt,
        algo.xi,
        algo.zi,
        zpm,
        zpv,
        crit,
        pe,
        "example31_excursion_bss.png",
        "BSS particles at an intermediate threshold",
        threshold=algo.u_current,
        particles_x=xp,
        particles_height=particle_height,
        probability_label="relative target",
    )


def _set_inversion_problem():
    return gpc.ComputerExperiment(
        1,
        [[-2.0], [2.0]],
        single_function=_excursion_test_function,
    )


def example40_set_inversion_grid():
    _seed(140)
    problem = _set_inversion_problem()
    nt = 700
    xt = gp.misc.designs.regulargrid(problem.input_dim, nt, problem.input_box)
    zt = problem(xt)
    xi = xt[[140, 322, 420, 560]]
    box_target = gnp.array([[0.90], [1.10]])

    model = gpc.Model_ConstantMean_Maternp_REMAP(
        "GP1d",
        output_dim=problem.output_dim,
        mean_specification={"type": "constant"},
        covariance_specification={"p": 2},
    )
    algo = si.SetInversionGridSearch(
        problem,
        model,
        xt,
        box_target,
        options={"beta": 1.0},
    )
    _quiet_call(algo.set_initial_design, xi)
    for _ in range(5):
        _quiet_call(algo.step)

    zpm, zpv = algo.predict(xt, convert_out=False)
    crit, _ = sampcrit.box_wMSE(algo.box_target, zpm, zpv, beta=1.0)
    pe, _ = sampcrit.box_probability(algo.box_target, zpm, zpv)
    _plot_prediction_diagnostics_1d(
        xt,
        zt,
        algo.xi,
        algo.zi,
        zpm,
        zpv,
        crit,
        pe,
        "example40_set_inversion_grid.png",
        "Set inversion on a fixed grid",
        box=box_target,
        probability_label="box probability",
    )


def example41_set_inversion_bss():
    _seed(141)
    problem = _set_inversion_problem()
    nt = 700
    xt = gp.misc.designs.regulargrid(problem.input_dim, nt, problem.input_box)
    zt = problem(xt)
    xi = xt[[140, 322, 420, 560]]
    box_init = gnp.array([[0.0], [1.2]])
    box_target = gnp.array([[0.99], [1.02]])

    model = gpc.Model_ConstantMean_Maternp_REML(
        "GP1d",
        output_dim=problem.output_dim,
        mean_specification={"type": "constant"},
        covariance_specification={"p": 2},
    )
    algo = si.SetInversionBSS(
        problem,
        model,
        box_init,
        box_target,
        options={"n_smc": 160, "update_search_space_at_init": False, "beta": 1.0},
    )
    _quiet_call(algo.set_initial_design, xi)
    _quiet_call(algo.step_move_particles_with_mu, 0.85)

    zpm, zpv = algo.predict(xt, convert_out=False)
    crit, _ = sampcrit.box_wMSE(algo.box_current, zpm, zpv, beta=1.0)
    pe, _ = sampcrit.box_probability(algo.box_current, zpm, zpv)
    xp = _flat(algo.smc.particles.x)
    logpx = _flat(algo.smc.particles.logpx)
    particle_height = np.exp(logpx - np.nanmax(logpx))

    _plot_prediction_diagnostics_1d(
        xt,
        zt,
        algo.xi,
        algo.zi,
        zpm,
        zpv,
        crit,
        pe,
        "example41_set_inversion_bss.png",
        "BSS set inversion at an intermediate box",
        box=algo.box_current,
        particles_x=xp,
        particles_height=particle_height,
        probability_label="relative target",
    )


def main():
    example01_branin()
    getting_started_hartmann4_prediction()
    guide_hartmann4_remap_prior()
    example02_gp_prediction()
    example04_noisy_sequential()
    example10_ei_grid()
    example11_ei_smc()
    example20_regp()
    example30_excursion_grid()
    example31_excursion_bss()
    example40_set_inversion_grid()
    example41_set_inversion_bss()


if __name__ == "__main__":
    main()
