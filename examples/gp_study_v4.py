import math, time, warnings
import numpy as np
import gpmp.num as gnp
import gpmp as gp
import gpmp.misc.plotutils as plotutils

dim_act = 5
dim_inact = 45
dim_tot = dim_act + dim_inact
ni_list = [50, 100, 150, 200, 250, 300, 400, 500, 600, 800, 1000]
nt = 500  # test sample size remains fixed

# Model settings

def kernel(x, y, covparam, pairwise=False):
    p = 2
    return gp.kernel.maternp_covariance(x, y, p, covparam, pairwise)


def constant_mean(x, param):
    return gnp.ones((x.shape[0], 1))

# Start experiments
seed = 123
np.random.seed(seed)
results = []
exp_count = 0
total_exps = len(ni_list)

print("Starting experiments: {} total configurations.".format(total_exps))
for ni in ni_list:
    exp_count += 1
    print("Experiment {}/{}: dim_tot={}, dim_act={}, dim_inact={}, ni={}".format(
        exp_count, total_exps, dim_tot, dim_act, dim_inact, ni))

    timings = {}  # dictionary to store elapsed times for each part

    # 1. Generate designs for active dimensions.
    t0 = time.time()
    box_act = gnp.array([[0.0] * dim_act, [1.0] * dim_act])
    xi_act = gnp.asarray(gp.misc.designs.randunif(dim_act, ni, box_act))
    xt_act = gnp.asarray(gp.misc.designs.randunif(dim_act, nt, box_act))
    timings["design_active"] = time.time() - t0

    # 2. Generate designs for inactive dimensions (if any).
    t0 = time.time()
    if dim_inact > 0:
        box_inact = gnp.array([[0.0] * dim_inact, [1.0] * dim_inact])
        xi_inact = gnp.asarray(gp.misc.designs.randunif(dim_inact, ni, box_inact))
        xt_inact = gnp.asarray(gp.misc.designs.randunif(dim_inact, nt, box_inact))
    else:
        xi_inact = gnp.empty((ni, 0))
        xt_inact = gnp.empty((nt, 0))
    timings["design_inactive"] = time.time() - t0

    # 3. Combine to get full designs (dimension = total_dim).
    xi = gnp.hstack((xi_act, xi_inact))
    xt = gnp.hstack((xt_act, xt_inact))

    # 4. Build a GP model with fixed covariance parameters.
    t0 = time.time()
    logsigma = math.log(0.5**2)
    # rho = 0.9
    delta = 1.0
    rho = gnp.exp(gnp.gammaln(dim_act / 2 + 1) / dim_act) / (gnp.pi**0.5) * delta
    loginvrho = -math.log(rho)
    covparam = gnp.array([logsigma, loginvrho])
    model = gp.core.Model(constant_mean, kernel, None, covparam)
    timings["model_initialization"] = time.time() - t0

    # 5. Generate a GP sample path using the active design only (for reproducibility).
    t0 = time.time()
    xixt_act = np.vstack((xi_act, xt_act))
    n_samplepaths = 1
    zsim = model.sample_paths(xixt_act, n_samplepaths, method="chol")
    zi = zsim[:ni, 0]
    zt = zsim[ni:, 0]
    timings["sample_path"] = time.time() - t0

    # 6. Update model parameters via parameter selection (remapping).
    t0 = time.time()
    model_remap, info = gp.kernel.select_parameters_with_remap(model, xi, zi, info=True)
    timings["parameter_selection"] = time.time() - t0
    print("  Parameter selection complete.")

    # 7. Compute performance metrics (e.g., leave-one-out, test set).
    t0 = time.time()
    model_perf = gp.modeldiagnosis.compute_performance(
        model_remap, xi, zi, loo=True, loo_res=None, xtzt=(xt, zt), zpmzpv=None
    )
    timings["performance_metrics"] = time.time() - t0
    print("  Performance metrics computed.")

    # # 8. Compute selection criterion statistics.
    # t0 = time.time()
    # # Here we assume an integration space of size 1 + total_dim.
    # param_box_bounds = [[-9] * (1 + total_dim), [9] * (1 + total_dim)]
    # # Use all indices in the integration vector; adjust if needed.
    # ind = list(range(1 + total_dim))
    # sel_crit_stats = gp.modeldiagnosis.selection_criterion_statistics_fast(
    #     info, model_remap, xi, ind=ind, param_box=param_box_bounds, n_points=1000, verbose=False
    # )
    # timings["sel_crit_stats"] = time.time() - t0
    # print("  Selection criterion statistics computed.\n")

    exp_result = {
        "total_dim": dim_tot,
        "dim_act": dim_act,
        "dim_inact": dim_inact,
        "ni": ni,
        "covparam": model_remap.covparam,
        "model_perf": model_perf,
        #      "sel_crit_stats": sel_crit_stats["parameter_statistics"],
        "timings": timings
    }
    results.append(exp_result)

for res in results:
    print("Total dim:", res["total_dim"],
          "Active:", res["dim_act"],
          "Inactive:", res["dim_inact"],
          "ni:", res["ni"])
    print("LOO tss:", res["model_perf"]['data_tss'])
    print("LOO press:", res["model_perf"]['loo_press'])
    print("LOO Q2:", res["model_perf"]['loo_Q2'])
    print("Test R2:", res["model_perf"]['test_R2'])
    # print("Selection criterion statistics:")
    # print(res["sel_crit_stats"])
    # print("Timings:", res["timings"], "\n")


covparam = gnp.empty((total_exps, 1+dim_tot))
for i, res in enumerate(results):
    covparam[i, :] = res["covparam"]
