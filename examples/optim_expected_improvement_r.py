import numpy as np
import matplotlib.pyplot as plt
import gpmp as gp
import gpmpcontrib.optim.expectedimprovement_r as ei_r
from gpmpcontrib.optim.expectedimprovement import AbortException
import lhsmdu
import sys
import os
import gpmpcontrib.optim.test_problems as test_problems

# -- Settings
plot = False

# Default values and types for different options
env_options = {
    "OUTPUT_DIR": ("output", str),
    "N_ITERATIONS": (300, int),
    "SLURM_ARRAY_TASK_ID": (None, int),
    "N_RUNS": (1, int),
    "PROBLEM": ("goldsteinprice", str),
    "N0_OVER_D": (3, int),
    "STRATEGY": ("Constant", str),
    "Q_STRATEGY": (0.25, float),
    "CRIT_OPT_METHOD": ("L-BFGS-B", str),
    "RELAXED_INIT": ("flat", str),
    "FTOL": (1e-14, float),
    "GTOL": (1e-15, float),
    "EPS": (None, float),
    "MAXFUN": (1000, int),
    "MAXITER": (1000, int),
    "N_SMC": (1000, int),
}

# Initialize options and crit_optim_options
options = {}
crit_optim_options = {}

# Loop through the environment options
for key, (default, value_type) in env_options.items():
    value = os.getenv(key, default)
    if value is not None:
        if key == "CRIT_OPT_METHOD":
            # Add to crit_optim_options
            crit_optim_options["method"] = value_type(value)
        elif key in [
            "RELAXED_INIT",
            "FTOL",
            "GTOL",
            "EPS",
            "MAXFUN",
            "MAXITER",
        ]:
            # Add to crit_optim_options
            crit_optim_options[key.lower()] = value_type(value)
        elif key == "SLURM_ARRAY_TASK_ID" and value is not None:
            idx_run_list = [value_type(value)]
        elif key == "N_RUNS" and "SLURM_ARRAY_TASK_ID" not in os.environ:
            idx_run_list = list(range(value_type(value)))
        elif key == "PROBLEM":
            problem = getattr(test_problems, value)
        elif key == "STRATEGY":
            # Check if Q_STRATEGY is set, use its value if available, otherwise use default
            q_strategy_value = float(
                os.getenv("Q_STRATEGY", env_options["Q_STRATEGY"][0])
            )
            options["threshold_strategy"] = ei_r.threshold_strategy[value](q_strategy_value)
        elif key == "Q_STRATEGY":
            continue  # Handled with STRATEGY
        else:
            # Add to options directly
            options[key.lower()] = value_type(value)

# Set crit_optim_options in options
if crit_optim_options:
    options["crit_optim_options"] = crit_optim_options

# -- Initialize records for storing results
history_records = []
xi_records = []

# -- Create initial dataset and run optimization
for i in idx_run_list:
    # Generate initial design points using Latin Hypercube Sampling
    ni0 = options["n0_over_d"] * problem.input_dim
    xi = gp.misc.designs.scale(
        np.array(lhsmdu.sample(problem.input_dim, ni0, randomSeed=None).T),
        problem.input_box,
    )

    # Initialize the Expected Improvement algorithm
    eialgo = ei_r.ExpectedImprovementR(problem, options=options)
    eialgo.set_initial_design(xi=xi)

    # Plot initial state if enabled
    if plot:
        plt.figure()
        plt.plot(eialgo.zi, eialgo.zi_relaxed, "o")
        plt.axhline(np.quantile(eialgo.zi, 0.25), color="b", label="t0")
        plt.semilogy()
        plt.semilogx()
        plt.xlabel("Truth")
        plt.ylabel("Relaxed")
        plt.legend()
        plt.show()

    # Perform optimization steps
    for _ in range(options["n_iterations"]):
        if plot:
            plt.figure()
            plt.plot(eialgo.xi[:, 0], eialgo.xi[:, 1], "go")
            plt.plot(eialgo.smc.x[:, 0], eialgo.smc.x[:, 1], "bo", markersize=3)

        # Run a step of the algorithm
        try:
            eialgo.step()
        except AbortException as e:
            print("Aborting: {}".format(e))
            break

        # Plot current state if enabled
        if plot:
            plt.plot(eialgo.xi[-1, 0], eialgo.xi[-1, 1], "ko")
            plt.show()
            plt.figure()
            plt.plot(eialgo.zi, eialgo.zi_relaxed, "o")
            plt.axhline(np.quantile(eialgo.zi, 0.25), color="b", label="t0")
            plt.semilogy()
            plt.semilogx()
            plt.xlabel("Truth")
            plt.ylabel("Relaxed")
            plt.legend()
            plt.show()

    # Store the history of observations
    history_records.append(eialgo.zi)
    xi_records.append(eialgo.xi)

    # Prepare output directory
    i_output_dir = os.path.join(options["output_dir"], str(i))
    if not os.path.exists(i_output_dir):
        os.makedirs(i_output_dir)

    # Save data
    np.save(os.path.join(i_output_dir, "data.npy"), np.hstack((eialgo.xi, eialgo.zi)))

# Plot final results if enabled
if plot:
    for i in range(len(history_records)):
        plt.figure()
        plt.plot(
            np.minimum.accumulate(history_records[i]),
            label="best observation so far",
            color="blue",
        )
        plt.plot(history_records[i], label="observation", color="green")
        plt.axhline(3, color="r", label="min")
        plt.legend()
        plt.xlabel("Iterations")
        plt.ylabel("GoldsteinPrice")
        plt.semilogy()
        plt.show()
