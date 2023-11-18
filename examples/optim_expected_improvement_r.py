import numpy as np
import matplotlib.pyplot as plt
import gpmp as gp
import gpmpcontrib.optim.expectedimprovement_r as ei_r
from gpmpcontrib.optim.expectedimprovement import AbortException
import lhsmdu
import sys
import os
import gpmpcontrib.optim.test_problems as test_problems

# # Detect if running in interactive mode
# if len(sys.argv) < 4:
#     # Interactive Mode: Prompt user for inputs
#     output_dir = "output"
#     n_runs = 100
#     n_iterations = 20
#     i_range_input = "1,2"
#     if i_range_input:
#         idx_run_list = [int(i) for i in i_range_input.split(',')]
#     else:
#         idx_run_list = list(range(n_repeat))
# else:
#     # Command Line Mode: Use arguments from command line
#     output_dir = sys.argv[1]
#     n_runs = int(sys.argv[2])
#     n_iterations = int(sys.argv[3])
#     if len(sys.argv) > 4:
#         idx_run_list = [int(_tmp) for _tmp in sys.argv[4:]]
#     else:
#         idx_run_list = list(range(n_repeat))

# -- Settings

# Output directory
if "OUTPUT_DIR" in os.environ:
    output_dir = os.environ["OUTPUT_DIR"]
else:
    raise RuntimeError('The environment variable "OUTPUT_DIR" must be set.')

# n_iterations
if "N_ITERATIONS" in os.environ:
    n_iterations = int(os.environ["N_ITERATIONS"])
else:
    raise RuntimeError('The environment variable "N_ITER" must be set.')

# runs
assert ("N_RUN" in os.environ) != ("IDX_RUN" in os.environ), 'One and only one of the environment variables "N_RUN" ' \
                                                             'and "IDX_RUN" must be set.'

if "N_RUNS" in os.environ:
    idx_run_list = list(range(int(os.environ["N_RUNS"])))
if "IDX_RUN" in os.environ:
    idx_run_list = [int(os.environ["IDX_RUN"])]

# Define the optimization problem
if "PROBLEM" in os.environ:
    problem_name = os.environ["PROBLEM"]
else:
    raise RuntimeError('The environment variable "PROBLEM" must be set.')

problem = getattr(test_problems, problem_name)

# Define the optimization strategy
if "STRATEGY" in os.environ:
    strategy = os.environ["STRATEGY"]
else:
    raise RuntimeError("Set the STRATEGY environment variable.")

# Strategy quantile level
q_strategy = 0.25

# Criterion optimization options
crit_optim_options = {}

if "CRIT_OPT_METHOD" in os.environ:
    crit_optim_options['method'] = os.environ["CRIT_OPT_METHOD"]

if "RELAXED_INIT" in os.environ:
    crit_optim_options["relaxed_init"] = os.environ["RELAXED_INIT"]

# See gpmp.kernel
if "FTOL" in os.environ:
    crit_optim_options["ftol"] = float(os.environ["ftol"])

if "GTOL" in os.environ:
    crit_optim_options["gtol"] = float(os.environ["gtol"])

if "EPS" in os.environ:
    crit_optim_options["eps"] = float(os.environ["eps"])

if "MAXFUN" in os.environ:
    crit_optim_options["maxfun"] = int(os.environ["maxfun"])

if "MAXITER" in os.environ:
    crit_optim_options["maxiter"] = int(os.environ["maxiter"])

# Enable or disable plotting
plot = False

# -- Initialize records for storing results
history_records = []
xi_records = []

# -- Create initial dataset and run optimization
for i in idx_run_list:
    # Generate initial design points using Latin Hypercube Sampling
    ni0 = 3 * problem.input_dim
    xi = gp.misc.designs.scale(np.array(lhsmdu.sample(problem.input_dim, ni0, randomSeed=None).T, problem.input_box))

    # Initialize the Expected Improvement algorithm
    eialgo = ei_r.ExpectedImprovementR(
        problem,
        options={
            't_getter': ei_r.t_getters[strategy](q_strategy),
            'crit_optim_options': crit_optim_options,
        }
    )
    eialgo.set_initial_design(xi=xi)

    # Plot initial state if enabled
    if plot:
        plt.figure()
        plt.plot(eialgo.zi, eialgo.zi_relaxed, 'o')
        plt.axhline(np.quantile(eialgo.zi, 0.25), color='b', label='t0')
        plt.semilogy()
        plt.semilogx()
        plt.xlabel("Truth")
        plt.ylabel("Relaxed")
        plt.legend()
        plt.show()

    # Perform optimization steps
    for _ in range(n_iterations):
        if plot:
            plt.figure()
            plt.plot(eialgo.xi[:, 0], eialgo.xi[:, 1], 'go')
            plt.plot(eialgo.smc.x[:, 0], eialgo.smc.x[:, 1], 'bo', markersize=3)

        # Run a step of the algorithm
        try:
            eialgo.step()
        except AbortException:
            break

        # Plot current state if enabled
        if plot:
            plt.plot(eialgo.xi[-1, 0], eialgo.xi[-1, 1], 'ko')
            plt.show()
            plt.figure()
            plt.plot(eialgo.zi, eialgo.zi_relaxed, 'o')
            plt.axhline(np.quantile(eialgo.zi, 0.25), color='b', label='t0')
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
    i_output_dir = os.path.join(output_dir, str(i))
    if not os.path.exists(i_output_dir):
        os.makedirs(i_output_dir)

    # Save data
    np.save(os.path.join(i_output_dir, 'data.npy'), np.hstack((eialgo.xi, eialgo.zi)))

# Plot final results if enabled
if plot:
    for i in range(len(history_records)):
        plt.figure()
        plt.plot(np.minimum.accumulate(history_records[i]), label='best observation so far', color='blue')
        plt.plot(history_records[i], label='observation', color='green')
        plt.axhline(3, color='r', label='min')
        plt.legend()
        plt.xlabel("Iterations")
        plt.ylabel('GoldsteinPrice')
        plt.semilogy()
        plt.show()
