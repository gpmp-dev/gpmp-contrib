# Importing necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import gpmp as gp
import gpmpcontrib.optim.expectedimprovement_r as ei_r
from gpmpcontrib.optim.expectedimprovement import AbortException
import lhsmdu
import scipy.special
import sys
import os

# Test problem
from gpmpcontrib.optim.test_problems import goldsteinprice

# Detect if running in interactive mode
if len(sys.argv) < 4:
    # Interactive Mode: Prompt user for inputs
    output_dir = "output"
    n_repeat = 100
    n_run = 20
    i_range_input = "1,2"
    if i_range_input:
        i_range = [int(i) for i in i_range_input.split(',')]
    else:
        i_range = list(range(n_repeat))
else:
    # Command Line Mode: Use arguments from command line
    output_dir = sys.argv[1]
    n_repeat = int(sys.argv[2])
    n_run = int(sys.argv[3])
    if len(sys.argv) > 4:
        i_range = [int(_tmp) for _tmp in sys.argv[4:]]
    else:
        i_range = list(range(n_repeat))

# -- Settings

# Define the optimization problem
problem = goldsteinprice

# Define the optimization strategy
strategy = "Concentration"

# Enable or disable plotting
plot = False

# -- Initialize records for storing results
history_records = []
xi_records = []

# -- Create initial dataset and run optimization
for i in i_range:
    # Generate initial design points using Latin Hypercube Sampling
    ni0 = 6
    xi = gp.misc.designs.scale(np.array(lhsmdu.sample(problem.input_dim, ni0, randomSeed=None).T, problem.input_box)

    # Initialize the Expected Improvement algorithm
    eialgo = ei_r.ExpectedImprovementR(problem, options={'t_getter': ei_r.t_getters[strategy](0.25)})
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
    for _ in range(n_run):
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
