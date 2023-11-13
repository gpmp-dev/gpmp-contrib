import numpy as np
import matplotlib.pyplot as plt
import gpmp as gp
import gpmpcontrib.optim.expectedimprovement_r as ei_r
from gpmpcontrib.optim.expectedimprovement import AbortException
import lhsmdu
import scipy.special
import torch
import sys
import os

from gpmpcontrib.optim.test_problems import goldsteinprice

output_dir = sys.argv[1]

n_repeat = int(sys.argv[2])
n_run = int(sys.argv[3])

if len(sys.argv) > 4:
    i_range = [int(_tmp) for _tmp in sys.argv[4:]]
else:
    i_range = list(range(n_repeat))

## -- settings

problem = goldsteinprice

strategy = "Concentration"

plot = False

## init records

history_records = []
xi_records = []

## -- create initial dataset

for i in i_range:
    xi = -2 + 4 * np.array(lhsmdu.sample(2, 6, randomSeed=None).T)

    ## -- initialize the ei algorithm

    eialgo = ei_r.ExpectedImprovementR(problem, options={'t_getter': ei_r.t_getters[strategy](0.25)})

    eialgo.set_initial_design(xi=xi)

    if plot:
        plt.figure()
        plt.plot(eialgo.zi, eialgo.zi_relaxed, 'o')
        # plt.axhline(custom_records[0][i]['p0'][-1], color='r', label='init')
        plt.axhline(np.quantile(eialgo.zi, 0.25), color='b', label='t0')
        plt.semilogy()
        plt.semilogx()
        plt.xlabel("Truth")
        plt.ylabel("Relaxed")
        plt.legend()
        plt.show()

    # make n new evaluations
    for _ in range(n_run):
        if plot:
            plt.figure()

            plt.plot(eialgo.xi[:, 0], eialgo.xi[:, 1], 'go')

            plt.plot(eialgo.smc.x[:, 0], eialgo.smc.x[:, 1], 'bo', markersize=3)

        try:
            eialgo.step()
        except AbortException:
            break

        if plot:
            plt.plot(eialgo.xi[-1, 0], eialgo.xi[-1, 1], 'ko')

            plt.show()

            plt.figure()
            plt.plot(eialgo.zi, eialgo.zi_relaxed, 'o')
            # plt.axhline(custom_records[0][i]['p0'][-1], color='r', label='init')
            plt.axhline(np.quantile(eialgo.zi, 0.25), color='b', label='t0')
            plt.semilogy()
            plt.semilogx()
            plt.xlabel("Truth")
            plt.ylabel("Relaxed")
            plt.legend()
            plt.show()

    history_records.append(eialgo.zi)

    xi_records.append(eialgo.xi)

    i_output_dir = os.path.join(output_dir, str(i))

    if not os.path.exists(i_output_dir):
        os.makedirs(i_output_dir)

    np.save(os.path.join(i_output_dir, 'data.npy'), np.hstack((eialgo.xi , eialgo.zi)))

if plot:
    # Plot histories
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