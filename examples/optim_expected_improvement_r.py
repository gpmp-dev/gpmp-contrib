import numpy as np
import matplotlib.pyplot as plt
import gpmp as gp
import gpmpcontrib.optim.expectedimprovement_r as ei_r

from gpmpcontrib.optim.test_problems import goldsteinprice

plot = True

## -- definition of a mono-objective problem

problem = goldsteinprice

## -- create initial dataset

nt = 2000
xt = gp.misc.designs.regulargrid(problem.input_dim, nt, problem.input_box)
zt = goldsteinprice.eval(xt)

#TODO:() Use LHS
xi = -2 + 4 * np.random.uniform(size=(6, 2))

## -- initialize the ei algorithm

eialgo = ei_r.ExpectedImprovementR(problem, options={'t_getter': ei_r.t_getters['Concentration'](0.25)})

eialgo.set_initial_design(xi)

# make n new evaluations
n = 100
for _ in range(n):
    if plot:
        plt.figure()

        plt.plot(eialgo.xi[:, 0], eialgo.xi[:, 1], 'go')

        plt.plot(eialgo.smc.x[:, 0], eialgo.smc.x[:, 1], 'bo', markersize=3)

    eialgo.step()

    if plot:
        plt.plot(eialgo.xi[-1, 0], eialgo.xi[-1, 1], 'ko')

        plt.show()

# Visualize results
plt.figure()

plt.plot(np.minimum.accumulate(eialgo.zi), label='best observation so far', color='blue')
plt.plot(eialgo.zi, label='observation', color='green')

plt.axhline(3, color='r', label='min')
plt.legend()
plt.xlabel("Iterations")
plt.ylabel('GoldsteinPrice')
plt.semilogy()

plt.show()