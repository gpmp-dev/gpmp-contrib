'''Implement a sketch of the EI algorithm

Author: Emmanuel Vazquez <emmanuel.vazquez@centralesupelec.fr>
Copyright (c) 2022-2023, CentraleSupelec
License: GPLv3 (see LICENSE)

'''
import numpy as np
import matplotlib.pyplot as plt
import gpmp as gp
import gpmpcontrib.optim.expectedimprovement as ei
import gpmpcontrib.samplingcriteria as sampcrit
import gpmpcontrib.computerexperiment as cptexp

## -- definition of a mono-objective problem

problem = cptexp.ComputerExperiment(
    1,                                              # dim of search space
    [[-1], [1]],                                    # search box
    single_function=gp.misc.testfunctions.twobumps  # test function
)

# objectives=[{'function': tf.f1, 'goal': 'minimize'}

## -- create initial dataset

nt = 2000
xt = gp.misc.designs.regulargrid(problem.input_dim, nt, problem.input_box)
zt = gp.misc.testfunctions.twobumps(xt)

ni = 3
ind = [100, 1000, 1600]
xi = xt[ind]

## -- initialize the ei algorithm

eialgo = ei.ExpectedImprovement(problem)

eialgo.set_initial_design(xi)

## -- visualization


def plot(show=True, x=None, z=None):
    
    zpm, zpv = eialgo.predict(xt)
    ei = sampcrit.expected_improvement(-np.min(eialgo.zi), -zpm, zpv)
    pe = sampcrit.probability_excursion(-np.min(eialgo.zi), -zpm, zpv)

    fig = gp.misc.plotutils.Figure(nrows=3, ncols=1, isinteractive=True)
    fig.subplot(1)
    fig.plot(xt, zt, 'k', linewidth=0.5)
    if z is not None:
        fig.plot(x, z, 'b', linewidth=0.5)
    fig.plotdata(eialgo.xi, eialgo.zi)
    fig.plotgp(xt, zpm, zpv, colorscheme='simple')
    fig.ylabel('$z$')
    fig.title(f'Posterior GP, ni={eialgo.xi.shape[0]}')
    fig.subplot(2)
    fig.plot(xt, -ei, 'k', linewidth=0.5)
    fig.ylabel('EI')
    fig.subplot(3)
    fig.plot(xt, pe, 'k', linewidth=0.5)
    fig.plot(eialgo.smc.x, np.zeros(eialgo.smc.n), '.')
    fig.ylabel('Prob. excursion')
    fig.xlabel('x')
    if show:
        fig.show()

    return fig


plot()

# make n = 3 new evaluations
n = 5
for i in range(n):
    eialgo.step()
    plot(show=True)

# print model diagnosis
gp.misc.modeldiagnosis.diag(eialgo.models[0]['model'],
                            eialgo.models[0]['info'],
                            eialgo.xi,
                            eialgo.zi)
