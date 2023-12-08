# --------------------------------------------------------------
# Author: Emmanuel Vazquez <emmanuel.vazquez@centralesupelec.fr>
# Copyright (c) 2022-2023, CentraleSupelec
# License: GPLv3 (see LICENSE)
# --------------------------------------------------------------
import gpmp.num as gnp


def isinbox(box, x):
    b = gnp.logical_and(gnp.all(x >= box[0], axis=1),
                        gnp.all(x <= box[1], axis=1))
    return b


def probability_excursion(t, zpm, zpv):
    '''Computes the probabilities of exceeding the threshold t for
    Gaussian predictive distributions with means zpm and variances
    zpv. The input argument must have the following sizes:

        * zpm        M x 1,
        * zpv        M x 1,
 
     where M is the number of points where the EI must be
     computed. The output has size M x 1.
    '''
    p = gnp.empty(zpm.shape)
    delta = zpm - t
    sigma = gnp.sqrt(zpv)
    b = sigma > 0

    # Where sigma > 0
    u = gnp.where(b, delta / sigma, 0)  # Avoid division by zero
    p = gnp.where(b, gnp.normal.cdf(u), 0)  # Compute p where sigma > 0

    # Condition where sigma == 0
    p = gnp.where(b, p, delta > 0)
    
    # # Compute p where sigma > 0
    # u = delta[b] / sigma[b]
    # p[b] = gnp.normal.cdf(u)

    # # Compute p where sigma == 0
    # b = gnp.logical_not(b)
    # p[b] = gnp.asdouble(delta[b] > 0)

    return p

def probability_box(box, zpm, zpv):
    '''Computes the probability to be in a box
    '''
    dim_output = zpm.shape[1]

    pn = gnp.empty(zpm.shape)
    for j in range(dim_output):

        delta_min = box[0][j] - zpm[:, j] 
        delta_max = box[1][j] - zpm[:, j]
        sigma = gnp.sqrt(zpv)
        b = sigma > 0

        # Compute pn where sigma > 0
        u_min = gnp.where(b, delta_min / sigma, 0)
        u_max = gnp.where(b, delta_max / sigma, 0)
        pn = gnp.where(b, normal.cdf(u_max) - normal.cdf(u_min), 0)

        # Compute pn where sigma == 0
        pn = gnp.where(b, pn, delta_max > 0 & delta_min < 0)

        return pn

def expected_improvement(t, zpm, zpv):
    '''Computes the Expected Improvement (EI) criterion for a
     maximization problem given a threshold t and Gaussian predictive
     distributions with means zpm and variances zpv. The input
     argument must have the following sizes:
 
        * zpm        M x 1,
        * zpv        M x 1,
 
     where M is the number of points where the EI must be
     computed. The output has size M x 1.
 
     REFERENCES
 
    [1] D. R. Jones, M. Schonlau and William J. Welch. Efficient global
        optimization of expensive black-box functions.  Journal of Global
        Optimization, 13(4):455-492, 1998.
 
    [2] J. Mockus, V. Tiesis and A. Zilinskas. The application of Bayesian
        methods for seeking the extremum. In L.C.W. Dixon and G.P. Szego,
        editors, Towards Global Optimization, volume 2, pages 117-129, North
        Holland, New York, 1978.

    '''
    ei = gnp.empty(zpm.shape)
    delta = zpm - t
    sigma = gnp.sqrt(zpv)
    b = sigma > 0

    # Compute the EI where sigma > 0
    u = gnp.where(b, delta / sigma, 0)
    ei = gnp.where(b, sigma * (gnp.normal.pdf(u) + u * gnp.normal.cdf(u)), 0)

    # Compute the EI where sigma == 0
    ei = gnp.where(b, ei, gnp.maximum(0, delta))

    # Correct numerical inaccuracies
    ei = gnp.where(ei > 0, ei, 0)

    return ei
