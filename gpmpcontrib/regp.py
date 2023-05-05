from scipy.stats import norm
import time
import numpy as np
import gpmp.num as gnp
import gpmp as gp


def get_membership_indices(zi, R):
    """
    Get the index of the membership of each row in zi to an interval in R.

    Parameters
    ----------
    zi : ndarray
        An (n, 1) shaped array of data points.
    R : list of lists
        A list of intervals represented as [l_i, u_i] pairs.

    Returns
    -------
    ei : ndarray
        An (n, 1) shaped array of membership indices.
    """

    n = zi.shape[0]
    ei = np.zeros(n, dtype=int)

    for idx, (l, u) in enumerate(R, start=1):
        mask = np.logical_and(zi > l, zi <= u)
        c = idx * mask
        ei = ei + c

    return ei


def split_data(xi, zi, ei, R):
    """Split the data based on the membership indices.

    Parameters
    ----------
    xi : ndarray
        An (n, d) shaped array of data points.
    zi : ndarray
        An (n, 1) shaped array of data points.
    ei : ndarray
        An (n, 1) shaped array of membership indices.
    R : list of intervals
        List of intervals, each specified as [l_k, u_k].

    Returns
    -------
    (x0, z0, ind0) : tuple of ndarrays
        A tuple of (n0, d), (n0, 1) shaped arrays for xi and zi rows
        with ei = 0, and the corresponding indices.
    (x1, z1, bounds, ind1) : tuple of ndarrays
        Rows of xi and zi with ei > 0, and the corresponding interval
        bounds and indices.

    """

    mask = ei.reshape(-1) == 0
    ind0 = np.where(mask)[0]
    ind1 = np.where(~mask)[0]

    x0, z0 = xi[mask], zi[mask]

    x1, z1, e1 = xi[~mask], zi[~mask], ei[~mask]

    interval_indices = e1 - 1
    bounds = [R[i] for i in interval_indices]

    return (x0, z0, ind0), (x1, z1, bounds, ind1)


def make_regp_criterion_with_gradient(model, x0, z0, x1):
    """
    Make regp criterion function with gradient.

    Parameters
    ----------
    model : gpmp model
        Gaussian process model.
    x0 : ndarray, shape (n0, d)
        Locations of the observed data points not relaxed
    z0 : ndarray, shape (n0,)
        Observed values at the data points not relaxed
    x1 : ndarray, shape (n1, d)
        Locations of the relaxed  data points

    Returns
    -------
    crit_jit : function
        Selection criterion function with gradient.
    dcrit : function
        Gradient of the selection criterion function.
    """
    x0 = gnp.asarray(x0)
    x1 = gnp.asarray(x1)
    z0 = gnp.asarray(z0)

    xi = gnp.vstack((x0, x1))

    n1 = x1.shape[0]

    # selection criterion

    selection_criterion = model.negative_log_restricted_likelihood

    def crit_(param):

        covparam = param[0:-n1]
        z1 = param[-n1:]
        zi = gnp.concatenate((z0, z1))
        l = selection_criterion(covparam, xi, zi)
        return l

    crit_jit = gnp.jax.jit(crit_)

    dcrit = gnp.jax.jit(gnp.grad(crit_jit))

    return crit_jit, dcrit


def remodel(model, xi, zi, R, covparam0=None, info=False, verbosity=0):
    """
    Perform reGP optimization (REML + relaxation)

    Parameters
    ----------
    model : GPModel
        Gaussian process model.
    xi : ndarray, shape (n, d)
        Locations of the observed data points.
    zi : ndarray, shape (n,)
        Observed values at the data points.
    R : list of intervals
        List of relaxation intervals, each specified as [l_k, u_k].
    covparam0 : ndarray, optional
        Initial guess for the covariance parameters.
    info : bool, optional
        Whether to return additional information.
    verbosity : int, optional
        Verbosity level.

    Returns
    -------
    model : GPmp model
        Updated GPmp Gaussian process model.
    zi_relaxed : ndarray, shape (n,)
        Relaxed output data.
    ind_relaxed : ndarray
        Indices of the relaxed data points in the input zi array.
    info_ret : dict, optional
        Additional information (if info=True).
    """

    tic = time.time()

    # Initial guess for the covariance parameters if not provided
    if covparam0 is None:
        covparam0 = gp.kernel.anisotropic_parameters_initial_guess(model, xi, zi)
    covparam_dim = covparam0.shape[0]
    covparam_bounds = [gnp.array([-gnp.inf, gnp.inf])] * covparam0.shape[0]

    # Membership indices and split data
    ei = get_membership_indices(gnp.to_np(zi), R)
    (x0, z0, ind0), (x1, z1, z1_bounds, ind1) = split_data(xi, gnp.to_np(zi), ei, R)
    z1_size = z1.shape[0]

    # Initial parameter vector and bounds
    p0 = np.concatenate((covparam0, z1.reshape(-1)))

    bounds = covparam_bounds + z1_bounds

    # reGP criterion
    nlrl, dnlrl = make_regp_criterion_with_gradient(model, x0, z0, x1)

    # Verbosity level
    silent = True
    if verbosity == 1:
        print("reGP stuff...")
    elif verbosity == 2:
        silent = False

    # Optimize parameters
    popt, info_ret = gp.kernel.autoselect_parameters(
        p0, nlrl, dnlrl, bounds=bounds, silent=silent, info=True
    )

    if verbosity == 1:
        print("done.")

    # Update the model and relaxed data
    model.covparam = gnp.asarray(popt[0:covparam_dim])

    z1_relaxed = popt[covparam_dim:]

    zi_relaxed = gnp.zeros(zi.shape)
    if gnp._gpmp_backend_ == 'jax':
        zi_relaxed = zi_relaxed.at[ind0].set(gnp.asarray(z0))
        zi_relaxed = zi_relaxed.at[ind1].set(gnp.asarray(z1_relaxed))
    else:
        zi_relaxed[ind0] = gnp.asarray(z0)
        zi_relaxed[ind1] = gnp.asarray(z1_relaxed)
    
    # Return results
    if info:
        info_ret["covparam0"] = covparam0
        info_ret["covparam"] = model.covparam
        info_ret["selection_criterion"] = nlrl
        info_ret["time"] = time.time() - tic
        return model, zi_relaxed, ind1, info_ret
    else:
        return model, zi_relaxed, ind1


def predict(model, xi, zi, xt, R, covparam0=None, info=False, verbosity=0):
    """
    Perform reGP optimization (REML + relaxation) and prediction

    Parameters
    ----------
    model : GPModel
        Gaussian process model.
    xi : ndarray, shape (n, d)
        Locations of the observed data points.
    zi : ndarray, shape (n,)
        Observed values at the data points.
    xt : ndarray, shape (n, d)
        Locations of points to be predicted.
    R : list of intervals
        List of relaxation intervals, each specified as [l_k, u_k].
    covparam0 : ndarray, optional
        Initial guess for the covariance parameters.
    info : bool, optional
        Whether to return additional information.
    verbosity : int, optional
        Verbosity level.

    Returns
    -------
    zi_relaxed : ndarray, shape (n,)
        Relaxed output data.
    (zpm, zpv) : tuple of ndarrays
        Relaxed posterior mean and variance.
    model : GPmp model
        Updated GPmp Gaussian process model.
    info_ret : dict, optional
        Additional information (if info=True).
    """
    if info is True:
        model, zi_relaxed, ind_relaxed, info_ret = remodel(
            model, xi, zi, R,
            covparam0=None,
            info=info,
            verbosity=verbosity
        )
    else:
        model, zi_relaxed, ind_relaxed = remodel(
            model, xi, zi, R,
            covparam0=None,
            info=info,
            verbosity=verbosity
        )
        info_ret = None

    zpm, zpv = model.predict(xi, zi_relaxed, xt)

    return zi_relaxed, (zpm, zpv), model, info_ret


def select_optimal_threshold_above_t0(model, xi, zi, t0, G=20):
    """
    Choose threshold for reGP with relaxation above t0

    This function selects an optimal threshold for a reGP above t0 by
    minimizing the truncated continuous ranked probability score
    (tCRPS) over a range of possible thresholds.

    Parameters
    ----------
    model : GPModel
        Gaussian process model.
    xi : ndarray, shape (n, d)
        Locations of the observed data points.
    zi : ndarray, shape (n,)
        Observed values at the data points.
    t0 : float
        Lower limit of the threshold range.
    G : int, optional, default: 20
        Number of candidate thresholds to evaluate.

    Returns
    -------
    Rgopt : ndarray, shape (1, 2)
        Optimal relaxation interval, specified as [threshold, inf].
    """
    t = gnp.logspace(gnp.log10(t0 - zi.min()), gnp.log10(gnp.max(zi) - zi.min()), G + 1) + zi.min()
    t = t[:-1]

    J = gnp.numpy.zeros(G)
    for g in range(G):
        Rg = gnp.numpy.array([[t[g], gnp.numpy.inf]])
        model, zi_relaxed, _ = remodel(model, xi, zi, Rg)
        zloom, zloov, _ = model.loo(xi, zi_relaxed)
        tCRPS = gp.misc.scoringrules.tcrps_gaussian(zloom, gnp.sqrt(zloov), zi_relaxed, a=-gnp.inf, b=t0)
        J[g] = gnp.sum(tCRPS)

    gopt = gnp.argmin(gnp.asarray(J))
    Rgopt = gnp.numpy.array([[t[gopt], gnp.numpy.inf]])

    return Rgopt


# ---------------------------------------
