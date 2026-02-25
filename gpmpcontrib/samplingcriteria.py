# --------------------------------------------------------------
# Author: Emmanuel Vazquez <emmanuel.vazquez@centralesupelec.fr>
# Copyright (c) 2022-2026, CentraleSupelec
# License: GPLv3 (see LICENSE)
# --------------------------------------------------------------
import gpmp.num as gnp


def isinbox(box, x):
    b = gnp.logical_and(gnp.all(x >= box[0], axis=1), gnp.all(x <= box[1], axis=1))
    return b


def excursion_probability(t, zpm, zpv):
    """Computes the probabilities of exceeding the threshold t for
    Gaussian predictive distributions with means zpm and variances
    zpv. The input argument must have the following sizes:

        * zpm        M x 1,
        * zpv        M x 1,

     where M is the number of points where the EI must be
     computed. The output has size M x 1.
    """
    p = gnp.empty(zpm.shape)
    delta = zpm - t
    sigma = gnp.sqrt(zpv)
    b = sigma > 0.0

    # Where sigma > 0
    u = gnp.where(b, delta / sigma, 0.0)  # Avoid division by zero
    p = gnp.where(b, gnp.normal.cdf(u), 0.0)  # Compute p where sigma > 0

    # Condition where sigma == 0
    p = gnp.where(b, p, delta > 0.0)

    # # Compute p where sigma > 0
    # u = delta[b] / sigma[b]
    # p[b] = gnp.normal.cdf(u)

    # # Compute p where sigma == 0
    # b = gnp.logical_not(b)
    # p[b] = gnp.asdouble(delta[b] > 0)

    return p


def excursion_logprobability(t, zpm, zpv):
    """Computes the log probabilities of exceeding the threshold t for
    Gaussian predictive distributions with means zpm and variances zpv.

    Parameters:
      t (float): Threshold value.
      zpm (gnp.array): Predictive means, shape (M, 1).
      zpv (gnp.array): Predictive variances, shape (M, 1).

    Returns:
      gnp.array: Log probabilities of exceeding the threshold, shape (M, 1).
    """
    delta = zpm - t
    sigma = gnp.sqrt(zpv)
    b = sigma > 0.0

    u = gnp.where(b, delta / sigma, 0.0)  # Avoid division by zero
    log_p = gnp.where(b, gnp.normal.logcdf(u), gnp.where(delta > 0, 0.0, -gnp.inf))

    return log_p


def excursion_misclassification_probability(t, zpm, zpv):
    """
    Computes the probability of misclassification for the excursion set.

    The misclassification probability is defined as:
        tau_n(x) = min(P_n(ξ(x) > t), 1 - P_n(ξ(x) > t))

    This measures the uncertainty in classifying whether a point belongs
    to the excursion set or not assuming a hard classifier eta_n(x) = 1_{p_n(x) > 1/2}

    See Sequential design of computer experiments for the estimation of a probability of failure.
    J Bect, D Ginsbourger, L Li, V Picheny, E Vazquez. Statistics and Computing 22, 773-793
    https://arxiv.org/pdf/1009.5177

    Parameters
    ----------
    t : float
        Threshold value defining the excursion set.
    zpm : gnp.array, shape (M, 1)
        Predictive mean values at M points.
    zpv : gnp.array, shape (M, 1)
        Predictive variance values at M points.

    Returns
    -------
    gnp.array, shape (M, 1)
        Misclassification probabilities for each point.
    """
    g = excursion_probability(t, zpm, zpv)
    return gnp.minimum(g, 1 - g)

def excursion_wMSE(t, zpm, zpv, alpha=1.0, beta=0.5):
    """
    Computes the weighted mean squared error (wMSE) for excursion set estimation.

    The wMSE is defined as:
        wMSE(x) = tau(x) * zpv

    where:
      - tau(x) is the misclassification probability,
      - zpv is the predictive variance.

    Parameters
    ----------
    t : float
        Threshold value defining the excursion set.
    zpm : gnp.array, shape (M, 1)
        Predictive mean values at M points.
    zpv : gnp.array, shape (M, 1)
        Predictive variance values at M points.

    Returns
    -------
    gnp.array, shape (M, 1)
        Weighted mean squared error at each point.

    Notes
    -----
    - Weighting behavior:
      * α controls misclassification sensitivity
      * Higher α emphasizes uncertain classifications (boundary regions)
      * Higher β emphasizes high-variance regions
      * β=1.0: Strong focus on high-variance regions (σ² term
        dominates). For global exploration, β=1.0 may be preferred.
      * β=0.5: Balanced exploration (σ term, recommended default). For
        excursion set estimation, β=0.5 typically provides better
        boundary detection.
    """
    tau = excursion_misclassification_probability(t, zpm, zpv)
    return tau**alpha * zpv**beta


def box_probability(box, zpm, zpv):
    """Compute probability that predictions fall within given bounds
    for each dimension.

    For Gaussian predictions N(zpm, zpv), calculates P(lower <= value
    <= upper) per dimension.

    Parameters
    ----------
    box : (2, dim_output) array-like
        [lower, upper] bounds for each output dimension
    zpm : (n, dim_output) gnp.array
        Predictive means
    zpv : (n, dim_output) gnp.array
        Predictive variances (non-negative)

    Returns
    -------
    Returns
    -------
    tuple of (gnp.array, gnp.array)
        - product_probs: (n, 1) array of product probabilities across dimensions
        - probs: (n, dim_output) array of probabilities per dimension

    Notes
    -----
    - Returns 1 when zpv=0 and zpm is within bounds, 0 otherwise
    - Handles infinite bounds

    """
    lower, upper = gnp.asarray(box[0]), gnp.asarray(box[1])
    delta_min, delta_max = lower - zpm, upper - zpm
    sigma = gnp.sqrt(zpv)
    b = sigma > 0

    # Compute probabilities
    u_min = gnp.where(b, delta_min / sigma, 0.0)
    u_max = gnp.where(b, delta_max / sigma, 0.0)

    probs = gnp.where(
        b, 
        gnp.normal.cdf(u_max) - gnp.normal.cdf(u_min), 
        gnp.asdouble((delta_max > 0) & (delta_min < 0))
    )

    # Compute product across dimensions (axis=1)
    probs_product = gnp.prod(probs, axis=1, keepdims=True)

    return probs_product, probs


def box_logprobability(box, zpm, zpv):
    """Compute log probabilities of predictions being within given bounds.

    Returns
    -------
    tuple of (gnp.array, gnp.array)
        - sum_log_probs: (n, 1) array of summed log probabilities across dimensions
        - log_probs: (n, dim_output) array of log probabilities per dimension
    """
    lower, upper = gnp.asarray(box[0]), gnp.asarray(box[1])
    delta_min, delta_max = lower - zpm, upper - zpm
    sigma = gnp.sqrt(zpv)
    b = sigma > 0

    # Compute log probabilities
    u_min = gnp.where(b, delta_min / sigma, 0)
    u_max = gnp.where(b, delta_max / sigma, 0)

    log_phi_max = gnp.normal.logcdf(u_max)
    log_phi_min = gnp.normal.logcdf(u_min)

    mask = log_phi_max > log_phi_min
    log_probs = gnp.where(
        b & mask,
        log_phi_max + gnp.log1p(-gnp.exp(log_phi_min - log_phi_max)),
        gnp.where((delta_max > 0) & (delta_min < 0), 0.0, -gnp.inf),
    )

    # Compute sum across dimensions (axis=1)
    sum_logprobs = gnp.sum(log_probs, axis=1, keepdims=True)

    return sum_logprobs, log_probs


def box_misclassification_probability(box, zpm, zpv):
    """
    Computes the probability of misclassification for a box.

    Returns
    -------
    gnp.array, shape (M, 1)
        Misclassification probabilities for each point.
    """
    g_prod, g = box_probability(box, zpm, zpv)
    return gnp.minimum(g_prod, 1 - g_prod), gnp.minimum(g, 1 - g)


def box_misclassification_logprobability(box, zpm, zpv, clip_logthreshold=1e-6):
    """Computes log misclassification probabilities for box estimation.
    
    Numerically stable version using log probabilities to avoid underflow.
    
    Returns
    -------
    tuple of (gnp.array, gnp.array)
        - sum_log_tau: (n, 1) array of summed log misclassification probs
        - log_tau: (n, dim_output) array of log misclassification probs
    """
    # Get log probabilities (more stable than log(probability))
    sum_log_probs, log_probs = box_logprobability(box, zpm, zpv)
    
    # Compute log(1 - exp(log_probs)) using log1mexp for stability
    probs = gnp.exp(log_probs)
    probs = gnp.clip(probs, clip_logthreshold, 1 - clip_logthreshold)
    log_1_minus_p = gnp.log1p(-probs)  # More accurate than log(1-p)
    
    # log_tau = log(min(p, 1-p)) = min(log_probs, log_1_minus_p)
    log_tau = gnp.minimum(log_probs, log_1_minus_p)
    
    # Sum across dimensions
    sum_log_tau = gnp.sum(log_tau, axis=1, keepdims=True)
    
    return sum_log_tau, log_tau

def box_wMSE(box, zpm, zpv, normalization=1.0, alpha=1.0, beta=0.5):
    """Computes the weighted mean squared error (wMSE) for box/bounds estimation.

    The wMSE combines misclassification uncertainty and predictive variance:
        wMSE = τ^α * (σ² * normalization)^β
    where:
        τ = misclassification probability (min(p, 1-p))
        σ² = predictive variance
        α, β = weighting exponents

    Parameters
    ----------
    box : (2, dim_output) array-like
        [lower, upper] bounds for each output dimension
    zpm : (n, dim_output) gnp.array
        Predictive means
    zpv : (n, dim_output) gnp.array
        Predictive variances (must be non-negative)
    normalization : float or (dim_output,) array-like, optional
        Scaling factor for variance (default: 1.0). Can be either:
        - Single float (applied to all dimensions)
        - Per-dimension scaling factors
    alpha : float, optional
        Exponent for misclassification term (default: 1.0)
    beta : float, optional
        Exponent for variance term (default: 1.0)

    Returns
    -------
    tuple of (gnp.array, gnp.array)
        - wmse_sum: (n, 1) array of summed wMSE across dimensions
        - wmse: (n, dim_output) array of wMSE per dimension

    Notes
    -----
    - The normalization parameter allows per-dimension scaling
    - Weighting behavior:
      * α controls misclassification sensitivity
      * Higher α emphasizes uncertain classifications (boundary regions)
      * Higher β emphasizes high-variance regions
      * β=1.0: Strong focus on high-variance regions (σ² term
        dominates). For global exploration, β=1.0 may be preferred.
      * β=0.5: Balanced exploration (σ term, recommended default). For
        excursion set estimation, β=0.5 typically provides better
        boundary detection.
    """
    # Get log misclassification probabilities
    sum_log_tau, log_tau = box_misclassification_logprobability(box, zpm, zpv)
    # Convert back from log space with proper scaling
    tau_alpha = gnp.exp(alpha * log_tau)
    # Compute variance term
    var_term = (zpv * normalization)**beta
    # Compute per-dimension wMSE
    wmse = tau_alpha * var_term
    # Sum across dimensions (in original space)
    wmse_sum = gnp.sum(wmse, axis=1, keepdims=True)
    
    return wmse_sum, wmse


def expected_improvement(t, zpm, zpv):
    """Computes the Expected Improvement (EI) criterion for a
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

    """
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
