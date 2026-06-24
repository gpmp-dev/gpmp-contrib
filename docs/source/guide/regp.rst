Relaxed Gaussian processes
==========================

The ``gpmpcontrib.regp`` module implements relaxed Gaussian-process modeling.
The procedure starts from a GP model and observations ``(xi, zi)``.  Values whose
outputs fall in a relaxation interval are allowed to move inside that interval
during parameter selection.

The method is described by :cite:t:`petit2025regp`.  It is intended for
threshold-oriented prediction tasks where interpolation outside the region of
interest can degrade predictions inside the region of interest.

Problem setup
-------------

Let ``R`` be a list of output intervals.  For example, ``R = [[u, +inf]]`` means
that observations above ``u`` may be relaxed.  Observations outside ``R`` remain
fixed.  Observations inside ``R`` become optimization variables constrained to
stay inside their interval.

The optimized variables are:

- covariance parameters of the GP model.
- relaxed output values for observations assigned to a relaxation interval.

The input locations do not move.

Optimization problem
--------------------

Let :math:`I_0` be the indices of fixed observations and :math:`I_1` the
indices of observations assigned to a relaxation interval.  The reGP procedure
keeps

.. math::

   \widetilde z_i = z_i,\qquad i\in I_0,

and optimizes relaxed values :math:`\widetilde z_i` for :math:`i\in I_1` with
the interval constraints

.. math::

   \widetilde z_i \in R(i).

The optimized criterion is the restricted likelihood evaluated at the relaxed
observation vector:

.. math::

   \min_{\theta,\,\widetilde z_{I_1}}
   J_\mathrm{REML}\left(\theta, X, \widetilde z\right)
   \quad
   \text{subject to}\quad
   \widetilde z_i=z_i\ (i\in I_0),\quad
   \widetilde z_i\in R(i)\ (i\in I_1).

After optimization, prediction is the usual GP prediction conditioned on
``(xi, zi_relaxed)`` and on the selected covariance parameters.  The relaxed
values are not new observations.  They are optimization variables constrained by
the target interval.

Main functions
--------------

``get_membership_indices(zi, R)``
    Returns an integer index for each observation.  Index ``0`` means that the
    value is not relaxed.  Positive indices identify the interval in ``R``.

``split_data(xi, zi, ei, R)``
    Splits observations into fixed values and relaxed values according to
    membership indices.

``remodel(model, xi, zi, R, covparam0=None, info=False, verbosity=0)``
    Selects covariance parameters and relaxed output values.  Returns the
    updated model, relaxed observations, and indices of relaxed values.  When
    ``info=True``, it also returns the optimizer report.

``predict(model, xi, zi, xt, R, covparam0=None, info=False, verbosity=0)``
    Runs ``remodel`` and returns relaxed observations, posterior predictions at
    ``xt``, the updated model, and optional optimizer information.

``select_optimal_threshold_above_t0(model, xi, zi, t0, G=20)``
    Searches thresholds above ``t0`` and returns a relaxation interval selected
    by a tCRPS-based criterion.

Threshold selection
-------------------

For intervals of the form :math:`R_u=[u,+\infty)`, threshold selection evaluates
a finite set of candidate thresholds above :math:`t_0`.  For each candidate,
reGP remodeling gives relaxed values and leave-one-out Gaussian predictions
with means :math:`m_{-i}` and standard deviations :math:`s_{-i}`.  The selected
threshold minimizes the average threshold-weighted CRPS on the region
:math:`(-\infty,t_0]`:

.. math::

   \frac{1}{n}\sum_{i=1}^n
   \mathrm{tCRPS}\left(
      \mathcal{N}(m_{-i}, s_{-i}^2),
      \widetilde z_i,
      -\infty, t_0
   \right).

Shapes
------

- ``xi`` has shape ``(n, d)``.
- ``zi`` has shape ``(n,)`` or ``(n, 1)``.
- ``xt`` has shape ``(m, d)``.
- ``R`` is a list or array of intervals ``[[lower_0, upper_0], ...]``.

Example
-------

.. code-block:: python

   R = [[u, float("inf")]]
   zi_relaxed, (zpm, zpv), model, info = regp.predict(
       model,
       xi,
       zi,
       xt,
       R,
       info=True,
   )

Returned state
--------------

``zi_relaxed``
    Relaxed observation vector.  It has the same number of rows as ``zi``.
    Values outside all relaxation intervals are unchanged.

``(zpm, zpv)``
    Posterior means and variances computed with the relaxed observations.

``model``
    Updated GP model after reGP parameter selection.

``info``
    Optimizer report when ``info=True``.

Values inside a relaxation interval are optimizer variables constrained by the
corresponding interval bounds.  The executable script
:doc:`../examples/example20_regp` shows the full procedure and plots the fixed
observations, relaxed observations, posterior mean, and confidence band.

Failure conditions
------------------

The reGP functions raise when interval bounds are invalid, ``xi`` and ``zi``
have incompatible row counts, the starting covariance vector has incompatible
dimension, or the SciPy optimizer or reGP criterion raises an error.
