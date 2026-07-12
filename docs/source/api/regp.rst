Relaxed Gaussian processes
==========================

The ``gpmpcontrib.regp`` package exports ``predict`` and
``select_optimal_threshold_above_t0``.  Lower-level functions, including
``remodel``, are defined in ``gpmpcontrib.regp.regp``.

Function contracts
------------------

.. py:function:: get_membership_indices(zi, R)
   :no-index:

   Assign observations to relaxation intervals.  ``zi`` has shape ``(n,)`` or
   ``(n, 1)``.  ``R`` is a list of intervals ``[lower, upper]``.  Returns an
   integer NumPy array with shape ``(n,)``.  Value ``0`` means that the
   observation is outside all relaxation intervals.  Positive values identify
   the interval index starting at ``1``.

.. py:function:: split_data(xi, zi, ei, R)
   :no-index:

   Split observations according to membership indices.  Returns
   ``(x0, z0, ind0)`` for non-relaxed rows and ``(x1, z1, bounds, ind1)`` for
   relaxed rows.  ``bounds`` contains the interval bounds associated with each
   relaxed observation.

.. py:function:: make_regp_criterion_with_gradient(model, x0, z0, x1)
   :no-index:

   Build a differentiable reGP criterion for SciPy optimization.  Returns
   ``(crit_pre_grad, dcrit)``.  The first callable evaluates the objective and
   caches the gradient graph.  The second callable returns the gradient for the
   same parameter vector.

.. py:function:: gpmpcontrib.regp.regp.remodel(model, xi, zi, R, covparam0=None, info=False, verbosity=0, convert_in=True, convert_out=True)
   :no-index:

   Optimize covariance parameters and relaxed observation values.  ``xi`` has
   shape ``(n, d)`` and ``zi`` has shape ``(n,)`` or ``(n, 1)``.  If
   ``covparam0`` is ``None``, the function computes an anisotropic initial
   guess.  Returns ``(model, zi_relaxed, ind_relaxed)``.  With ``info=True``,
   returns ``(model, zi_relaxed, ind_relaxed, info_ret)``.

   The supplied ``model`` is modified in place through its ``covparam`` field.

.. py:function:: gpmpcontrib.regp.predict(model, xi, zi, xt, R, covparam0=None, info=False, verbosity=0)
   :no-index:

   Run ``remodel`` and predict at ``xt``.  Returns
   ``(zi_relaxed, (zpm, zpv), model, info_ret)``.  ``info_ret`` is ``None``
   when ``info=False``.

.. py:function:: gpmpcontrib.regp.select_optimal_threshold_above_t0(model, xi, zi, t0, G=20)
   :no-index:

   Evaluate ``G`` thresholds above ``t0`` and return the interval
   ``[[threshold, inf]]`` with the smallest tCRPS criterion.

The membership, split, criterion, and ``remodel`` functions are lower-level
helpers.  Import them from ``gpmpcontrib.regp.regp`` when direct access is
needed.
