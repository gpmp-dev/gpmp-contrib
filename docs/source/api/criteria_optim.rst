Sampling criteria and optimization procedures
=============================================

Sampling criteria operate on posterior means and variances.  Strategy classes
combine those criteria with observation updates and candidate-set management.

Array conventions
-----------------

The criterion functions accept backend objects compatible with ``gpmp.num``.
For scalar-output criteria, ``zpm`` and ``zpv`` have shape ``(m, 1)`` or
``(m,)``.  For multi-output box criteria, ``zpm`` and ``zpv`` have shape
``(m, q)`` and ``box`` has shape ``(2, q)``.  Variances must be non-negative.

Sampling criteria
-----------------

.. automodule:: gpmpcontrib.samplingcriteria
   :members: isinbox, excursion_probability, excursion_logprobability, excursion_misclassification_probability, box_probability, box_logprobability, box_misclassification_probability, box_misclassification_logprobability, expected_improvement
   :show-inheritance:

.. py:function:: excursion_wMSE(t, zpm, zpv, alpha=1.0, beta=0.5)
   :no-index:

   Criterion used for excursion-set learning at threshold ``t``.

   ``zpm`` and ``zpv`` are posterior means and variances.  The criterion combines
   posterior classification uncertainty and posterior variance.  It is evaluated
   pointwise on a candidate set or particle set.  Returns a backend object with
   shape matching ``zpm``.

.. py:function:: box_wMSE(box, zpm, zpv, normalization=1.0, alpha=1.0, beta=0.5)
   :no-index:

   Multi-output criterion for inverse-image estimation with an output-space box.

   ``box`` contains lower and upper target bounds for each output.  The function
   returns a scalar criterion per candidate point and auxiliary per-output
   quantities: ``(wmse_sum, wmse)`` with shapes ``(m, 1)`` and ``(m, q)``.

.. py:function:: expected_improvement(t, zpm, zpv)
   :no-index:

   Expected improvement for a maximization convention.  ``t`` is the current
   threshold, and ``zpm`` and ``zpv`` are Gaussian posterior means and
   variances.  For minimization examples in this package, callers pass
   ``-z_best``, ``-zpm``, and ``zpv``.

Expected improvement
--------------------

.. automodule:: gpmpcontrib.optim.expectedimprovement
   :members:
   :undoc-members:
   :show-inheritance:

.. py:class:: ExpectedImprovementGridSearch(problem, model, xt, options=None)
   :no-index:

   Sequential minimization strategy on a fixed candidate set.  ``xt`` has shape
   ``(m, d)``.  The current estimate is ``min(zi)``.  The sampling criterion is
   ``expected_improvement(-current_estimate, -zpm, zpv)``.

.. py:class:: ExpectedImprovementSMC(problem, model, options=None)
   :no-index:

   Sequential minimization strategy with an SMC candidate set.  The SMC target
   log-density is the posterior log-probability of improving on the current
   observed minimum, computed inside the input box and set to ``-inf`` outside
   it.

Excursion sets
--------------

.. automodule:: gpmpcontrib.optim.excursionset
   :members: ExcursionSetGridSearch
   :show-inheritance:

.. py:class:: ExcursionSetBSS(problem, model, u_init, u_target, options=None)
   :no-index:

   BSS-style strategy for excursion-set estimation.

   The target set is ``{x : xi(x) > u_target}``.  The strategy introduces
   ``mu in [0, 1]`` and the intermediate threshold
   ``u(mu) = (1 - mu) * u_init + mu * u_target``.  At a fixed ``mu``, SMC
   particles target the posterior log-probability of excursion above ``u(mu)``
   inside the input box and ``-inf`` outside the box.

Set inversion
-------------

.. automodule:: gpmpcontrib.optim.setinversion
   :members: SetInversionGridSearch
   :show-inheritance:

.. py:class:: SetInversionBSS(problem, model, box_init, box_target, options=None)
   :no-index:

   BSS-style strategy for inverse-image estimation.

   The target set is ``{x : xi(x) in box_target}``.  The strategy interpolates
   between ``box_init`` and ``box_target`` with ``mu in [0, 1]``.  At each
   intermediate box, SMC particles target posterior box-membership probability
   inside the input domain.

Pareto utilities
----------------

Pareto utilities use NumPy arrays.  ``z`` has shape ``(n, q)`` and stores costs
to minimize.

``pareto_points(z)``
    Return a boolean mask selecting non-dominated rows.

``pareto_filter(zopt, z)``
    Return a boolean mask indicating rows of ``z`` dominated by at least one row
    of ``zopt``.

``dominated_area_2d(z_ref, z_opt)``
    Return the two-dimensional dominated area relative to ``z_ref``.

``hausdorff_distance(z1, z2)``
    Return the symmetric Hausdorff distance between two point clouds.

.. automodule:: gpmpcontrib.optim.pareto
   :members:
   :undoc-members:
   :show-inheritance:
