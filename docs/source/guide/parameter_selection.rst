Parameter selection
===================

``gpmp-contrib`` delegates numerical parameter selection to ``gpmp.kernel``.
The container class chooses the criterion, builds an optimizer start, calls
SciPy, and stores the result for each output.

Parameter selection chooses covariance parameters inside the selected GP family.
For the provided ML class, it also chooses the constant mean parameter.
Background on covariance modeling and kriging is given by
:cite:t:`stein1999kriging` and
:cite:t:`chiles1999geostatistics`.  Empirical comparisons of GP parameter
selection are discussed by :cite:t:`petit2023parameter`.

Covariance parameter vector
---------------------------

For the Matérn classes, the covariance parameter vector has length
``1 + input_dim``:

.. code-block:: text

   covparam = [log(sigma2), -log(rho_0), ..., -log(rho_{d-1})]

The first entry is the log process variance.  The remaining entries use the
inverse log-lengthscale convention.  A positive value means a shorter
lengthscale than ``1``.  A negative value means a longer lengthscale than ``1``.

The physical parameters are recovered by

.. math::

   \sigma^2 = \exp(\texttt{covparam[0]}),
   \qquad
   \rho_j = \exp(-\texttt{covparam[j + 1]}).

Optimizer starts, bounds, and reported raw values are expressed in the
``covparam`` coordinates.

Which criterion is used
-----------------------

``Model_ConstantMean_Maternp_ML``
    Uses maximum likelihood.  The constant mean parameter and covariance
    parameters are optimized together.

``Model_ConstantMean_Maternp_REML``
    Uses restricted likelihood.  The mean basis is constant or linear.  Mean
    coefficients are profiled out by the restricted likelihood, and the
    criterion is optimized over covariance parameters.

``Model_ConstantMean_Maternp_REMAP``
    Uses restricted likelihood plus prior terms.  The default REMAP class uses a
    log-variance prior and a log-lengthscale prior.  See :doc:`priors`.

Maximum likelihood
------------------

In negative-log form, the ML criterion is

.. math::

   J_\mathrm{ML}(\theta, \beta)
   = \frac{1}{2}\log\det K_\theta
     + \frac{1}{2}(z - H\beta)^\top K_\theta^{-1}(z - H\beta)
     + \text{constant}.

Here ``theta`` denotes covariance parameters, ``beta`` mean parameters,
``K_theta`` the covariance matrix, and ``H`` the mean design matrix.  For
``Model_ConstantMean_Maternp_ML``, ``H`` is a column of ones.

Restricted likelihood
---------------------

The REML criterion adds the standard correction for the mean space:

.. math::

   J_\mathrm{REML}(\theta)
   = \frac{1}{2}\log\det K_\theta
     + \frac{1}{2}\log\det(H^\top K_\theta^{-1}H)
     + \frac{1}{2} z^\top P_\theta z
     + \text{constant},

where

.. math::

   P_\theta
   = K_\theta^{-1}
     - K_\theta^{-1}H
       (H^\top K_\theta^{-1}H)^{-1}
       H^\top K_\theta^{-1}.

The matrix ``H`` is built from the mean basis specified by
``mean_specification``.

REMAP
-----

REMAP adds a log-prior term to REML:

.. math::

   J_\mathrm{REMAP}(\theta)
   = J_\mathrm{REML}(\theta) - \log \pi(\theta).

In ``gpmp-contrib``, REMAP prior anchors are separate from optimizer starts.
Changing the optimizer start does not by itself change the prior anchor.  Use
``set_prior`` to change prior anchors or hyperparameters.

What select_params does
-----------------------

Example:

.. code-block:: python

   model.select_params(
       xi,
       zi,
       method="SLSQP",
       method_options={"maxiter": 100},
   )

For each output, ``select_params``:

1. checks the observation and response shapes.
2. builds or rebuilds the selection criterion when needed.
3. chooses an optimizer start.
4. chooses bounds.
5. calls ``gpmp.kernel.autoselect_parameters``.
6. stores the selected raw vectors, readable ``Param`` object, and optimizer
   report in ``model[k]``.

Optimizer starts
----------------

``param0`` sets the optimizer start.  For one output, it can be a raw vector or
a ``Param`` object.  For several outputs, pass a list with one entry per output.

.. code-block:: python

   param0 = model[0].get_param()
   model.select_params(xi, zi, param0=param0)

When ``param0`` is not provided, the start is selected as follows:

1. if ``force_param_initial_guess=True``, use the configured initial-guess
   procedure.
2. otherwise, if the underlying ``gpmp.core.Model`` already has parameters, use
   the current model parameters.
3. otherwise, use the configured initial-guess procedure.

When ``param0`` is provided, it takes precedence and
``force_param_initial_guess`` has no effect.

Bounds and SciPy options
------------------------

Bounds are expressed in the normalized coordinates optimized by SciPy.  They are
selected in this order:

1. explicit ``bounds``.
2. ``bounds_factory(model_dict, xi, zi_vector, param0_init)``.
3. bounds stored in the current ``Param`` object when
   ``use_bounds_from_param_obj=True``.
4. no bounds.

``bounds_delta`` intersects the selected bounds with a local interval around the
optimizer start.  ``method`` and ``method_options`` are forwarded to SciPy
through ``gpmp.kernel.autoselect_parameters``.

Example:

.. code-block:: python

   model.select_params(
       xi,
       zi,
       bounds_delta=5.0,
       method="SLSQP",
       method_options={"maxiter": 80, "ftol": 1e-6},
   )

Stored results
--------------

After ``select_params(xi, zi)``, each output entry ``model[k]`` stores:

- ``model[k]["model"].covparam``: selected covariance parameter vector.
- ``model[k]["model"].meanparam``: selected mean parameter vector, if any.
- ``model[k]["param"]`` or ``model[k].get_param()``: readable ``Param`` object.
- ``model[k]["info"]``: optimizer report and criterion wrappers.
- ``model[k]["info"]["param0"]``: optimizer start used for that output.
- ``model[k]["info"]["selection_criterion_nograd"]``: criterion callable for
  diagnosis, plotting, and posterior sampling without gradient tracking.

For REMAP models with priors, the resolved prior object is stored in
``model.get_prior(k)``.

Failure diagnosis
-----------------

If selection fails or returns low prediction scores, inspect these objects in
order:

1. ``model[k]["info"]`` for optimizer success, initial value, final value, and
   number of evaluations.
2. ``model[k].get_param()`` for selected values and bounds.
3. ``model.get_prior(k)`` for REMAP prior anchors, when applicable.
4. the design ``xi`` and response ``zi`` for poor coverage, duplicates, or
   scale problems.
5. the optimizer start ``model[k]["info"]["param0"]`` when the criterion is
   multimodal.
