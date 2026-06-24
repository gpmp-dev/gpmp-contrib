Diagnostics and inspection
==========================

Diagnostics answer two questions:

1. Did parameter selection end in a usable state?
2. Are the LOO and test-set scores consistent with the expected predictive
   accuracy?

Run diagnostics after ``select_params``:

.. code-block:: python

   model.select_params(xi, zi)
   model.run_diagnosis(xi, zi)
   model.run_perf(xi, zi, xtzt=(xt, zt), zpmzpv=(zpm, zpv))

Model diagnosis
---------------

``run_diagnosis(xi, zi)`` prints one block per output.  For the seeded Hartmann4
example in :doc:`../getting_started`, the output is:

.. literalinclude:: ../_static/example_results/getting_started_hartmann4_diagnosis.txt
   :language: text

The displayed block is produced by ``model.run_diagnosis(xi, zi)`` after
``model.select_params(xi, zi)``.

The first part reports the optimizer status:

``cvg_reached``
    SciPy optimizer success flag.

``optimal_val``
    Whether the final criterion value is finite and usable.

``n_evals``
    Number of criterion evaluations.

``initial_val`` and ``final_val``
    Criterion value before and after optimization.  Compare them to detect
    optimizer-start or bounds problems.

The parameter table prints normalized coordinates, bounds, raw values, and
denormalized values.  For Matérn lengthscales, the raw covariance vector stores
``-log(rho)`` while the denormalized display reports ``rho``.

Prediction performance
----------------------

``run_perf`` reports leave-one-out summaries and, when test values are supplied,
test-set summaries:

.. literalinclude:: ../_static/example_results/getting_started_hartmann4_perf.txt
   :language: text

The displayed block is produced by
``model.run_perf(xi, zi, xtzt=(xt, zt), zpmzpv=(zpm, zpv))``.

For each output, the displayed quantities are defined as follows.  For
reference values :math:`z_i`,

.. math::

   \mathrm{TSS} = \sum_i (z_i - \bar z)^2.

For leave-one-out prediction, let
:math:`e_i^{\mathrm{LOO}} = z_i - \widehat z_{-i}(x_i)`, where
:math:`\widehat z_{-i}` is computed without observation :math:`i`.  Then

.. math::

   \mathrm{PRESS} = \sum_i \left(e_i^{\mathrm{LOO}}\right)^2,
   \qquad
   Q^2 = 1 - \frac{\mathrm{PRESS}}{\mathrm{TSS}}.

For a test set, let :math:`e_i^{\mathrm{test}} = z_i - \widehat z(x_i)`.  Then

.. math::

   \mathrm{RSS} = \sum_i \left(e_i^{\mathrm{test}}\right)^2,
   \qquad
   R^2 = 1 - \frac{\mathrm{RSS}}{\mathrm{TSS}}.

In both blocks,
:math:`\mathrm{RMSE} = \sqrt{\mathrm{SSE}/n}` with
:math:`\mathrm{SSE}=\mathrm{PRESS}` for LOO and
:math:`\mathrm{SSE}=\mathrm{RSS}` for the test set.  ``std(z)`` is the
empirical standard deviation of the reference values in the block.
``rmse/std(z)`` is a scale-free error.  ``press/tss`` and ``rss/tss`` are
error-to-variance ratios, so smaller values are better.

A low ``Q2`` or ``R2`` does not identify the cause by itself.  Inspect the
observation design, covariance class, bounds, optimizer start, and prior anchors.

Model state checks
------------------

Use per-output entries to inspect the selected raw vectors, readable ``Param``
object, optimizer report, and prior object when one exists:

.. code-block:: python

   covparam = model[0]["model"].covparam
   param = model[0].get_param()
   info = model[0]["info"]
   prior = model[0].get("prior", None)

Use :doc:`model_state` for the storage convention and :doc:`priors` for prior
objects.

Selection-criterion inspection
------------------------------

The optimizer report stores ``selection_criterion_nograd``.  Use this callable
for diagnosis, plotting, and posterior parameter sampling when gradients are not
needed:

.. code-block:: python

   criterion = model[0]["info"]["selection_criterion_nograd"]
   value = criterion(model[0]["model"].covparam)

Cross sections of the selection criterion help check whether the optimizer
result is isolated or lies on a flat region.  Cross sections are provided by
``gpmp.modeldiagnosis``.

Posterior parameter sampling
----------------------------

``sample_parameters`` forwards to posterior-sampling functions in
``gpmp.mcmc.param_posterior``.  The common methods are ``"mh"``, ``"nuts"``,
and ``"smc"``:

.. code-block:: python

   res = model.sample_parameters(method="mh", n_steps_total=2000)

The sampler uses the stored ``info`` object, so ``select_params`` must be called
first.  The return dictionary contains method-specific objects and, for methods
that record them, criterion or log-target traces.
