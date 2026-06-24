Model containers
================

``ModelContainer`` manages one or more independent ``gpmp.core.Model`` objects,
one per output.  It stores the mean and covariance callables, initial-guess
procedure, selection criterion, selected parameter object, optimizer report, and
underlying ``gpmp`` model for each output.

Per-output access
-----------------

Use ``model[k]`` to inspect output ``k``.  The entry is an attribute-access
dictionary.  Main keys are:

- ``model[k]["model"]``: underlying ``gpmp.core.Model``.
- ``model[k]["param"]`` or ``model[k].get_param()``: readable ``Param`` object.
- ``model[k]["selection_criterion"]``: selection criterion used by optimization.
- ``model[k]["info"]``: optimizer and criterion information after
  ``select_params``.
- ``model[k]["prior"]``: resolved prior object for classes using REMAP
  selection with priors.

Core object
-----------

.. py:class:: ModelContainer(name, output_dim, parameterized_mean, mean_specification, covariance_specification, initial_guess_procedures=None, selection_criteria=None)
   :no-index:

   Container for one or more ``gpmp.core.Model`` objects.

   Direct construction requires mean specifications, covariance specifications,
   and usually subclass methods for building initial guesses, selection
   criteria, and parameter objects.  Applications usually instantiate one of the
   provided Matérn container classes, such as
   ``Model_ConstantMean_Maternp_REML`` or ``Model_ConstantMean_Maternp_REMAP``.

   .. py:method:: select_params(xi, zi, *, force_param_initial_guess=True, param0=None, use_bounds_from_param_obj=True, bounds=None, bounds_factory=None, bounds_delta=None, method="SLSQP", method_options=None, rebuild_selection_criterion=None)
      :no-index:

      Select parameters independently for each output.

      ``xi`` has shape ``(n, d)``.  ``zi`` has shape ``(n,)`` for one output or
      ``(n, output_dim)`` for several outputs.  If ``param0`` is provided, it is
      used as the optimizer start and ``force_param_initial_guess`` is ignored.
      If ``param0`` is not provided, the method uses the configured initial
      guess procedure when ``force_param_initial_guess`` is true or when current
      parameters are missing.

      Bounds are selected in this order: explicit ``bounds``, then
      ``bounds_factory``, then bounds from the parameter object when
      ``use_bounds_from_param_obj`` is true.  ``bounds_delta`` intersects the
      chosen bounds with a local box around the optimizer start.

      Stores, for each output ``k``:

      - ``model[k]["model"].covparam``.
      - ``model[k]["model"].meanparam`` when the mean is parameterized.
      - ``model[k]["param"]``, a readable ``gpmp.parameter.Param`` object.
      - ``model[k]["info"]``, the optimizer report and criterion callables.
      - ``model[k]["prior"]`` for classes using REMAP selection with priors.

      Raises ``ValueError`` for incompatible shapes, invalid bounds, missing
      observation data when a REMAP criterion needs observations, or missing
      per-output initialization values.

   .. py:method:: predict(xi, zi, xt, convert_in=True, convert_out=True)
      :no-index:

      Compute posterior means and variances at ``xt`` for all outputs.
      ``xi`` has shape ``(n, d)``, ``zi`` has shape ``(n,)`` or
      ``(n, output_dim)``, and ``xt`` has shape ``(m, d)``.  Returns
      ``(zpm, zpv)`` with shape ``(m, output_dim)``.  With
      ``convert_out=True``, returned arrays are NumPy arrays.  With
      ``convert_out=False``, they are backend objects.

   .. py:method:: loo(xi, zi, convert_in=True, convert_out=True)
      :no-index:

      Compute leave-one-out predictions for all outputs.  Returns
      ``(zloo, sigma2loo, eloo)``, each with shape ``(n, output_dim)``.

   .. py:method:: run_diagnosis(xi, zi)
      :no-index:

      Print per-output parameter-selection status, parameter values, and data
      summaries through ``gpmp.modeldiagnosis``.  Raises ``ValueError`` if
      ``select_params`` has not populated ``model[k]["info"]``.

   .. py:method:: run_perf(xi, zi, *, output_ind=None, loo=True, loo_res=None, xtzt=None, zpmzpv=None, convert_in=True)
      :no-index:

      Print prediction-performance summaries through ``gpmp.modeldiagnosis``.
      If ``output_ind`` is ``None``, reports all outputs.  Otherwise, reports
      only the selected output.

      ``loo=True`` enables leave-one-out metrics.  ``loo_res`` may provide
      precomputed ``(zloom, zloov, eloo)`` arrays.  ``xtzt`` provides test
      points and reference values as ``(xt, zt)``.  ``zpmzpv`` may provide
      precomputed predictions as ``(zpm, zpv)``.  For multi-output arrays,
      ``run_perf`` slices the second dimension for each output.

      The printed metrics are ``tss``, ``press``, ``rss``, ``rmse``,
      ``rmse/std(z)``, ``Q2``, and ``R2``.  Their definitions are given in
      :doc:`../guide/diagnostics`.

   .. py:method:: compute_conditional_simulations(xi, zi, xt, n_samplepaths=1)
      :no-index:

      Draw conditional sample paths at ``xt``.  Returns shape
      ``(nt, n_samplepaths)`` for one output and
      ``(nt, n_samplepaths, output_dim)`` for several outputs.

   .. py:method:: sample_parameters(method="nuts", model_indices=None, init_box=None, sampling_box=None, **kwargs)
      :no-index:

      Sample covariance parameters from the stored selection criterion.
      Implemented public methods are ``"mh"``, ``"hmc"``, ``"nuts"``, and
      ``"smc"``.  ``init_box`` initializes random starts or particles.
      ``sampling_box`` truncates the sampling domain.  Points outside the box
      get log-probability ``-inf``.

      Requires ``select_params`` first, because the sampler uses
      ``model[k]["info"]``.  ``init_box`` is mandatory for ``method="smc"``.
      Returns a dictionary keyed by output index:

      - ``"mh"``: ``{"samples", "mh", "criterion_values"}``.
      - ``"hmc"`` and ``"nuts"``: ``{"samples", "hmc"}`` or
        ``{"samples", "nuts"}``.
      - ``"smc"``: ``{"particles", "smc"}``.

Backend conventions
-------------------

``ModelContainer`` converts inputs with ``gpmp.num.asarray`` when
``convert_in=True``.  Prediction, LOO, and conditional-simulation methods return
NumPy arrays when ``convert_out=True`` and backend objects when
``convert_out=False``.  Optimizer reports and stored model parameters use
backend objects.

Supporting objects
------------------

.. automodule:: gpmpcontrib.modelcontainer
   :members: AttrDict, SelectionCriterionBuildContext, mean_parameterized_constant, mean_linpred_constant, mean_linpred_linear
   :show-inheritance:
