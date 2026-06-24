Sequential procedures
=====================

Sequential procedures maintain the current data set, update the GP model, make
posterior predictions, and select new evaluation points.

``SequentialPrediction`` stores observations, model state, and predictions.  The
strategy classes add sampling-criterion based decisions.

``SequentialPrediction``
------------------------

.. py:class:: SequentialPrediction(model, force_param_initial_guess=True)
   :no-index:

   Store a ``ModelContainer`` and a growing observation set.

   .. py:method:: set_data(xi, zi)
      :no-index:

      Store observations.  ``xi`` has shape ``(n, d)`` and ``zi`` is reshaped to
      ``(n, output_dim)``.

   .. py:method:: set_data_with_model_selection(xi, zi)
      :no-index:

      Store observations, then call ``model.select_params``.

   .. py:method:: set_new_eval(xnew, znew)
      :no-index:

      Append one or more observations.  ``xnew`` can have shape ``(d,)`` or
      ``(m, d)``.  ``znew`` is reshaped to ``(m, output_dim)``.  Raises
      ``ValueError`` when dimensions do not match the stored data.

   .. py:method:: set_new_eval_with_model_selection(xnew, znew)
      :no-index:

      Append observations, then update model parameters.

   .. py:method:: predict(xt, convert_out=True, use_cache=False)
      :no-index:

      Predict at ``xt`` using stored data.  Returns ``(zpm, zpv)`` with shape
      ``(m, output_dim)``.  Raises ``ValueError`` if no data has been set.

   .. py:method:: compute_conditional_simulations(xt, n_samplepaths=1, type="intersection", method="svd", convert_in=True, convert_out=True)
      :no-index:

      Forward conditional simulation to the underlying ``ModelContainer``.

Sequential strategies
---------------------

.. py:class:: SequentialStrategy(problem, model, options=None)
   :no-index:

   Base class for criteria that select new evaluation points.  Subclasses must
   define ``set_initial_xt``, ``update_current_estimate``,
   ``update_search_space``, ``sampling_criterion``, and ``step``.

   Shared ``options`` keys are:

   - ``update_model_at_init``.
   - ``update_predictions_at_init``.
   - ``update_estimate_at_init``.
   - ``update_search_space_at_init``.
   - ``maximize_criterion``.

   ``set_initial_design(xi)`` evaluates the problem at ``xi``, stores the
   observations, and optionally updates model parameters, predictions, the
   estimate, and the search space according to these options.

.. py:class:: SequentialStrategyGridSearch(problem, model, xt, options=None)
   :no-index:

   Use a fixed candidate set ``xt`` with shape ``(m, d)``.  ``step`` evaluates
   the sampling criterion on ``xt``, selects the best candidate according to
   ``maximize_criterion``, evaluates the problem there, updates the model, and
   stores the selected index in ``history["eval_indices"]``.

.. py:class:: SequentialStrategySMC(problem, model, options=None)
   :no-index:

   Use ``gpmp.mcmc.smc.SMC`` particles as the candidate set.  Additional
   options include ``n_smc``, ``initial_distribution_type``, and
   ``update_method``.  Subclasses must define the SMC target through
   ``smc_log_density`` and ``update_smc_target_log_density_param``.

.. py:class:: SequentialStrategyBSS(problem, model, options=None)
   :no-index:

   Use an SMC candidate set for Bayesian subset simulation style updates.
   Additional options include ``n_smc`` and ``initial_distribution_type``.

.. automodule:: gpmpcontrib.sequentialprediction
   :members: SequentialPrediction
   :show-inheritance:

.. automodule:: gpmpcontrib.sequentialstrategy
   :members: SequentialStrategy, SequentialStrategyGridSearch, SequentialStrategySMC, SequentialStrategyBSS
   :show-inheritance:
