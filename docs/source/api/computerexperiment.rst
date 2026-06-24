Computer experiments
====================

Computer experiments define the input domain and output structure of an
objective, constraint, or multi-output simulator.  They are plain Python objects
that evaluate user callables and return numerical backend-compatible arrays.

``ComputerExperiment``
----------------------

.. py:class:: ComputerExperiment(input_dim, input_box, *, single_function=None, function_list=None, single_objective=None, objective_list=None, single_constraint=None, constraint_list=None, constraint_bounds=None)
   :no-index:

   Define a deterministic computer experiment.

   ``input_dim`` is the input dimension ``d``.  ``input_box`` can have shape
   ``(2, d)`` with lower and upper rows or shape ``(d, 2)`` with one
   ``(lower, upper)`` row per coordinate.  Bounds must be finite and upper
   bounds must be larger than lower bounds.  The object stores the box with
   shape ``(2, d)``.

   A function specification can be a callable or a dictionary.  Dictionary keys
   include:

   - ``"function"``: callable accepting ``x`` with shape ``(n, d)``.
   - ``"output_dim"``: number of outputs from that callable.
   - ``"type"``: ``"objective"``, ``"constraint"``, or a list of such labels.
   - ``"bounds"``: feasibility bounds for constraint outputs.

   Use either combined functions through ``single_function`` or
   ``function_list``, or separated objectives and constraints through
   ``single_objective``, ``objective_list``, ``single_constraint``, and
   ``constraint_list``.

   .. py:method:: eval(x, *, normalize_input=None)
      :no-index:

      Evaluate all outputs.  ``x`` has shape ``(n, d)`` or ``(d,)``.  Returns a
      NumPy array with shape ``(n, output_dim)``.  When ``normalize_input`` is
      true, ``x`` is interpreted in ``[0, 1]^d`` and mapped to the physical
      input box before evaluation.

   .. py:method:: eval_objectives(x, *, normalize_input=None)
      :no-index:

      Evaluate objective outputs only.  Returns shape ``(n, n_objectives)``.
      Returns an empty second dimension when no objective is registered.

   .. py:method:: eval_constraints(x, *, normalize_input=None)
      :no-index:

      Evaluate constraint outputs only.  Returns shape ``(n, n_constraints)``.
      Returns an empty second dimension when no constraint is registered.

   The object caches the last deterministic evaluation.  It raises
   ``ValueError`` for invalid input boxes, mutually exclusive construction
   paths, incompatible output dimensions, missing constraint bounds, or input
   arrays with the wrong shape.

``StochasticComputerExperiment``
--------------------------------

.. py:class:: StochasticComputerExperiment(input_dim, input_box, *, single_function=None, function_list=None, single_objective=None, objective_list=None, single_constraint=None, constraint_list=None, simulated_noise_variance=None)
   :no-index:

   Extend ``ComputerExperiment`` with additive Gaussian output noise.

   ``simulated_noise_variance`` can be a scalar or a list with one variance per
   output.  Per-function dictionaries may also contain
   ``"simulated_noise_variance"``.

   .. py:method:: eval(x, *, simulate_noise=True, batch_size=1, rng=None)
      :no-index:

      Evaluate all outputs.  With ``batch_size=1``, returns a NumPy array with
      shape ``(n, output_dim)``.  With ``batch_size > 1``, returns shape
      ``(n, output_dim, batch_size)``.  ``rng`` can be a
      ``numpy.random.Generator`` for reproducible noise draws.

   Stochastic evaluations do not use the deterministic cache.  The
   ``eval_objectives`` and ``eval_constraints`` shortcuts are not supported for
   stochastic experiments because repeated calls would draw different noise.

.. automodule:: gpmpcontrib.computerexperiment
   :members:
   :undoc-members:
   :show-inheritance:
