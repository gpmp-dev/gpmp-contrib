Test problems
=============

``gpmpcontrib.test_problems`` collects deterministic benchmark problems as
``ComputerExperiment`` instances.  They cover scalar, constrained, and
multi-output examples used by the example scripts.

Several scalar problems follow standard analytical benchmark functions.  For
definitions and variants of many such functions, see the Virtual Library of
Simulation Experiments test-function collection :cite:p:`surjanovic_bingham`.

Import the module with:

.. code-block:: python

   import gpmpcontrib as gpc

   problem = gpc.test_problems.branin
   problem.normalize_input = True
   z = problem([[0.5, 0.5]])

Problem object contract
-----------------------

Each problem listed here is a ``ComputerExperiment`` instance.  Common fields
are:

- ``input_dim``: input dimension.
- ``input_box``: current input box with shape ``(2, input_dim)``.
- ``input_box_org``: original physical input box.
- ``output_dim``: total number of objective and constraint outputs.
- ``normalize_input``: when true, calls interpret inputs in ``[0, 1]^d``.

Calling ``problem(x)`` or ``problem.eval(x)`` returns a NumPy array with shape
``(n, output_dim)``.  ``eval_objectives`` and ``eval_constraints`` return the
corresponding output blocks.  Constraint outputs use the bounds stored in the
problem metadata.

Frequently used scalar problems include:

- ``twobumps``
- ``branin`` and ``branin100``
- ``hartmann3`` and ``hartmann6``
- ``ishigami``
- ``borehole``
- ``linkletter``
- ``levitan``

Frequently used constrained or multi-output problems include:

- ``test_problem02``, ``test_problem03``, ``test_problem04``
- ``bnh``
- ``tnk_experiment``
- ``constr_experiment``
- ``welded_beam_design_experiment``
- ``two_bar_truss_experiment``

Factory functions are available for parameterized dimensions:

Factory functions return ``ComputerExperiment`` instances with one scalar
objective.  The argument ``d`` is the input dimension except for
``create_shekel_problem(m)``, where ``m`` selects the number of Shekel terms and
the input dimension is four.

.. currentmodule:: gpmpcontrib.test_problems

.. autofunction:: create_ackley_problem
.. autofunction:: create_rastrigin_problem
.. autofunction:: create_rosenbrock_problem
.. autofunction:: create_schwefel_problem
.. autofunction:: create_dixon_price_problem
.. autofunction:: create_trid_problem
.. autofunction:: create_perm_problem
.. autofunction:: create_michalewicz_problem
.. autofunction:: create_zakharov_problem
.. autofunction:: create_shekel_problem
