User guide
==========

The guide starts after the Hartmann4 run in :doc:`../getting_started`.  It
explains how ``gpmp-contrib`` stores models, parameters, diagnostics, priors,
and sequential-procedure state.

Main path
---------

Start with these pages if the goal is to build and inspect a Gaussian-process
model from observations:

1. :doc:`concepts` gives the package map and states which objects belong to
   ``gpmp`` and which objects are added by ``gpmp-contrib``.
2. :doc:`models` explains ``ComputerExperiment``, ``ModelContainer``, the
   provided Matérn classes, and the shape conventions for ``xi`` and ``zi``.
3. :doc:`parameter_selection` explains what ``select_params`` does, how
   optimizer starts and bounds are chosen, and when ML, REML, or REMAP is used.
4. :doc:`model_state` shows where selected parameters, ``Param`` objects,
   optimizer reports, and resolved prior objects are stored.
5. :doc:`diagnostics` explains the printed diagnosis and prediction-performance
   summaries.

Specialized procedures
----------------------

Read these pages when the model is already clear and the task needs one of
these procedures:

- :doc:`priors` for REMAP models with explicit prior anchors or hyperparameters.
- :doc:`sequential` for expected improvement, excursion sets, set inversion,
  SMC particles, and BSS particles.
- :doc:`regp` for relaxed Gaussian-process modeling.

Relation to examples
--------------------

The guide describes objects and stored state.  The example pages show complete
scripts, plots, and representative results.  After reading a guide page, use
:doc:`../examples/index` to find the closest executable script.

.. toctree::
   :maxdepth: 2

   concepts
   models
   parameter_selection
   model_state
   diagnostics
   priors
   sequential
   regp
