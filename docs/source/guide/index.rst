User guide
==========

The guide documents how ``gpmp-contrib`` stores models, parameters,
diagnostics, priors, and sequential-procedure state.  The runnable Hartmann4
sequence is in :doc:`../getting_started`.

Model construction and inspection
---------------------------------

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

Procedure-specific topics
-------------------------

- :doc:`priors` covers REMAP models with explicit prior anchors or
  hyperparameters.
- :doc:`sequential` covers expected improvement, excursion sets, set inversion,
  SMC particles, and BSS particles.
- :doc:`regp` covers the relaxed Gaussian-process procedure.

Examples and API links
----------------------

The guide describes objects and stored state.  :doc:`../examples/index` gives
complete scripts, plots, and representative results.  :doc:`../api/index`
gives signatures, shapes, return values, stored side effects, and failure
conditions.

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
