GPmp-contrib documentation
==========================

``gpmp-contrib`` extends ``gpmp`` with computer-experiment objects, model
containers, Matérn container classes, sequential strategies, optimization
criteria, excursion/set-inversion tools, and relaxed Gaussian-process
utilities.

Use ``gpmp`` directly when you need models, kernels, numerical backends, or
parameter-selection functions.  Use ``gpmp-contrib`` when you want computer
experiments, model containers, or sequential-design classes that coordinate
these objects.

For standard ``ModelContainer`` calls, model methods convert inputs by default
and prediction calls return NumPy arrays with ``convert_out=True``.  Explicit
backend conversion is needed when code passes arrays between ``gpmp`` internals,
posterior samplers, or external NumPy-only libraries.

Main documentation sections
---------------------------

Installation
    Package dependencies, backend notes, and documentation build commands.

Getting started
    Hartmann4 run: define a problem, build a model, select parameters, predict,
    and inspect diagnostics.

User guide
    Concepts and procedures: package organization, model containers, parameter
    objects, parameter selection, priors, diagnostics, sequential strategies,
    and reGP.

Examples
    Script-oriented explanations for the main examples in ``examples/``.

API reference
    Public modules, classes, and functions.

References
    Literature cited by the guide and examples.

Main procedures
---------------

First example
    See :doc:`getting_started`.  It builds the Hartmann4 example used in the
    core ``gpmp`` tutorial, selects Matérn covariance parameters, predicts at
    test points, and shows the expected diagnostic output.

Model construction
    Read :doc:`guide/models`, :doc:`guide/model_state`, and
    :doc:`guide/parameter_selection` to choose a likelihood,
    restricted-likelihood, or REMAP parameter-selection rule and to understand
    stored parameter values.

Priors in REMAP selection
    Read :doc:`guide/priors` when REMAP prior anchors or hyperparameters must
    be inspected or set by hand.

Sequential design
    Read :doc:`guide/sequential`, then compare the fixed-grid and SMC examples.

API details
    Use :doc:`api/index` for signatures, shapes, return values, stored side
    effects, and failure conditions.

.. toctree::
   :maxdepth: 2

   installation
   getting_started
   guide/index
   examples/index
   api/index
   references
