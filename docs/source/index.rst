GPmp-contrib documentation
==========================

``gpmp-contrib`` extends ``gpmp`` with computer-experiment objects, model
containers, Matérn container classes, sequential strategies, optimization
criteria, excursion/set-inversion tools, and relaxed Gaussian-process
utilities.

The provided ``Maternp`` classes fix the Matérn regularity at
:math:`\nu=p+1/2`.  ``Model_ConstantMean_Matern_REML`` selects :math:`\nu`
with the variance and lengthscales.

Use ``gpmp`` directly when you need models, kernels, numerical backends, or
parameter-selection functions.  Use ``gpmp-contrib`` when you want computer
experiments, model containers, or sequential-design classes built from these
objects.

Documentation contents
----------------------

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

Technical entry points
----------------------

Hartmann4 example
    :doc:`getting_started` builds the Hartmann4 example used in the core
    ``gpmp`` tutorial, selects Matérn covariance parameters, predicts at test
    points, and shows the expected diagnostic output.

Model construction
    :doc:`guide/models`, :doc:`guide/model_state`, and
    :doc:`guide/parameter_selection` describe model containers,
    fixed or selected Matérn regularity, parameter-selection rules, stored
    covariance vectors, and readable ``Param`` objects.

Priors in REMAP selection
    :doc:`guide/priors` documents ``set_prior``, ``get_prior``, prior anchors,
    and REMAP hyperparameters.

Sequential design
    :doc:`guide/sequential` describes expected improvement, excursion sets,
    set inversion, SMC particles, and BSS particles.

API details
    :doc:`api/index` gives signatures, shapes, return values, stored side
    effects, and failure conditions.

.. toctree::
   :maxdepth: 2

   installation
   getting_started
   guide/index
   examples/index
   api/index
   references
