Concepts
========

``gpmp`` contains the numerical GP objects: ``gpmp.core.Model``, covariance
functions, parameter-selection functions, diagnosis tools, posterior samplers,
and backend operations.  ``gpmp-contrib`` adds ``ComputerExperiment``,
``ModelContainer``, provided Matérn container classes, sequential strategies,
optimization and set-estimation criteria, benchmark problems, and reGP utilities.

Object map
----------

``ComputerExperiment``
    Describes an input domain and one or more callable outputs.  Outputs can be
    objectives, constraints, or generic scalar functions.  Constraint outputs
    may carry bounds that define feasibility.

``ModelContainer``
    Stores one independent ``gpmp.core.Model`` per output.  Each output entry
    stores the underlying model, selection criterion, raw parameter vectors,
    ``Param`` object, optimizer report, and resolved prior object when the model
    class uses REMAP priors.

Provided Matérn container classes
    Configure Matérn GP models on top of ``ModelContainer``.  They specify the
    mean form, covariance function, initial-guess procedure, selection criterion,
    and ``Param`` construction.

``SequentialPrediction``
    Stores observations and a model container.  It updates model parameters,
    computes predictions, and draws conditional simulations from the current
    model.

Sequential strategy classes
    Add a candidate set or particle set and a rule for selecting the next
    evaluation.  The implemented rules cover expected improvement, excursion
    sets, set inversion, and particle-based searches.

Common notation
---------------

The guide uses ``xi`` for observation points and ``zi`` for observed values.
Mathematically, write

.. math::

   X = (x_1,\ldots,x_n)^\top,\qquad
   z = (z_1,\ldots,z_n)^\top.

For a scalar output, the provided Matérn classes build a Gaussian process

.. math::

   Z(x) = m(x) + Z_0(x),

where :math:`Z_0` is centered and has covariance
:math:`K_\theta(x,x')`.  A multi-output ``ModelContainer`` stores one
independent scalar model per output.  Cross-output covariance is not modeled by
the provided container classes.

After conditioning on observations, predictions at a point :math:`x` are
Gaussian:

.. math::

   Z(x)\mid (X,z),\theta \sim
   \mathcal{N}\left(\mu_\theta(x), s_\theta^2(x)\right).

The arrays returned by ``predict`` are these posterior means and variances,
stacked over prediction points and outputs.

Relation to gpmp
----------------

The GP model is the standard kriging model used in spatial statistics and
computer experiments.  Background on covariance modeling and kriging is given by
:cite:p:`stein1999kriging,chiles1999geostatistics`.  In ``gpmp-contrib``, the
model formulas are not reimplemented.  Prediction, likelihood evaluation, REML,
REMAP, and posterior parameter sampling are delegated to ``gpmp``.

``gpmp-contrib`` decides how the model is assembled and how state is stored:
which covariance class is used, which parameter-selection criterion is called,
how optimizer starts are chosen, and how user-readable parameter objects are
created.

When to use gpmp-contrib
------------------------

Use ``gpmp-contrib`` when the task involves a computer experiment, a
multi-output model container, a sequential design, a provided benchmark problem,
or the reGP procedure.  Use ``gpmp`` directly when implementing a new covariance
function, selection criterion, MCMC kernel, backend operation, or diagnostic.
