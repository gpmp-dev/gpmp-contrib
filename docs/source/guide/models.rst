Models and computer experiments
===============================

Model construction starts with a ``ComputerExperiment`` and a
``ModelContainer``.  The ``ComputerExperiment`` describes the callable problem:
input dimension, input box, outputs, and optional constraint bounds.  The
``ModelContainer`` stores one ``gpmp.core.Model`` per output and manages
parameter selection, prediction, diagnosis, and stored state.

Prediction and parameter-selection computations are done by ``gpmp``.  The
container classes in ``gpmp-contrib`` configure the mean form, covariance
function, initial-guess procedure, selection criterion, and per-output state.

Shapes
------

Array shapes:

``xi``
    Observation points, shape ``(n, d)``.

``zi``
    Observed values.  Shape ``(n,)`` or ``(n, 1)`` for one output, and
    ``(n, output_dim)`` for several outputs.

``xt``
    Prediction points, shape ``(m, d)``.

``zpm`` and ``zpv``
    Posterior means and variances returned by ``predict``.  Shape
    ``(m, output_dim)``.

ComputerExperiment
------------------

``ComputerExperiment`` defines the input dimension, input box, and outputs.  It
can wrap one callable, separate objective and constraint callables, or callable
metadata for multiple outputs.

.. code-block:: python

   import numpy as np
   import gpmpcontrib as gpc

   def objective(x):
       return (x[:, 0] - 0.25) ** 2 + (x[:, 1] - 0.75) ** 2

   def constraint(x):
       return x[:, 0] + x[:, 1]

   problem = gpc.ComputerExperiment(
       2,
       [[0.0, 0.0], [1.0, 1.0]],
       single_objective=objective,
       single_constraint={"function": constraint, "bounds": [0.5, np.inf]},
   )

``problem(x)`` evaluates all outputs.  ``problem.eval_objectives(x)`` and
``problem.eval_constraints(x)`` return the corresponding blocks.

ModelContainer
--------------

``ModelContainer`` is a per-output container.  For output ``k``, ``model[k]``
stores an entry with the underlying ``gpmp.core.Model`` and the state built by
``select_params``:

.. code-block:: python

   model.select_params(xi, zi)

   entry = model[0]
   gp_model = entry["model"]
   param = entry.get_param()
   info = entry["info"]

The entry is described in :doc:`model_state`.  Raw vectors are stored in
``entry["model"]``.  Readable parameter information is stored in
``entry.get_param()``.

Matérn model used by the provided classes
-----------------------------------------

For each output, the provided Matérn classes define a scalar-valued GP

.. math::

   Z(x) = m(x) + Z_0(x),

where :math:`Z_0` is a centered GP with Matérn covariance
:math:`K_\theta`.  The mean term depends on the class:

- ``Model_ConstantMean_Maternp_ML`` uses a parameterized constant mean and
  selects the mean coefficient with the covariance parameters.
- ``Model_ConstantMean_Maternp_REML`` and REMAP subclasses use a
  linear-predictor mean with constant or linear basis functions.  The criterion
  is optimized over covariance parameters.

The covariance parameter vector is

.. code-block:: text

   covparam = [log(sigma2), -log(rho_0), ..., -log(rho_{d-1})]

Thus

.. math::

   \sigma^2 = \exp(\texttt{covparam[0]}),
   \qquad
   \rho_j = \exp(-\texttt{covparam[j + 1]}).

The raw lengthscale coordinates are ``-log(rho_j)``.  Larger raw values mean
shorter physical lengthscales.

Posterior prediction
--------------------

For fixed covariance parameters and a fixed mean, the conditional-Gaussian
prediction formula has the form

.. math::

   \mu_\theta(x)
   = m(x) + k_\theta(x, X) K_\theta(X, X)^{-1}(z - m(X)),

and

.. math::

   s_\theta^2(x)
   = K_\theta(x, x)
     - k_\theta(x, X) K_\theta(X, X)^{-1} k_\theta(X, x).

Here :math:`K_\theta(X,X)` is the observation covariance matrix and
:math:`k_\theta(x,X)` is the row vector of covariances between :math:`x` and
the observation points.  The provided classes call ``gpmp.core.Model`` for this
computation.  When the mean is parameterized or profiled, ``gpmp`` applies the
corresponding kriging formulas internally.

Mean and covariance specifications
----------------------------------

The Hartmann4 examples in this guide use:

.. code-block:: python

   mean_specification = {"type": "constant"}
   covariance_specification = {"p": 2}

``p`` is the Matérn smoothness parameter used by ``gpmp.kernel``.  The exact
set of accepted specifications depends on the selected container class.

Provided Matérn container classes
---------------------------------

``Model_ConstantMean_Maternp_ML``
    Constant mean with explicit mean parameter.  Mean and covariance parameters
    are selected by maximum likelihood.

``Model_ConstantMean_Maternp_REML``
    Linear-predictor mean.  Covariance parameters are selected by restricted
    likelihood.

``Model_ConstantMean_Maternp_REMAP``
    Current default REMAP class.  It aliases
    ``Model_ConstantMean_Maternp_REMAP_logsigma2_and_logrho_prior``.

``Model_ConstantMean_Maternp_REMAP_logsigma2``
    REMAP class with a Gaussian prior on ``log(sigma^2)``.

``Model_ConstantMean_Maternp_REMAP_logsigma2_and_logrho_prior``
    REMAP class with a Gaussian prior on ``log(sigma^2)`` and a barrier-linear
    prior on ``logrho``.

``Model_Noisy_ConstantMean_Maternp_REML``
    Noisy-observation class.  The input matrix carries physical coordinates and
    observation-noise variance columns expected by the noisy covariance wrapper.

Constructor arguments
---------------------

The provided Matérn classes share the same core constructor arguments:

``name``
    Label used in displays and stored model entries.

``output_dim``
    Number of scalar outputs.  The container builds one independent
    ``gpmp.core.Model`` per output.

``mean_specification``
    Dictionary describing the mean form.  For the examples in this guide,
    ``{"type": "constant"}`` is used.

``covariance_specification``
    Dictionary describing the Matérn covariance.  The key ``"p"`` selects the
    Matérn smoothness parameter passed to ``gpmp.kernel``.

REMAP constructors also accept optional prior hyperparameters.  The same values
can be set later with ``set_prior`` before ``select_params``.  See
:doc:`priors`.

Minimal construction
--------------------

.. code-block:: python

   model = gpc.Model_ConstantMean_Maternp_REML(
       "hartmann4",
       output_dim=problem.output_dim,
       mean_specification={"type": "constant"},
       covariance_specification={"p": 3},
   )

   model.select_params(xi, zi)
   zpm, zpv = model.predict(xi, zi, xt)

After this call, ``model[0]["model"].covparam`` contains the selected raw
covariance vector, ``model[0].get_param()`` gives the readable parameter object,
and ``model[0]["info"]`` stores the optimizer report.

Where to continue
-----------------

- Use :doc:`parameter_selection` to understand optimizer starts, bounds, and
  ML/REML/REMAP criteria.
- Use :doc:`model_state` to inspect stored vectors and parameter objects.
- Use :doc:`diagnostics` to interpret the printed summaries.
