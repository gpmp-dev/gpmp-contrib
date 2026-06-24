Model state and parameter objects
=================================

``gpmp-contrib`` stores selected parameters in two forms:

- raw backend vectors used by ``gpmp.core.Model`` and ``gpmp.kernel``.
- ``gpmp.parameter.Param`` objects used for names, paths, transforms, bounds,
  and readable display.

The raw vectors are the numerical state of the GP model.  The ``Param`` object
is an adapter on top of those vectors.  It helps users inspect and pass
parameters without memorizing the position and sign convention of each entry.

Per-output state
----------------

``ModelContainer`` stores one entry per output.  After parameter selection, use
``model[k]`` to inspect output ``k``:

.. code-block:: python

   model.select_params(xi, zi)

   entry = model[0]
   gp_model = entry["model"]

   covparam = gp_model.covparam
   meanparam = gp_model.meanparam
   param = entry.get_param()
   info = entry["info"]
   prior = entry.get("prior", None)

The entries have distinct roles:

``entry["model"]``
    Underlying ``gpmp.core.Model``.  It stores the raw ``meanparam`` and
    ``covparam`` backend vectors used by prediction, likelihood evaluation, and
    posterior sampling functions.

``entry.get_param()`` or ``entry["param"]``
    Readable ``Param`` object built from the current raw vectors.  It carries
    names such as ``sigma2`` and ``rho_0``, paths such as ``covparam`` and
    ``lengthscale``, normalization rules, and optional bounds.

``entry["info"]``
    Parameter-selection report.  It contains optimizer status, criterion
    wrappers, the optimizer start, and the final raw vectors.

``entry["prior"]``
    Resolved prior object for classes using REMAP selection with priors.  It
    exists after ``select_params`` has built the criterion.

Raw covariance parameters
-------------------------

For the provided Matérn container classes, the covariance vector is

.. code-block:: text

   covparam = [log(sigma2), -log(rho_0), ..., -log(rho_{d-1})]

The first entry is the logarithm of the process variance.  The remaining entries
use the inverse log-lengthscale convention.  In this convention, larger values
correspond to shorter lengthscales.

The raw vector is what ``gpmp.kernel`` receives.  Use it when calling
``gpmp.core`` or ``gpmp.kernel`` functions, or when writing a new selection
criterion.

Parameter objects
-----------------

The ``Param`` object is the user-facing representation of the same parameters:

.. code-block:: python

   param = model[0].get_param()
   print(param)

For the seeded Hartmann4 example in :doc:`../getting_started`, the display is:

.. literalinclude:: ../_static/example_results/getting_started_hartmann4_param.txt
   :language: text

The displayed text is produced by ``print(model[0].get_param())`` after
``model.select_params(xi, zi)``.

For Matérn covariance parameters, ``rho`` entries are displayed as lengthscales
even though the raw vector stores ``-log(rho)``.  The ``Denorm`` column is the
physical value.

Use the object when you need named access:

.. code-block:: python

   cov_entries = param.get_by_path(["covparam"], prefix_match=True)
   rho_entries = param.get_by_path(
       ["covparam", "lengthscale"], prefix_match=True
   )

Use ``param`` as an optimizer start when the start should retain names and
bounds:

.. code-block:: python

   model.select_params(xi, zi, param0=model[0].get_param())

For multi-output models, pass one vector or one ``Param`` object per output:

.. code-block:: python

   starts = [model[k].get_param() for k in range(model.output_dim)]
   model.select_params(xi, zi, param0=starts)

Bounds
------

``Param.bounds`` stores bounds in normalized coordinates.  The normalized
coordinates are the coordinates optimized by SciPy.  Bounds passed directly to
``select_params`` take precedence over bounds stored in the ``Param`` object.

The precedence is:

1. explicit ``bounds``.
2. bounds returned by ``bounds_factory``.
3. bounds stored in the current ``Param`` object.
4. no bounds.

When ``bounds_delta`` is provided, ``select_params`` intersects the chosen bounds
with a local interval around the optimizer start.

Optimizer information
---------------------

The ``info`` object stores both the optimizer report and values needed for later
diagnostics:

.. code-block:: python

   info = model[0]["info"]

   start = info["param0"]
   cov_start = info["covparam0"]
   cov_final = info["covparam"]
   criterion = info["selection_criterion_nograd"]

``selection_criterion_nograd`` is the criterion to use for diagnosis, plotting,
and posterior sampling when gradients are not needed.

Prior objects
-------------

Classes using REMAP selection with priors store resolved prior objects per
output:

.. code-block:: python

   prior_from_entry = model[0]["prior"]
   prior = model.get_prior(0)

The prior object records the hyperparameters and anchors used by the criterion.
It is separate from the selected GP parameters.  For example,
``prior.covparam0`` is the covariance vector used as the prior anchor, while
``model[0]["model"].covparam`` is the selected covariance vector.

Use :doc:`priors` to set prior values and inspect resolved prior fields.
