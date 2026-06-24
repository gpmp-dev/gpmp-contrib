Numerical backend objects
=========================

``gpmp-contrib`` follows the numerical backend selected by ``gpmp``.  The
backend controls the concrete object type returned by model calls.

Default rule
------------

Do not add conversions unless a boundary requires them.  ``ModelContainer``
methods convert inputs by default.  Prediction, LOO, and
conditional-simulation methods return NumPy arrays when ``convert_out=True``.

Use ``gpmp.num`` when writing backend-independent code around ``gpmp.core`` or
``gpmp.kernel`` calls:

.. code-block:: python

   import gpmp.num as gnp

   x = gnp.asarray(x)

With the NumPy backend, backend objects are NumPy arrays.  With the PyTorch
backend, backend objects are PyTorch tensors.  Arrays returned by plotting,
``ComputerExperiment`` evaluation, or calls with ``convert_out=True`` may be
NumPy arrays even when the active backend is PyTorch.

Common method conventions
-------------------------

``convert_in=True``
    Convert input arrays to backend objects before calling ``gpmp``.

``convert_out=True``
    Convert returned arrays to NumPy arrays.

``convert_out=False``
    Return backend objects.  Use this when the result will be passed to another
    backend-aware ``gpmp`` or ``gpmp-contrib`` function.

``gpmp`` selection criteria
    Criteria used for optimization and posterior sampling are backend-aware.
    When a method records criterion values, the stored array follows the
    backend unless the method explicitly converts it.

Practical checks
----------------

Use explicit conversions only at boundaries:

.. code-block:: python

   zpm, zpv = model.predict(xi, zi, xt, convert_out=False)
   zpm_np = gnp.to_np(zpm)

Avoid direct calls to ``numpy.asarray`` on PyTorch tensors that may carry a
gradient graph.  Use ``gpmp.num.to_np`` at that boundary.  Use
``gpmp.num.to_scalar`` only when an external API or a log message needs a Python
scalar.
