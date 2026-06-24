Plotting functions
==================

Plotting functions provide visualizations for one-dimensional GP models,
truth-vs-prediction diagnostics, LOO errors, matrix plots, and parallel
coordinates.

All Matplotlib functions create or update figures and call ``show``.  They are
intended for interactive examples and diagnostics.  Convert backend tensors to
NumPy arrays before passing them to functions that use Matplotlib directly.

.. automodule:: gpmpcontrib.plot.visualization
   :members: plot_1d, show_truth_vs_prediction, show_loo_errors, plotmatrix
   :show-inheritance:

.. py:function:: plot_1d(xt, zt, xi, zi, zpm, zpv, zpsim=None, xnew=None, title=None)
   :no-index:

   Plot a one-dimensional GP prediction.  ``xt`` and ``xi`` are input
   locations.  ``zt`` is an optional reference curve.  ``zpm`` and ``zpv`` are
   posterior mean and variance arrays on ``xt``.  ``zpsim`` can contain
   conditional sample paths with one path per column.

.. py:function:: show_truth_vs_prediction(zt, zpm)
   :no-index:

   Draw one scatter plot per output comparing reference values ``zt`` and
   posterior means ``zpm``.  Both arrays have shape ``(m, output_dim)``.

.. py:function:: show_loo_errors(zi, zloom, zloov)
   :no-index:

   Draw one LOO plot per output.  ``zi``, ``zloom``, and ``zloov`` have shape
   ``(n, output_dim)``.

.. py:function:: plotmatrix(data, colors=None)
   :no-index:

   Draw pairwise scatter plots and marginal histograms for ``data`` with shape
   ``(n, d)``.  ``colors`` can provide one scalar color value per row.

.. py:function:: parallel_coordinates_plot(x, z, p=None, show_p=False, xi=None, zi=None, ci=None, show_type=False)
   :no-index:

   Draw a Plotly parallel-coordinates plot for multivariate data.
   ``x`` has shape ``(n, d)`` and ``z`` has shape ``(n, q)``.  Optional
   ``xi`` and ``zi`` add a second data set with matching dimensions.  Returns a
   Plotly ``go.Figure``.
