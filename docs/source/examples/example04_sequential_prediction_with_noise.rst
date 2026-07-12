Example 04: noisy sequential prediction
=======================================

Complete script: `examples/example04_sequential_prediction_with_noise.py <https://github.com/gpmp-dev/gpmp-contrib/blob/main/examples/example04_sequential_prediction_with_noise.py>`_

Purpose
-------

The script uses ``SequentialPrediction`` with noisy scalar observations.  The
model class is ``Model_Noisy_ConstantMean_Maternp_REML``.
The physical input is augmented with one observation-noise variance column, so
observations and prediction points carry different noise values.  The noise
variance is added
to the observation covariance :cite:p:`stein1999kriging,chiles1999geostatistics`.

What is computed
----------------

- noisy observations of the one-dimensional ``twobumps`` function.
- covariance parameters selected by noisy REML.
- posterior means and variances at zero-noise prediction points.
- sequential additions selected by maximum posterior variance.
- conditional sample paths from the final noisy model.

Main objects
------------

- ``gpmpcontrib.computerexperiment.StochasticComputerExperiment``
- ``gpmpcontrib.Model_Noisy_ConstantMean_Maternp_REML``
- ``gpmpcontrib.SequentialPrediction``
- ``gpmpcontrib.plot.plot_1d``

Outputs
-------

Run ``python examples/example04_sequential_prediction_with_noise.py`` from the
repository root to execute the example.  Regenerate the static figure with
``cd docs && python make_example_results.py``.

.. figure:: ../_static/example_results/example04_noisy_prediction.png
   :alt: Noisy one-dimensional GP prediction with replicated observations.
   :width: 85%

   Red points are noisy observations.  Several observations are replicated at
   the same physical input.  The prediction grid is passed with zero observation
   noise, so the posterior variance represents uncertainty on the latent
   function, not observation noise at a new noisy measurement.

Source excerpt
--------------

.. literalinclude:: ../../../examples/example04_sequential_prediction_with_noise.py
   :language: python
   :lines: 45-86
