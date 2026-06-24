Provided Matérn container classes
=================================

The ``gpmpcontrib.models`` package contains preconfigured Matérn model classes.
They configure ``ModelContainer`` and ``gpmp.core.Model`` for common Matérn GP
models.  Each class installs adapters that build ``gpmp.parameter.Param``
objects from raw ``meanparam`` and ``covparam`` vectors.

Classes using REMAP selection build prior-regularized criteria when
``select_params`` is called.  For classes with priors, use ``get_prior`` and
``set_prior`` on the model container to inspect or set per-output prior values.
Use ``model[k].get_param()`` to inspect selected parameters in named form.

Constructor contracts
---------------------

``Model_ConstantMean_Maternp_ML(name, output_dim, covariance_specification=None)``
    Constant mean with one explicit mean parameter per output.  Parameters are
    selected by maximum likelihood.  ``covariance_specification`` must provide
    ``{"p": int}`` for the Matérn regularity.

``Model_ConstantMean_Maternp_REML(name, output_dim, mean_specification, covariance_specification)``
    Linear-predictor mean with covariance parameters selected by restricted
    likelihood.  ``mean_specification`` accepts ``{"type": "constant"}`` or
    ``{"type": "linear"}``.  ``covariance_specification`` must provide
    ``{"p": int}``.

``Model_ConstantMean_Maternp_REMAP_logsigma2(name, output_dim, mean_specification, covariance_specification, gamma=None, sigma2_coverage=None)``
    Matérn class using REMAP selection with a Gaussian prior on
    ``log(sigma^2)``.  Missing prior hyperparameters are resolved from
    ``gpmp.kernel.prior_defaults`` when ``select_params`` builds the criterion.

``Model_ConstantMean_Maternp_REMAP_logsigma2_and_logrho_prior(name, output_dim, mean_specification, covariance_specification, gamma=None, sigma2_coverage=None, alpha=None, rho_min_range_factor=None, logrho_min=None, covparam0_prior=None, logsigma2_0_prior=None, logrho_0_prior=None)``
    Matérn class using REMAP selection with priors on ``log(sigma^2)`` and
    ``logrho``.  Missing prior anchors are resolved from ``covparam0_prior``
    when provided, then from the anisotropic initial guess computed on the
    current observations.  Direct anchors ``logsigma2_0_prior`` and
    ``logrho_0_prior`` take priority over ``covparam0_prior``.

``Model_ConstantMean_Maternp_REMAP``
    Alias of
    ``Model_ConstantMean_Maternp_REMAP_logsigma2_and_logrho_prior``.

Prior access
------------

.. py:method:: set_prior(*, gamma, sigma2_coverage, alpha, rho_min_range_factor, logrho_min, covparam0_prior, logsigma2_0_prior, logrho_0_prior, output_idx=None)
   :no-index:

   Set prior values on classes using REMAP selection with priors.
   ``output_idx=None`` applies the supplied values to every output.
   ``covparam0_prior`` is in covariance coordinates
   ``[log(sigma2), -log(rho_0), ...]``.  ``logrho_0_prior`` is in ``logrho``
   coordinates, so it has the opposite sign from the stored lengthscale
   coordinates.

.. py:method:: get_prior(output_idx=None, resolved=True)
   :no-index:

   Return the resolved prior object for one output or a list of resolved prior
   objects for all outputs.  Raises ``ValueError`` before ``select_params`` when
   required defaults or data-dependent anchors have not yet been resolved.

.. automodule:: gpmpcontrib.models
   :members:
   :undoc-members:
   :show-inheritance:

Maximum likelihood models
-------------------------

.. automodule:: gpmpcontrib.models.models_ML
   :members:
   :undoc-members:
   :show-inheritance:

Classes with restricted likelihood selection
--------------------------------------------

.. automodule:: gpmpcontrib.models.models_REML
   :members:
   :undoc-members:
   :show-inheritance:

Classes with REMAP selection
----------------------------

.. automodule:: gpmpcontrib.models.models_REMAP
   :members:
   :undoc-members:
   :show-inheritance:

Noisy classes with restricted likelihood selection
--------------------------------------------------

.. automodule:: gpmpcontrib.models.models_noisy_REML
   :members:
   :undoc-members:
   :show-inheritance:
