Provided Matérn container classes
=================================

``gpmpcontrib.models`` provides configured ``ModelContainer`` subclasses.  Each
container stores one independent ``gpmp.core.Model`` per output and builds a
``gpmp.parameter.Param`` object from the selected raw vectors.  The common
``select_params``, ``predict``, ``run_diagnosis``, and state-access contracts are
documented in :doc:`modelcontainer`.

Naming and covariance parameters
--------------------------------

``Maternp`` fixes the Matérn regularity at :math:`\nu=p+1/2` and uses

.. code-block:: text

   [log(sigma2), -log(rho_0), ..., -log(rho_{d-1})]

``Matern`` selects :math:`\nu` and uses

.. code-block:: text

   [log(sigma2), log(nu), -log(rho_0), ..., -log(rho_{d-1})]

The corresponding ``Param`` paths are ``covparam/variance``,
``covparam/regularity``, and ``covparam/lengthscale``.  Variance and regularity
use logarithmic normalization.  Lengthscales use inverse-log normalization.

Constructor contracts
---------------------

.. py:class:: Model_ConstantMean_Maternp_ML(name, output_dim, covariance_specification=None)
   :no-index:

   Build ``output_dim`` fixed-regularity Matérn models.  Each output has one
   explicit constant-mean parameter.  Mean and covariance parameters are
   selected by maximum likelihood.  ``covariance_specification`` must contain
   ``{"p": int}`` unless a custom covariance function is supplied.

.. py:class:: Model_ConstantMean_Maternp_REML(name, output_dim, mean_specification, covariance_specification)
   :no-index:

   Build fixed-regularity Matérn models with covariance parameters selected by
   restricted likelihood.  ``mean_specification`` accepts
   ``{"type": "constant"}`` or ``{"type": "linear"}``.
   ``covariance_specification`` must contain ``{"p": int}`` unless a custom
   covariance function is supplied.

.. py:class:: Model_ConstantMean_Matern_REML(name, output_dim, mean_specification, covariance_specification=None)
   :no-index:

   Build Matérn models whose variance, regularity, and componentwise
   lengthscales are selected by restricted likelihood.  A ``d``-dimensional
   input gives ``d + 2`` covariance parameters.  ``covariance_specification``
   is normally omitted; the class uses ``gp.kernel.matern_covariance`` and its
   default numerical policy.

   The initial guess is computed by
   ``gp.kernel.anisotropic_parameters_initial_guess_matern``.  After
   ``select_params``, use ``model[k].get_param()`` and path
   ``["covparam", "regularity"]`` to read ``log(nu)``.  Exponentiation gives
   the selected positive regularity.

.. py:class:: Model_ConstantMean_Maternp_REMAP_logsigma2(name, output_dim, mean_specification, covariance_specification, gamma=None, sigma2_coverage=None)
   :no-index:

   Build fixed-regularity Matérn models selected by REMAP with a Gaussian prior
   on ``log(sigma2)``.  Missing hyperparameters are resolved from
   ``gpmp.kernel.prior_defaults`` when ``select_params`` builds the criterion.

.. py:class:: Model_ConstantMean_Maternp_REMAP_logsigma2_and_logrho_prior(name, output_dim, mean_specification, covariance_specification, gamma=None, sigma2_coverage=None, alpha=None, rho_min_range_factor=None, logrho_min=None, covparam0_prior=None, logsigma2_0_prior=None, logrho_0_prior=None)
   :no-index:

   Build fixed-regularity Matérn models selected by REMAP with priors on
   ``log(sigma2)`` and ``logrho``.  Direct anchors ``logsigma2_0_prior`` and
   ``logrho_0_prior`` take priority over ``covparam0_prior``.  Missing anchors
   are obtained from the anisotropic initial guess on the current observations.

.. py:class:: Model_ConstantMean_Maternp_REMAP
   :no-index:

   Alias of
   ``Model_ConstantMean_Maternp_REMAP_logsigma2_and_logrho_prior``.

.. py:class:: Model_Noisy_ConstantMean_Maternp_REML(name, output_dim, mean_specification, covariance_specification)
   :no-index:

   Build fixed-regularity Matérn models with known, point-dependent observation
   noise.  The input matrix contains physical coordinates followed by one noise
   variance column per output.  Prediction points use zero in these columns
   when prediction concerns the latent function.

Parameter adapters
------------------

``Model_ConstantMean_Matern_REML.build_param_procedures`` accepts the optional
arguments ``logsigma2_bounds``, ``lognu_bounds``, ``loginvrho_bounds``, and
``name_prefix``.  It returns ``(param_from_vectors, vectors_from_param)``.
Bounds use the normalized coordinates stored in ``Param``:

.. code-block:: python

   import gpmp.num as gnp

   param_from_vectors, vectors_from_param = model.build_param_procedures(
       0,
       logsigma2_bounds=(-10.0, 10.0),
       lognu_bounds=(-2.0, 4.0),
       loginvrho_bounds=(-8.0, 8.0),
   )
   param = param_from_vectors(gnp.asarray([]), covparam)

The first callable accepts ``(meanparam, covparam)`` and returns a ``Param``
object.  The second accepts a ``Param`` object and returns
``(meanparam, covparam)``.

Prior access
------------

.. py:method:: Model_ConstantMean_Maternp_REMAP_logsigma2.set_prior(*, gamma=..., sigma2_coverage=..., covparam0_prior=..., logsigma2_0_prior=..., output_idx=None)
   :no-index:

   Update the supplied log-variance prior fields.  Omitted fields remain
   unchanged.  ``output_idx=None`` applies each update to every output.

.. py:method:: Model_ConstantMean_Maternp_REMAP_logsigma2_and_logrho_prior.set_prior(*, gamma=..., sigma2_coverage=..., alpha=..., rho_min_range_factor=..., logrho_min=..., covparam0_prior=..., logsigma2_0_prior=..., logrho_0_prior=..., output_idx=None)
   :no-index:

   Update the supplied log-variance and log-lengthscale prior fields.  Omitted
   fields remain unchanged.  ``output_idx=None`` applies each update to every
   output.  ``covparam0_prior`` uses covariance coordinates
   ``[log(sigma2), -log(rho_0), ...]``.  ``logrho_0_prior`` uses ``logrho`` and
   therefore has the opposite sign on lengthscale coordinates.

.. py:method:: get_prior(output_idx=None, resolved=True)
   :no-index:

   Return one resolved prior object or a list with one object per output.
   ``resolved=True`` raises ``ValueError`` before data-dependent defaults have
   been resolved by ``select_params``.
