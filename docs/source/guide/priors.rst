Priors for REMAP selection
==========================

REMAP selection uses

.. math::

   J_\mathrm{REMAP}(\theta) = J_\mathrm{REML}(\theta) - \log \pi(\theta).

Only REMAP container classes use the prior objects described here.  ML and REML
classes do not build these priors.

Inspect and set priors
----------------------

Set prior values before ``select_params``.  The example below uses the same
Hartmann4 observations as :doc:`../getting_started`:

.. code-block:: python

   model = gpc.Model_ConstantMean_Maternp_REMAP(
       "hartmann4",
       output_dim=problem.output_dim,
       mean_specification={"type": "constant"},
       covariance_specification={"p": 3},
   )

   model.set_prior(
       gamma=1.5,
       sigma2_coverage=0.95,
       alpha=1.0,
       output_idx=0,
   )

   model.select_params(xi, zi)
   prior = model.get_prior(0)
   print(prior)

The resolved prior object is:

.. literalinclude:: ../_static/example_results/guide_hartmann4_remap_prior.txt
   :language: text

The displayed text is produced by ``print(model.get_prior(0))`` after
``model.select_params(xi, zi)`` has resolved the REMAP prior.

``get_prior`` returns the resolved prior object.  It raises if called before
``select_params`` has built and resolved the criterion.

When ``output_idx`` is omitted, scalar values are applied to every output.
Vector fields such as ``covparam0_prior`` and ``logrho_min`` are replicated
across outputs unless a list of per-output vectors is provided.

Available prior objects
-----------------------

``LogSigma2Prior``
    Stores the Gaussian prior on ``log(sigma^2)``.

``LogSigma2AndLogRhoPrior``
    Stores the Gaussian prior on ``log(sigma^2)`` and the barrier-linear prior
    on ``logrho``.

The default ``Model_ConstantMean_Maternp_REMAP`` uses
``LogSigma2AndLogRhoPrior``.

Log-variance prior
------------------

The log-variance prior is centered on ``log_sigma2_0``.  The user controls its
scale through ``gamma`` and ``sigma2_coverage``:

.. math::

   \Pr\left(\sigma^2/\sigma_0^2 \in [1/\gamma, \gamma]\right)
   = \texttt{sigma2_coverage}.

Equivalently,

.. math::

   \log(\sigma^2)
   \sim \mathcal{N}\left(\log(\sigma_0^2), s_\sigma^2\right),
   \qquad
   s_\sigma =
   \frac{\log(\gamma)}
        {\Phi^{-1}\left((1 + \texttt{sigma2_coverage})/2\right)}.

The additive contribution to the REMAP objective is therefore

.. math::

   \frac{1}{2}
   \left(
      \frac{\log(\sigma^2) - \log(\sigma_0^2)}
           {s_\sigma}
   \right)^2,

up to a constant independent of the covariance parameters.

Larger ``gamma`` weakens the regularization.  Smaller ``sigma2_coverage`` also
weakens it.  The anchor can be supplied directly through
``logsigma2_0_prior`` or derived from ``covparam0_prior``.

Log-lengthscale prior
---------------------

The log-lengthscale prior is used by ``LogSigma2AndLogRhoPrior``.  It is written
in ``logrho = log(rho)`` coordinates.  Since the raw covariance vector stores
``-log(rho)``, ``logrho`` is obtained by changing sign on the lengthscale
entries of ``covparam``.

The prior combines a lower-support barrier with a linear right-tail penalty.
The barrier prevents physical lengthscales from becoming too small.  The
right-tail penalty regularizes very large physical lengthscales.

For one component, write :math:`\ell=\log(\rho)`,
:math:`\ell_{\min}=\texttt{logrho_min}`, and
:math:`\ell_0=\texttt{logrho_0}`.  The negative-log prior term is

.. math::

   f(\ell) =
   \begin{cases}
   +\infty, & \ell \leq \ell_{\min},\\
   -a\log(\ell-\ell_{\min}) + \alpha(\ell-\ell_{\min}),
   & \ell > \ell_{\min},
   \end{cases}

with

.. math::

   a = \alpha(\ell_0-\ell_{\min}).

This choice makes :math:`f` minimal at :math:`\ell_0`.  The full
lengthscale contribution is :math:`\sum_j f(\log(\rho_j))`.  Since
``covparam`` stores ``-log(rho_j)``, increasing a raw lengthscale coordinate
decreases :math:`\log(\rho_j)`.

The main fields are:

``logrho_min``
    Lower support bound in ``logrho`` coordinates.  Values at or below this
    bound have infinite negative-log prior.

``logrho_0``
    Resolved anchor.  When it is not supplied directly, it is derived from
    ``covparam0_prior`` by changing sign on the lengthscale entries.

``alpha``
    Penalty slope on the large-lengthscale side.

``rho_min_range_factor``
    Factor used to set a lower lengthscale safeguard from componentwise input
    ranges when ``logrho_min`` is not supplied.

When ``logrho_min`` is inferred from observations, each component uses

.. math::

   \texttt{logrho_min}_j = \max\left\{\log(\Delta_j), \log(r_j\,\texttt{rho_min_range_factor})\right\},

where :math:`\Delta_j` is the smallest nonzero spacing in input coordinate
:math:`j` and :math:`r_j` is the observed coordinate range.  The range-based
safeguard prevents the lower lengthscale bound from collapsing when one spacing
is much smaller than the rest of the design.

Resolution policy
-----------------

At criterion construction, the resolved prior is computed as follows:

1. ``logsigma2_0_prior`` and ``logrho_0_prior`` take precedence.
2. Missing anchors are derived from ``covparam0_prior``.
3. If ``covparam0_prior`` is missing, the anisotropic initial-guess procedure is
   run on the current observations.
4. Missing ``logrho_min`` is computed from componentwise input ranges and
   ``rho_min_range_factor``.
5. Missing scalar hyperparameters are read from
   ``gpmp.kernel.prior_defaults``.

The resolved object is stored in ``model[k]["prior"]`` and returned by
``model.get_prior(k)``.

Checks
------

Use these checks when REMAP behavior is unexpected:

.. code-block:: python

   prior = model.get_prior(0)
   covparam0_anchor = prior.covparam0
   log_sigma2_anchor = prior.log_sigma2_0
   logrho_anchor = prior.logrho_0
   logrho_min = prior.logrho_min

Compare ``prior.covparam0`` with ``model[0]["model"].covparam``.  The first is
the prior anchor.  The second is the selected covariance vector.
