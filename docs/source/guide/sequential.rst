Sequential design, optimization, and set estimation
===================================================

Sequential procedures repeatedly update a GP model and choose new evaluation
points.  A sequential procedure runs these operations:

1. Store the current observations.
2. Select or update GP parameters.
3. Compute posterior means and variances on candidates or particles.
4. Evaluate a criterion.
5. Add one or more new observations.

The implemented procedures cover optimization, excursion-set estimation, and
inverse-image estimation.

Base objects
------------

``SequentialPrediction`` stores observations and a model container.  It updates
parameters and predictions as observations are added.  Strategy classes add a
candidate set or particle set and a criterion for selecting new points.

``SequentialStrategyGridSearch`` evaluates every point in a fixed candidate
set.  It is deterministic when the candidate set can be enumerated.

``SequentialStrategySMC`` and ``SequentialStrategyBSS`` use particles when a
fixed grid would require too many candidate points or would poorly represent
the target event.

Predictive quantities used by criteria
--------------------------------------

At a candidate point :math:`x`, the current model provides

.. math::

   Z(x)\mid\mathcal{D}_n
   \sim \mathcal{N}\left(\mu_n(x), s_n^2(x)\right),

where :math:`\mathcal{D}_n` denotes the current observations.  The sequential
criteria in ``gpmp-contrib`` are functions of :math:`\mu_n(x)` and
:math:`s_n^2(x)`.

Expected improvement
--------------------

For minimization, the examples call the maximization-form EI criterion on the
negated response.  If ``z_best`` is the current observed minimum, the evaluated
criterion is

.. math::

   \mathrm{EI}(x) = \mathbb{E}\left[\max\{0, z_\mathrm{best} - Z(x)\}\right].

EI is the classical criterion used in efficient global optimization
:cite:p:`jones1998ego`.

The implemented primitive ``expected_improvement(t, zpm, zpv)`` is written for
maximization.  For :math:`Z(x)\sim\mathcal{N}(\mu,s^2)` and threshold
:math:`t`,

.. math::

   \mathrm{EI}_t(x)
   = \mathbb{E}\left[\max\{0, Z(x)-t\}\right]
   = s\left[\varphi(u) + u\Phi(u)\right],
   \qquad
   u=\frac{\mu-t}{s},

when :math:`s>0`.  The deterministic limit is
:math:`\max(0,\mu-t)` when :math:`s=0`.

Use cases:

- fixed-grid EI for one or two dimensions, or for a prescribed candidate set.
- SMC EI when the search domain is higher-dimensional and a grid would be
  inefficient.

Examples:

- :doc:`../examples/example10_optim_EI_gridsearch` for fixed-grid EI.
- :doc:`../examples/example11_optim_EI_smc` for SMC EI.

Excursion sets
--------------

For a threshold ``u``, the excursion set is

.. math::

   \Gamma_u = \{x : Z(x) > u\}.

The posterior excursion probability is

.. math::

   p_u(x)
   = \mathbb{P}\left(Z(x)>u\mid\mathcal{D}_n\right)
   = \Phi\left(\frac{\mu_n(x)-u}{s_n(x)}\right),

with the deterministic zero-variance convention used by the code.  The
misclassification probability is

.. math::

   \tau_u(x) = \min\{p_u(x), 1-p_u(x)\}.

``ExcursionSetGridSearch`` evaluates excursion probabilities and a weighted MSE
criterion on a fixed candidate set.  The criterion is large near uncertain
threshold crossings, where a new observation can change the estimated excursion
set.

The implemented weighted criterion is

.. math::

   \mathrm{wMSE}_u(x)
   = \tau_u(x)^\alpha\, \left(s_n^2(x)\right)^\beta.

``ExcursionSetBSS`` uses intermediate thresholds

.. math::

   u(\mu) = (1 - \mu)u_\mathrm{init} + \mu u_\mathrm{target}

and moves particles toward the target event.  This follows the Bayesian subset
simulation idea of using a sequence of intermediate targets and SMC particles
:cite:p:`bect2016bss`.

Examples:

- :doc:`../examples/example30_excursionset_gridset` for fixed-grid excursion
  estimation.
- :doc:`../examples/example31_excursionset_smc` for BSS-style excursion
  estimation.

Set inversion
-------------

For an output-space box ``B``, set inversion targets

.. math::

   \Gamma_B = \{x : Z(x) \in B\}.

For independent scalar output models and a box
:math:`B=[a_1,b_1]\times\cdots\times[a_q,b_q]`, the implemented box
probability is

.. math::

   p_B(x)
   = \prod_{r=1}^q
     \left[
       \Phi\left(\frac{b_r-\mu_{n,r}(x)}{s_{n,r}(x)}\right)
       -
       \Phi\left(\frac{a_r-\mu_{n,r}(x)}{s_{n,r}(x)}\right)
     \right].

The associated misclassification term is
:math:`\tau_B(x)=\min\{p_B(x),1-p_B(x)\}`.  ``box_wMSE`` combines this term
with posterior variances and sums the per-output contributions.

``SetInversionGridSearch`` computes posterior box-membership probabilities and a
box weighted MSE criterion on a fixed candidate set.  ``SetInversionBSS`` uses a
particle set and moves from an initial box to a target box.  Constrained and
multi-objective Bayesian optimization with SMC criteria is discussed by
:cite:t:`feliot2017constrained`.

Examples:

- :doc:`../examples/example40_setinversion_gridset` for fixed-grid set
  inversion.
- :doc:`../examples/example41_setinversion_smc` for BSS-style set inversion.

SMC and BSS particles
---------------------

SMC/BSS strategies maintain a particle population.  Each update reweights
particles with the change in target density, resamples when needed, and moves
particles with a Markov kernel.  The particle cloud is a numerical search
object.  It is not a posterior sample of GP covariance parameters.

For an event :math:`A_\lambda` controlled by a threshold, a box, or an
interpolation parameter, the particle target is proportional to an event
probability inside the input domain:

.. math::

   \pi_\lambda(x)
   \propto
   \mathbb{P}\left(Z(x)\in A_\lambda\mid\mathcal{D}_n\right)
   \mathbf{1}_{x\in\mathcal{X}}.

Changing :math:`\lambda` defines the sequence of intermediate targets used by
BSS-style strategies.

Which strategy to use
---------------------

Use a fixed grid when the candidate set is small enough to enumerate and when a
regular candidate set represents the region that can be evaluated.  Use an SMC
strategy when the input dimension or target geometry makes a grid inefficient.
Use BSS-style strategies when the target event is reached more reliably through
intermediate thresholds or boxes.
