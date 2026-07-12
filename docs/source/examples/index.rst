Examples
========

The examples are scripts in the repository ``examples/`` directory.  The pages
below document stable examples with one representative result figure, the
quantities computed by the script, and the API objects used.

Example sequence
----------------

1. ``example01`` defines computer experiments and constraints.
2. ``example02`` uses Matérn containers, selects parameters, predicts, and
   diagnoses the result.
3. ``example04`` handles noisy observations by passing observation-noise
   variances through the input matrix.
4. ``example06`` compares fixed and selected Matérn regularity.
5. ``example10`` runs expected improvement on a fixed grid.
6. ``example11`` replaces the fixed grid by an SMC particle set.
7. ``example20`` applies the reGP procedure to a threshold-oriented target.
8. ``example30`` and ``example31`` estimate excursion sets with fixed-grid and
   BSS-style search procedures.
9. ``example40`` and ``example41`` estimate inverse images of output boxes.

.. toctree::
   :maxdepth: 1

   example01_computer_experiment
   example02_models
   example04_sequential_prediction_with_noise
   example06_matern
   example10_optim_EI_gridsearch
   example11_optim_EI_smc
   example20_regp
   example30_excursionset_gridset
   example31_excursionset_smc
   example40_setinversion_gridset
   example41_setinversion_smc
