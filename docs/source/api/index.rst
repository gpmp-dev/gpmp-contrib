API reference
=============

The API reference documents the public ``gpmp-contrib`` modules.  The API uses
the numerical backend selected by ``gpmp``.  ``ModelContainer`` methods convert
inputs and return NumPy arrays by default.  Use ``gpmp.num`` when writing
backend-independent code that keeps arrays inside ``gpmp.core`` or
``gpmp.kernel`` calls.

.. toctree::
   :maxdepth: 2

   gpmpcontrib
   backend_objects
   computerexperiment
   modelcontainer
   models
   sequential
   criteria_optim
   regp
   plot
   test_problems
