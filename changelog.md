- 0.9.38 GPmp API alignment and Matérn regularity selection
  * modelcontainer.py: align diagnosis with the GPmp 0.9.38 API
  * models/models_REML.py: add Model_ConstantMean_Matern_REML
  * examples/tests: cover selected Matérn regularity with NumPy and Torch
  * docs: document model construction, parameter selection, examples, and APIs
  * packaging: require gpmp >= 0.9.38

- 0.9.37 release fixes after first PyPI publication
  * README.md: document pip install route
  * plot/visualization.py: defer optional pandas/plotly imports
  * examples/example03_sequential_prediction.py: remove accidental debugger call
  * packaging: require gpmp >= 0.9.37

- 0.9.36 documentation, packaging and model-container updates
  * models_REMAP.py: fix REMAP prior handling
  * parameter: update parameter imports
  * modelcontainer.py: add SVGD parameter sampling entry point
  * docs: add Sphinx documentation and fix documentation workflow dependencies
  * README.md: update package overview and documentation links
  * packaging: restrict published package contents and add PyPI publishing workflow
